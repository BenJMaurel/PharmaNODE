#!/usr/bin/env python3
"""Is the dose confined to the transport direction, leaving physiology elsewhere?

Reviewer 1 asked that latent analyses be quantitative rather than pictorial.
This is the direct test. The FiLM operator displaces the baseline state by
    delta_i = z_new_i - z_base_i ,
so the transport acts, on average, along a single direction u = mean(delta)/||mean(delta)||.
Projecting that direction out of the baseline state leaves

    z_orth = z_base - (z_base . u) u

and the question is what survives. If the dose is carried by the transport
direction, decoding the administered dose from z_orth should collapse, while
decoding the withheld physiological parameters should not.

Reported as cross-validated linear R2 from z_base and from z_orth, with the
residual direction count as a control (removing one of D dimensions cannot by
itself destroy information that is spread over the rest).

    orthogonal_probe.py --experiment 90000 --ckpt <film ckpt>
"""
import argparse, os, sys
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib.read_tacro import extract_gen_tac_film, TacroFilmDataset, collate_fn_tacro_film  # noqa: E402
from probe_entanglement import load_film_model, encode_mu, film_transport  # noqa: E402

TARGETS = ["K_ELIM", "HT", "K_12", "K_21"]


def build_targets(tr, te, data_dir, experiment, pids):
    """Ground-truth per-patient parameters.

    The cohort CSV carries the rate constants (K_ELIM = CL/Vc, K_12 = Q/Vc,
    K_21 = Q/Vp) and hematocrit. Rate constants mix a clearance with a volume,
    so a probe on them is hard to read. Where the cohort also ships
    confound_truth.csv we recover the structural parameters themselves:

        Vc = CL / K_ELIM ,   Q = K_12 * Vc ,   Vp = Q / K_21 ,

    and add the clearance and the saturation constant Km directly.
    """
    raw = pd.concat([pd.read_csv(tr), pd.read_csv(te)])
    t = raw.groupby("ID")[TARGETS].first()
    out = {c: t.loc[pids, c].values.astype(float) for c in TARGETS}
    ct = os.path.join(data_dir, experiment, "confound_truth.csv")
    if os.path.exists(ct):
        c = pd.read_csv(ct).set_index("ID")
        CL = c.loc[pids, "CL_base"].values.astype(float)
        out["CL"] = CL
        out["Vc"] = CL / out["K_ELIM"]
        out["Q"] = out["K_12"] * out["Vc"]
        out["Vp"] = out["Q"] / out["K_21"]
        if str(c["confound_param"].iloc[0]) == "Km":
            out["Km"] = c.loc[pids, "confound_par"].values.astype(float)
    return out


def probe(X, y, folds=5):
    m = RidgeCV(alphas=np.logspace(-3, 3, 13))
    return float(r2_score(y, cross_val_predict(m, X, y, cv=folds)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", default="./results/exp_film_run")
    p.add_argument("--n-directions", type=int, default=1,
                   help="How many transport directions to project out (1 = the mean displacement).")
    a = p.parse_args()
    dev = torch.device("cpu")
    tr = f"{a.data_dir}/{a.experiment}/virtual_cohort_film_train.csv"
    te = f"{a.data_dir}/{a.experiment}/virtual_cohort_film_test.csv"
    data, _ = extract_gen_tac_film(file_path=[tr, te])
    b = next(iter(DataLoader(TacroFilmDataset(data), batch_size=4000, shuffle=False,
                             collate_fn=lambda x: collate_fn_tacro_film(x, dev))))
    m, _ = load_film_model(a.ckpt, dev)
    with torch.no_grad():
        z = encode_mu(m, b["observed_data_v1"], b["observed_tp_v1"], b["dose_v1"], b["static_v1"])
        zn = film_transport(m, z, b["dose_v1"], b["dose_v2"], b["delta_t"], b["t_v1"])
    z = z.cpu().numpy(); delta = (zn - z).cpu().numpy()

    # The transport subspace is the principal AXIS of the displacements, not their
    # mean. A dose increase and a dose decrease move the state in opposite
    # directions along the same axis, so the displacements are antiparallel across
    # dose pairs (median angle between dose-pair mean directions: 173 deg) and their
    # mean very nearly cancels. Taking the mean would project out a direction that
    # carries no dose information at all. The uncentred SVD is sign-agnostic and
    # recovers the axis: its first component holds ~99% of the displacement energy.
    _, sv, Vt = np.linalg.svd(delta, full_matrices=False)
    U = Vt[: a.n_directions]
    print(f"  displacement energy on the {a.n_directions} retained axis/axes: "
          f"{100 * (sv[:a.n_directions] ** 2).sum() / (sv ** 2).sum():.1f}%")
    Q, _ = np.linalg.qr(U.T)                       # orthonormal basis of the transport subspace
    z_orth = z - (z @ Q) @ Q.T
    # control: remove an equal number of random directions instead
    rng = np.random.default_rng(0)
    R = np.linalg.qr(rng.normal(size=(z.shape[1], Q.shape[1])))[0]
    z_rand = z - (z @ R) @ R.T

    pids = [int(x) for x in b["patient_ids"]]
    Y = build_targets(tr, te, a.data_dir, a.experiment, pids)
    dose = b["dose_v1"].cpu().numpy().ravel()

    print(f"\ncohort {a.experiment}, {len(pids)} patients, latent dim {z.shape[1]}, "
          f"{Q.shape[1]} transport direction(s) removed")
    print(f"{'target':<18}{'z_base':>10}{'z_orth':>10}{'retained':>11}   {'z_rand (control)':>17}")
    order = ["Km", "CL", "Vc", "Vp", "Q", "K_ELIM", "HT", "K_12", "K_21"]
    rows = [("administered dose", dose)] + [(t, Y[t]) for t in order if t in Y]
    for name, y in rows:
        if np.std(y) == 0:
            continue
        b0, bo, br = probe(z, y), probe(z_orth, y), probe(z_rand, y)
        keep = f"{100*max(bo,0)/max(b0,1e-9):5.0f}%" if b0 > 0.05 else "    --"
        print(f"{name:<18}{b0:>10.3f}{bo:>10.3f}{keep:>11}   {br:>17.3f}")


if __name__ == "__main__":
    main()
