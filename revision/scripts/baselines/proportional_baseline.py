#!/usr/bin/env python3
"""Dose-proportional exposure baselines for the counterfactual task.

Reviewer 1 asked for comparison against "simple AUC/exposure prediction
baselines".  The simplest defensible rule for predicting exposure at a
counterfactual dose is linear scaling of the observed exposure,

    AUC_2_hat = AUC_1_hat * (d_2 / d_1),

which is exactly correct whenever clearance is dose-independent.  Two variants
are reported, and the distinction matters:

  oracle   AUC_1 is the simulator's true visit-1 AUC.  No model could do better
           with a proportional rule, so this is a ceiling rather than a method.
           Under linear PK it is exact by construction and its error is zero;
           under Michaelis-Menten elimination its error is the cost of ignoring
           saturation, which is precisely what our model has to earn.

  sparse   AUC_1 is a trapezoid over the same 3-ish sparse observations the
           model is given, integrated to the formulation's cutoff.  This is what
           a clinician without a PK model could compute, and it is the baseline
           our method actually has to beat.

Metrics match the evaluation harness exactly: MPE = mean((true-pred)/true),
RMSPE = sqrt(mean(((true-pred)/true)^2)), both in per cent, on the test split.

    proportional_baseline.py --experiments 90000 93000 confound_km09 ...
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib.read_tacro import extract_gen_tac_film, TacroFilmDataset, collate_fn_tacro_film  # noqa: E402
from test_dose_cond import unscale_data  # noqa: E402


def mpe(t, p):
    return float(np.mean((t - p) / t) * 100)


def rmspe(t, p):
    return float(np.sqrt(np.mean(((t - p) / t) ** 2)) * 100)


def sparse_auc(obs, tps, static, scaler):
    """Trapezoid over the sparse visit-1 samples, to 12 h (Prograf) or 24 h (Advagraf)."""
    conc = unscale_data(obs.squeeze(-1), scaler)
    t = tps.cpu().numpy()
    out = []
    for i in range(conc.shape[0]):
        cutoff = 12.0 if bool(static[i, 1].item()) else 24.0
        keep = (t <= cutoff) & (conc[i] > 0)
        out.append(np.trapezoid(conc[i][keep], t[keep]) if keep.sum() >= 2 else np.nan)
    return np.array(out)


def run(exp, data_dir, device):
    tr = os.path.join(data_dir, exp, "virtual_cohort_film_train.csv")
    te = os.path.join(data_dir, exp, "virtual_cohort_film_test.csv")
    allp, scaler = extract_gen_tac_film(file_path=[tr, te])
    train_ids = set(extract_gen_tac_film(file_path=[tr])[0].keys())
    test = {k: v for k, v in allp.items() if k not in train_ids}
    loader = DataLoader(TacroFilmDataset(test), batch_size=4000, shuffle=False,
                        collate_fn=lambda x: collate_fn_tacro_film(x, device))
    b = next(iter(loader))
    a1 = (b["auc_red_v1"] * scaler[0]).cpu().numpy()
    a2 = (b["auc_red_v2"] * scaler[0]).cpu().numpy()
    ratio = (b["dose_v2"] / b["dose_v1"]).cpu().numpy()      # normalisation cancels
    a1_sparse = sparse_auc(b["observed_data_v1"], b["observed_tp_v1"], b["static_v1"], scaler)
    ok = np.isfinite(a1_sparse) & (a1_sparse > 0)
    return {
        "n": len(a1), "n_sparse": int(ok.sum()),
        "oracle_mpe": mpe(a2, a1 * ratio), "oracle_rmspe": rmspe(a2, a1 * ratio),
        "sparse_mpe": mpe(a2[ok], a1_sparse[ok] * ratio[ok]),
        "sparse_rmspe": rmspe(a2[ok], a1_sparse[ok] * ratio[ok]),
        "sparse_v1_rmspe": rmspe(a1[ok], a1_sparse[ok]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments", nargs="+", required=True)
    ap.add_argument("--data-dir", default="./results/exp_film_run")
    a = ap.parse_args()
    dev = torch.device("cpu")
    print(f"{'cohort':<18}{'n':>5}{'oracle x d2/d1':>18}{'sparse x d2/d1':>18}{'sparse V1 AUC':>16}")
    print(f"{'':<18}{'':>5}{'RMSPE   (MPE)':>18}{'RMSPE   (MPE)':>18}{'RMSPE':>16}")
    print("-" * 75)
    for e in a.experiments:
        r = run(e, a.data_dir, dev)
        print(f"{e:<18}{r['n']:>5}"
              f"{r['oracle_rmspe']:>10.2f} ({r['oracle_mpe']:+5.1f})"
              f"{r['sparse_rmspe']:>11.2f} ({r['sparse_mpe']:+5.1f})"
              f"{r['sparse_v1_rmspe']:>15.2f}")


if __name__ == "__main__":
    main()
