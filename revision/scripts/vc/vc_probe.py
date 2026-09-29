#!/usr/bin/env python3
"""How much of log Vc does the latent carry, and does confounded training erode it?

Unlike Km, Vc IS inferable from a single-dose curve, so a legitimate route exists.
The decisive read-out is both models evaluated on the CONTROL held-out cohort, where
the dose carries no Vc information: any gap there is what the encoder failed to learn
to read off the curve because the dose was available during training.

Run from the repo root:
    python3 scripts/vc/vc_probe.py --family dc --seed 2 --epoch 6000
"""
import argparse, glob, os, sys
import numpy as np, torch, pandas as pd
from torch.utils.data import DataLoader
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score
sys.path.insert(0, '.')
from lib.read_tacro import extract_gen_tac_film, TacroFilmDataset, collate_fn_tacro_film
from probe_entanglement import load_film_model, load_dosecond_model, encode_mu

dev = torch.device("cpu")


def probe(X, y):
    X = np.asarray(X).reshape(len(y), -1)
    return float(r2_score(y, cross_val_predict(
        RidgeCV(alphas=np.logspace(-3, 3, 13)), X, y, cv=5)))


def latents(ckpt, exp, family):
    d = f"results/exp_film_run/{exp}"
    tr_file = f"{d}/virtual_cohort_film_train.csv"
    train_ids = set()
    if os.path.exists(tr_file) and len(pd.read_csv(tr_file)) > 0:
        train_ids = set(extract_gen_tac_film(file_path=[tr_file])[0].keys())
    ev, _ = extract_gen_tac_film(file_path=[f"{d}/virtual_cohort_film_test.csv"])
    ev = {k: v for k, v in ev.items() if k not in train_ids}
    ids = list(ev.keys())
    b = next(iter(DataLoader(TacroFilmDataset(ev), batch_size=4000, shuffle=False,
                             collate_fn=lambda x: collate_fn_tacro_film(x, dev))))
    m, _ = (load_film_model if family == "film" else load_dosecond_model)(ckpt, dev)
    with torch.no_grad():
        if family == "film":
            z1 = encode_mu(m, b["observed_data_v1"], b["observed_tp_v1"], b["dose_v1"], b["static_v1"])
            z2 = encode_mu(m, b["observed_data_v2"], b["observed_tp_v2"], b["dose_v2"], b["static_v2"])
        else:  # use the model's own encode: it honours encoder_dose='zero'
            z1 = m.encode(b["observed_data_v1"], b["observed_tp_v1"], b["dose_v1"], b["static_v1"])[0].squeeze(0)
            z2 = m.encode(b["observed_data_v2"], b["observed_tp_v2"], b["dose_v2"], b["static_v2"])[0].squeeze(0)
    t = pd.read_csv(f"{d}/confound_truth.csv").set_index("ID")
    y = np.log(t.loc[ids, "confound_par"].values)
    return z1.numpy(), z2.numpy(), y, b["dose_v1"].numpy().ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="dc", choices=("dc", "film"))
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--epoch", default="3000")
    ap.add_argument("--suffix", default="_big", help="evaluation cohort suffix")
    a = ap.parse_args()
    sub, run = ((f"dc_s{a.seed}", "exp_dosecond_run") if a.family == "dc"
                else (f"film_s{a.seed}", "exp_film_run"))
    print(f"\nR^2 for log Vc, 5-fold cross-validated ridge probe  "
          f"({a.family} seed {a.seed}, ep{a.epoch}; Vc never shown to any model)\n")
    print(f"  {'trained on':<13}{'evaluated on':<15}{'d1 alone':>9}{'z_v1':>8}{'z_v2':>8}{'z_v1 | d1 out':>15}")
    out = {}
    for trained, tag in (("confound_vc09", "confounded"), ("confound_vc00", "control")):
        pat = f"results/vc/{sub}/{run}/{trained}/traj/*_ep{int(a.epoch):06d}.ckpt"
        hits = sorted(glob.glob(pat))
        if len(hits) != 1:
            raise FileNotFoundError(f"{pat}: found {len(hits)}")
        for ev, evtag in ((f"confound_vc09{a.suffix}", "confounded"),
                          (f"confound_vc00{a.suffix}", "control")):
            z1, z2, y, d1 = latents(hits[0], ev, a.family)
            z1r = z1 - LinearRegression().fit(d1.reshape(-1, 1), z1).predict(d1.reshape(-1, 1))
            r = (probe(d1, y), probe(z1, y), probe(z2, y), probe(z1r, y))
            out[(tag, evtag)] = r
            print(f"  {tag:<13}{evtag:<15}{r[0]:>9.3f}{r[1]:>8.3f}{r[2]:>8.3f}{r[3]:>15.3f}")
    cc, ct = out[("control", "control")][1], out[("confounded", "control")][1]
    print(f"\n  on the CONTROL held-out set (dose carries no Vc information):")
    print(f"    control-trained {cc:.3f}   confounded-trained {ct:.3f}   erosion {cc - ct:+.3f}")


if __name__ == "__main__":
    main()
