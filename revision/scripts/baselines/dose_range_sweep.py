#!/usr/bin/env python3
"""Accuracy inside and outside the training dose range, for any of the three arms.

The counterfactual tables score each model at doses drawn from the grid it was
trained on. This asks the harder question the reviewer's extrapolation concern
implies: sweep the target dose continuously, and split the error by whether that
dose was ever administered during training. Ground truth comes from the
simulator (dose_sweep_truth.csv), so no retraining and no new cohorts are needed.

Cohort 93000 is the informative case: Michaelis--Menten elimination, trained at
2 and 3 mg only, with truth available from 1.5 to 6 mg. Under saturable kinetics
the dose--exposure relation is curved, so predicting 6 mg from 2--3 mg is a real
extrapolation rather than a rescaling.

    dose_range_sweep.py --experiment 93000 --model film --ckpt ... [--out x.npz]
"""
import argparse, os, sys
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib.read_tacro import extract_gen_tac_film, TacroFilmDataset, collate_fn_tacro_film  # noqa: E402
import lib.utils as utils  # noqa: E402
from test_dose_interpolation import predict_auc_at_dose, load_model  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", required=True)
    p.add_argument("--model", required=True, choices=["film", "dosecond", "lupk"])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", default="./results/exp_film_run")
    p.add_argument("--n-traj-samples", type=int, default=30)
    p.add_argument("--out", default=None)
    p.add_argument("--tag", type=str, default=None)
    cli = p.parse_args()
    dev = torch.device("cpu")

    tr = f"{cli.data_dir}/{cli.experiment}/virtual_cohort_film_train.csv"
    te = f"{cli.data_dir}/{cli.experiment}/virtual_cohort_film_test.csv"
    data_all, scaler = extract_gen_tac_film(file_path=[tr, te])
    train_ids = set(extract_gen_tac_film(file_path=[tr])[0].keys())
    data_eval = {k: v for k, v in data_all.items() if k not in train_ids}
    loader = DataLoader(TacroFilmDataset(data_eval), batch_size=2000, shuffle=False,
                        collate_fn=lambda x: collate_fn_tacro_film(x, dev))
    model, path = load_model(cli, dev)

    tdf = pd.read_csv(f"{cli.data_dir}/{cli.experiment}/dose_sweep_truth.csv")
    raw = pd.concat([pd.read_csv(tr), pd.read_csv(te)])
    dose_max = float(pd.to_numeric(raw["AMT"], errors="coerce").max())
    levels = sorted(pd.to_numeric(pd.read_csv(tr)["AMT"], errors="coerce").dropna().unique())
    truth = {(int(r.ID), round(float(r.DOSE), 4)): float(r.AUC) for r in tdf.itertuples()}
    sweep = sorted(tdf["DOSE"].unique())

    P, T = [], []
    with torch.no_grad():
        for b in loader:
            dense = utils.linspace_vector(b["tp_to_predict_v1"][0], torch.tensor(24.0), 100).to(dev)
            pids = [int(x) for x in b["patient_ids"]]
            P.append(np.stack([predict_auc_at_dose(model, cli.model, b, d / dose_max, dense,
                                                   scaler, cli.n_traj_samples) for d in sweep]))
            T.append(np.stack([np.array([truth[(i, round(d, 4))] for i in pids]) for d in sweep]))
    pred, true = np.concatenate(P, 1), np.concatenate(T, 1)
    if cli.out:
        np.savez(cli.out, dose_mg=np.array(sweep), pred=pred, true=true, train_levels=np.array(levels))

    rmspe = lambda t, q: float(np.sqrt(np.mean(((t - q) / t) ** 2)) * 100)
    lo, hi = min(levels), max(levels)
    name = cli.tag or cli.model
    print(f"\n{name}  --  cohort {cli.experiment}, {pred.shape[1]} patients, trained at "
          f"{'/'.join(f'{x:g}' for x in levels)} mg")
    print(f"  {'dose':>6}{'true AUC':>11}{'pred AUC':>11}{'RMSPE %':>10}   region")
    for i, d in enumerate(sweep):
        reg = "in range" if lo <= d <= hi else ("BELOW" if d < lo else "ABOVE")
        print(f"  {d:6.1f}{np.median(true[i]):11.1f}{np.median(pred[i]):11.1f}"
              f"{rmspe(true[i], pred[i]):10.2f}   {reg}")
    m_in = [i for i, d in enumerate(sweep) if lo <= d <= hi]
    m_ab = [i for i, d in enumerate(sweep) if d > hi]
    m_be = [i for i, d in enumerate(sweep) if d < lo]
    agg = lambda idx: rmspe(true[idx].ravel(), pred[idx].ravel()) if idx else float("nan")
    print(f"  {'':>6}{'in range':>22}{agg(m_in):10.2f}")
    print(f"  {'':>6}{'above range':>22}{agg(m_ab):10.2f}"
          f"   ({agg(m_ab)/agg(m_in):.2f}x the in-range error)")
    if m_be:
        print(f"  {'':>6}{'below range':>22}{agg(m_be):10.2f}"
              f"   ({agg(m_be)/agg(m_in):.2f}x)")


if __name__ == "__main__":
    main()
