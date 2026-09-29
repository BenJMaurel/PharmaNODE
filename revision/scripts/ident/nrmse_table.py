#!/usr/bin/env python3
"""V1/V2 curve nRMSE for every estimator, in range and out of range, scenario 4 control.

nRMSE per patient = RMSE over the 12 target times / mean observed concentration, against
the NOISY observed curve (the harness definition); reported as the MEAN over patients (as
the harness does) and the MEDIAN (robust).  Estimators:
  true parameters      the noise floor
  linear rescaling     the TRUE (noise-free) V1 curve x d2/d1 -- what linear PK promises
  EBE / posterior mean the true population model (ebe_s4_batch.py / bayes_s4.py)
  networks             4-seed harness JSONs over all 1000 patients, plus seed 1 on the
                       same 300 patients as the EBE (from curves_s4.py) for a paired view
"""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, 'scripts/ident'); sys.path.insert(0, '.')
from ebe_s4_batch import simulate, FIT
from mle_km_s4 import replay
REL = np.array([0., 0.33, 0.67, 1., 1.5, 2., 3., 4., 6., 9., 12., 24.])
TP = replay(1800, 4)
out = {}
for reg, tag in (("", "in range 1-8 mg"), ("_lo", "below 0.25-0.5 mg"), ("_hi", "above 10-12 mg")):
    f = "results/s4/ebe/bayes_vc00_s4_n300.csv" if reg == "" else f"results/s4/ebe/ebe_vc00_s4{reg}_n300.csv"
    d = pd.read_csv(f)
    ob = pd.read_csv(f"results/exp_film_run/confound_vc00_s4{reg}/virtual_cohort_film_test.csv")
    ob["DVn"] = pd.to_numeric(ob.DV, errors="coerce"); ob["Tn"] = pd.to_numeric(ob.TIME, errors="coerce")
    lin = np.full(len(d), np.nan)
    for form in ("Advagraf", "Prograf"):
        m = (d.form == form).values
        if not m.any(): continue
        ids = d.ID.values[m]
        th = np.array([[float(TP.loc[i][n]) for n in FIT] for i in ids]); cl = np.array([float(TP.loc[i].CL) for i in ids])
        c1 = simulate(th, cl, d.d1.values[m], form, REL).T                     # true V1 curve, no noise
        for k, i in enumerate(ids):
            y2 = ob[(ob.ID == i) & (ob.VISIT == 2) & ob.DVn.notna()].sort_values("Tn").DVn.values
            q = c1[k] * d.d2.values[m][k] / d.d1.values[m][k]
            lin[np.where(m)[0][k]] = np.sqrt(np.mean((q - y2) ** 2)) / np.mean(y2)
    rows = [("true parameters (noise floor)", d.nrmse1_replay, d.nrmse2_replay),
            ("linear rescaling of true V1", None, pd.Series(lin)),
            ("EBE, true population model", d.nrmse1_ebe, d.nrmse2_ebe)]
    if "nrmse2_pm" in d: rows.append(("posterior mean, true model", None, d.nrmse2_pm))
    rows.append(("population typical values", d.nrmse1_prior, d.nrmse2_prior))
    z = np.load(f"results/s4/curves/ep006000_s1_vc00_on_vc00{reg[1:] if reg else ''}.npz")
    pos = {int(i): k for k, i in enumerate(z["ids"])}; sel = [pos[int(i)] for i in d.ID]
    for arch, name in (("film", "OT-FiLM"), ("dc", "dose-cond")):
        rows.append((f"{name}, seed 1, same 300", None, pd.Series(z[f"{arch}_nrmse"][sel])))
    net = {}
    for arch, name in (("film", "OT-FiLM"), ("dc", "dose-cond")):
        v1 = [json.load(open(f"results/s4/eval/ep006000_{arch}_s{s}_vc00_on_vc00{reg[1:] if reg else ''}.json"))["v1"]["nrmse_pct"] for s in (1, 2, 3, 4)]
        v2 = [json.load(open(f"results/s4/eval/ep006000_{arch}_s{s}_vc00_on_vc00{reg[1:] if reg else ''}.json"))["v2"]["nrmse_pct"] for s in (1, 2, 3, 4)]
        net[name] = (np.mean(v1), np.std(v1, ddof=1), np.mean(v2), np.std(v2, ddof=1))
    print(f"\n== {tag}  (EBE rows: 300 patients; nRMSE %, mean / median over patients)")
    print(f"   {'estimator':34}{'V1 mean':>9}{'V1 med':>8}{'V2 mean':>10}{'V2 med':>8}")
    for name, a, b in rows:
        s1 = f"{100*a.mean():9.2f}{100*a.median():8.2f}" if a is not None else f"{'—':>9}{'—':>8}"
        print(f"   {name:34}{s1}{100*b.mean():10.2f}{100*b.median():8.2f}")
    for name, (m1, s1, m2, s2) in net.items():
        print(f"   {name + ', 4 seeds, all 1000':34}{m1:9.2f}{'':8}{m2:10.2f}   (sd {s1:.2f} / {s2:.2f})")
