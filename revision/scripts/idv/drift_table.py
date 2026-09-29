#!/usr/bin/env python3
"""Handoff 12.3: residual error, IIV and MAP-BE AUC error with a clearance change in the steady-state history
(paper scenario 1, one visit, seeds 1-3); baseline = the paper-rerun fits of the same seeds (no change, same data otherwise)."""
import os, numpy as np, pandas as pd
ROOT = "/Users/benjaminmaurel/Documents/PharmaNODE"; OUT = f"{ROOT}/results/idv/drift"
def params(f):
    d = pd.read_csv(f); return d.set_index(d.columns[0]).iloc[:, 0].astype(float)
def mapbe(f):
    x = pd.read_csv(f); r = x.auc_ipred / x.AUC_observed - 1
    return 100 * np.sqrt(np.mean(r ** 2)), 100 * np.median(np.abs(r)), 100 * np.mean(r)
print("truth: a 0.71, b 0.113, omega_CL 0.283, omega_Vc 0.316, omega_Q 0.539, omega_Vp 0.6 (sqrt of 0.08/0.10/0.29/0.36)")
print(f"{'setting':>14} {'seed':>4} | {'a':>6} {'b':>6} {'om_CL':>6} {'om_Vc':>6} {'om_Q':>6} {'om_Vp':>6} | MAP-BE: RMSPE med|e|   MPE")
for tag in ("none", "drift30_sw6", "drift50_sw6", "drift50_sw5"):
    for s in (1, 2, 3):
        if tag == "none":
            d = f"{ROOT}/results/scen_rerun/runs/s1_seed{s:03d}"; pf, mf = f"{d}/populationParameters.txt", f"{d}/tacro_mapbayest_auc_sigc1.csv"
        else:
            ID = f"{tag}_{10000 + s}"; pf, mf = f"{OUT}/popparams_{ID}.txt", f"{OUT}/mapbe_sigc1_{ID}.csv"
        if not (os.path.exists(pf) and os.path.exists(mf)): print(f"{tag:>14} {s:>4} | (missing)"); continue
        p = params(pf); rm, md, mpe = mapbe(mf)
        print(f"{tag:>14} {s:>4} | {p.get('a', np.nan):6.3f} {p.get('b', np.nan):6.3f} {p.get('omega_CL', np.nan):6.3f} "
              f"{p.get('omega_Vc', np.nan):6.3f} {p.get('omega_Q', np.nan):6.3f} {p.get('omega_Vp', np.nan):6.3f} | {rm:6.1f} {md:5.1f} {mpe:+6.1f}")
