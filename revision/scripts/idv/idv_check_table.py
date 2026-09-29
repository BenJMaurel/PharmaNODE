#!/usr/bin/env python3
"""Handoff 12 checkpoint table: residual error, IIV and MAP-BE AUC error vs dose-to-dose bioavailability variability
(kappa), paper scenario 1, seeds 1-3; kappa = 0 = the existing paper-rerun fits of the same seeds (same data otherwise)."""
import os, numpy as np, pandas as pd
ROOT = "/Users/benjaminmaurel/Documents/PharmaNODE"; OUT = f"{ROOT}/results/idv/check"
def params(f):
    d = pd.read_csv(f); return d.set_index(d.columns[0]).iloc[:, 0].astype(float)
def mapbe(f):
    x = pd.read_csv(f); r = x.auc_ipred / x.AUC_observed - 1
    return 100 * np.sqrt(np.mean(r ** 2)), 100 * np.median(np.abs(r)), 100 * np.mean(r)
print("truth: a 0.71, b 0.113, omega_CL 0.283 (sqrt 0.08), omega_KTR 0.245 (sqrt 0.06)")
print(f"{'kappa':>5} {'seed':>4} | {'a':>6} {'b':>6} {'om_CL':>6} {'om_KTR':>6} {'om_Vc':>6} | MAP-BE AUC: RMSPE  med|e|   MPE")
for K in (0, 25, 50):
    for s in (1, 2, 3):
        if K == 0:
            d = f"{ROOT}/results/scen_rerun/runs/s1_seed{s:03d}"
            pf, mf = f"{d}/populationParameters.txt", f"{d}/tacro_mapbayest_auc_sigc1.csv"
        else:
            ID = f"idv{K}_{10000 + s}"; pf, mf = f"{OUT}/popparams_{ID}.txt", f"{OUT}/mapbe_sigc1_{ID}.csv"
        if not (os.path.exists(pf) and os.path.exists(mf)): print(f"{K/100:5.2f} {s:>4} | (missing)"); continue
        p = params(pf); rm, md, mpe = mapbe(mf)
        print(f"{K/100:5.2f} {s:>4} | {p.get('a', np.nan):6.3f} {p.get('b', np.nan):6.3f} {p.get('omega_CL', np.nan):6.3f} "
              f"{p.get('omega_KTR', np.nan):6.3f} {p.get('omega_Vc', np.nan):6.3f} |        {rm:6.1f} {md:6.1f} {mpe:+6.1f}")
