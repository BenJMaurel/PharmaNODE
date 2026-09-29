#!/usr/bin/env python3
"""Handoff 12.3 sigma sweep: MAP-BE AUC error when the residual SD given to MAP-BE is scaled by lambda (same Monolix fit).
lambda < 1 = trust the 3 samples more than the fitted error model says; > 1 = lean more on the prior.
Per run: RMSPE, median |err|, MPE over the 40 test patients; then mean over runs. Best lambda per group marked *."""
import os, numpy as np, pandas as pd
ROOT = "/Users/benjaminmaurel/Documents/PharmaNODE"; SW = f"{ROOT}/results/idv/sigsweep"
LAMS = ["0.5", "0.7", "1", "1.4", "2.0"]
def lam1(tag):
    if tag.startswith("s") and "_seed" in tag: return f"{ROOT}/results/scen_rerun/runs/{tag}/tacro_mapbayest_auc_sigc1.csv"
    if tag.startswith("idv"): return f"{ROOT}/results/idv/check/mapbe_sigc1_{tag}.csv"
    return f"{ROOT}/results/idv/drift/mapbe_sigc1_{tag}.csv"
def err(f):
    if not os.path.exists(f): return None
    x = pd.read_csv(f); r = (x.auc_ipred / x.AUC_observed - 1).values
    return 100 * np.sqrt(np.mean(r ** 2)), 100 * np.median(np.abs(r)), 100 * np.mean(r)
GROUPS = [(f"s{sc}", [f"s{sc}_seed{s:03d}" for s in range(1, 11)]) for sc in (1, 2, 3)] + \
         [(g, [f"{g}_{10000 + s}" for s in (1, 2, 3)]) for g in ("idv25", "idv50", "drift30_sw6", "drift50_sw6", "drift50_sw5")]
for metric, k in (("RMSPE", 0), ("median |AUC err|", 1), ("MPE", 2)):
    print(f"\n{metric} (mean over runs)      lambda: " + "  ".join(f"{l:>6}" for l in LAMS) + "   runs")
    for g, tags in GROUPS:
        vals, n = [], 0
        for l in LAMS:
            v = [err(lam1(t) if l == "1" else f"{SW}/{t}/lam{l}/tacro_mapbayest_auc_lam{l}.csv") for t in tags]
            v = [e[k] for e in v if e is not None]; vals.append(np.mean(v) if v else np.nan); n = max(n, len(v))
        best = int(np.nanargmin(np.abs(vals) if k == 2 else vals)) if not all(np.isnan(vals)) else -1
        print(f"  {g:>12}                    " + "  ".join(f"{v:5.2f}{'*' if i == best else ' '}" for i, v in enumerate(vals)) + f"   {n}")
