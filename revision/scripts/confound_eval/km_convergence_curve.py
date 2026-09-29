#!/usr/bin/env python3
"""Held-out learning curve of the matched Km runs on 1000 new patients (km09_big, km00_big), epochs 600-3000.
Cells ep600-2100 from results/km_matched/eval_curve/ (km_curve_evals.sh), ep2400-3000 from results/km_matched/eval/.
Per architecture x training cohort: V2 median |AUC err|, level |bias| and bias-free spread, and curve nRMSE on the
control test set (km00_big), mean over 3 seeds; then the difference-in-differences per epoch (median |AUC err| and
nRMSE). Output: results/km_matched/convergence_curve.txt"""
import os, json, numpy as np
EPS = ("000600", "001200", "001800", "002100", "002400", "002700", "003000")
def path(ep, a, s, tr, ev):
    d = "results/km_matched/eval" if ep >= "002400" else "results/km_matched/eval_curve"
    return f"{d}/ep{ep}_{a}_s{s}_{tr}_on_{ev}.json"
def stats(f):
    if not os.path.exists(f): return (np.nan,) * 4
    j = json.load(open(f)); t = np.array(j["per_patient"]["true_auc_v2"]); p = np.array(j["per_patient"]["pred_auc_v2"])
    r = p / t - 1; b = np.median(r)
    return 100 * np.median(np.abs(r)), 100 * abs(b), 100 * np.median(np.abs(r - b)), j["v2"]["nrmse_pct"]
hdr = "".join(f"{int(e):>8d}" for e in EPS)
for k, name in ((0, "median |AUC err|"), (1, "|level bias|"), (2, "bias-free spread"), (3, "curve nRMSE")):
    print(f"\nV2 {name} on km00_big (control test set), mean of 3 seeds; epoch:{hdr}")
    for a in ("film", "dc"):
        for tr in ("km09", "km00"):
            v = [np.nanmean([stats(path(e, a, s, tr, "km00big"))[k] for s in (1, 2, 3)]) for e in EPS]
            print(f"  {a:4s} trained {tr}                                      " + "".join(f"{x:8.2f}" for x in v))
for k, name in ((0, "median |AUC err|"), (3, "curve nRMSE")):
    print(f"\nDiD ({name}), big test sets, mean ± sd over 3 seeds; epoch:{hdr}")
    for a in ("film", "dc"):
        cells = []
        for e in EPS:
            d = []
            for s in (1, 2, 3):
                g = lambda tr, ev: stats(path(e, a, s, tr, ev))[k]
                d.append((g("km09", "km00big") - g("km00", "km00big")) - (g("km09", "km09big") - g("km00", "km09big")))
            cells.append(f"{np.mean(d):+5.2f}±{np.std(d, ddof=1):4.2f}")
        print(f"  {a:4s} " + " ".join(f"{c:>11s}" for c in cells))
