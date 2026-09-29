#!/usr/bin/env python3
"""Summarise scenario-4 dose extrapolation (control-trained) from the eval JSONs.

    summarize_ood.py <root> <epoch6> [seeds]

Per regime (in / lo / hi) and architecture, mean +- sd over seeds of:
  nRMSE        v2.nrmse_pct (mean of per-patient ratios; robust)
  med|err|     median |pred-true|/true on the visit-2 AUC, from per_patient
  tRMSPE       RMSPE on the visit-2 AUC with the worst 1% of patients dropped
  bias         median signed relative error on the visit-2 AUC
  cov95        empirical coverage of the nominal 95% interval
Raw RMSPE is deliberately NOT reported (landmine 1.3).  The paired column is the
per-seed difference FiLM - dose-cond (negative = FiLM better)."""
import json, os, sys, numpy as np
root, ep = sys.argv[1], sys.argv[2]
seeds = [int(x) for x in (sys.argv[3].split() if len(sys.argv) > 3 else "1 2 3 4".split())]

def load(arch, s, reg):
    tag = "" if reg == "in" else reg
    f = f"{root}/eval/ep{ep}_{arch}_s{s}_vc00_on_vc00{tag}.json"
    if not os.path.exists(f): return None
    j = json.load(open(f))
    t = np.array(j["per_patient"]["true_auc_v2"]); p = np.array(j["per_patient"]["pred_auc_v2"])
    rel = (p - t) / t; a = np.abs(rel)
    keep = a <= np.quantile(a, 0.99)
    return dict(nrmse=j["v2"]["nrmse_pct"], med=100 * np.median(a),
                trmspe=100 * np.sqrt(np.mean(rel[keep] ** 2)), bias=100 * np.median(rel),
                cov95=100 * j["calibration_v2"]["intervals"]["0.95"]["coverage"],
                auc=np.median(t))

M = ("nrmse", "med", "trmspe", "bias", "cov95")
H = {"nrmse": "nRMSE", "med": "med|err|", "trmspe": "tRMSPE", "bias": "bias", "cov95": "cov95"}
print(f"Scenario 4, control-trained, V2 counterfactual AUC, {root} ep{ep}  (% ; mean +- sd over seeds)\n")
for reg, lab in (("in", "in support (1-8 mg)"), ("lo", "below (0.25, 0.5 mg)"), ("hi", "above (10, 12 mg)")):
    rows = {a: {s: load(a, s, reg) for s in seeds} for a in ("film", "dc")}
    have = [s for s in seeds if rows["film"][s] and rows["dc"][s]]
    if not have: print(f"  {lab}: no results yet\n"); continue
    med_auc = np.median([rows["dc"][s]["auc"] for s in have])
    print(f"  {lab}   n_seeds={len(have)}   median true AUC {med_auc:.1f}")
    print(f"    {'':10}" + "".join(f"{H[m]:>16}" for m in M))
    for a, name in (("film", "OT-FiLM"), ("dc", "dose-cond")):
        cells = []
        for m in M:
            v = [rows[a][s][m] for s in have]
            cells.append(f"{np.mean(v):7.2f} +-{np.std(v, ddof=1) if len(v) > 1 else 0:5.2f}")
        print(f"    {name:10}" + "".join(f"{c:>16}" for c in cells))
    d = [rows["film"][s]["nrmse"] - rows["dc"][s]["nrmse"] for s in have]
    print(f"    paired FiLM - dose-cond on nRMSE: {np.mean(d):+.2f} +- {np.std(d, ddof=1) if len(d) > 1 else 0:.2f}"
          f"   per seed {' '.join(f'{x:+.2f}' for x in d)}\n")
