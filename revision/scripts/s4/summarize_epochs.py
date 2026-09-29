#!/usr/bin/env python3
"""Scenario-4 read-out across epochs for one architecture.

    summarize_epochs.py <root> <arch film|dc> "<epochs6>" "<seeds>"

Table 1: 2x2 V2 nRMSE (train {ctrl,conf} x test {ctrl,conf}) per seed and epoch.
Table 2: control-trained, test on control: V1 / V2 nRMSE and V2 median |AUC err|
         in support (1-8 mg), below (0.25, 0.5) and above (10, 12 mg).
nRMSE = v2.nrmse_pct from the harness (mean over patients of RMSE / mean conc).
Raw AUC RMSPE deliberately not reported (landmine 1.3)."""
import json, os, sys, numpy as np
root, arch = sys.argv[1], sys.argv[2]
eps, seeds = sys.argv[3].split(), sys.argv[4].split()

def J(ep, s, tr, te, reg=""):
    f = f"{root}/eval/ep{ep}_{arch}_s{s}_{tr}_on_{te}{reg}.json"
    return json.load(open(f)) if os.path.exists(f) else None

def med(j, v="v2"):
    t = np.array(j["per_patient"][f"true_auc_{v}"]); p = np.array(j["per_patient"][f"pred_auc_{v}"])
    return 100 * np.median(np.abs(p - t) / t)

f = lambda x: "   --  " if x is None else f"{x:7.2f}"
print(f"{root}  {arch}  V2 nRMSE %, 2x2 (first = trained on, second = tested on)")
print(f"{'epoch':>7} {'seed':>4} {'ctrl>ctrl':>10} {'ctrl>conf':>10} {'conf>ctrl':>10} {'conf>conf':>10}")
for ep in eps:
    for s in seeds:
        c = [J(ep, s, tr, te) for tr, te in (("vc00","vc00"),("vc00","vc09"),("vc09","vc00"),("vc09","vc09"))]
        print(f"{int(ep):>7} {s:>4} " + " ".join(f"{f(j['v2']['nrmse_pct'] if j else None):>10}" for j in c))
print()
print("control-trained, control-tested:  nRMSE %  and  V2 median |AUC err| %")
print(f"{'epoch':>7} {'seed':>4} {'V1 nRMSE':>9} {'V2 in':>7} {'V2 lo':>7} {'V2 hi':>7} | {'med in':>7} {'med lo':>7} {'med hi':>7} {'med V1':>7}")
for ep in eps:
    for s in seeds:
        i, lo, hi = (J(ep, s, "vc00", "vc00", r) for r in ("", "lo", "hi"))
        print(f"{int(ep):>7} {s:>4} {f(i['v1']['nrmse_pct'] if i else None):>9} "
              + " ".join(f(j['v2']['nrmse_pct'] if j else None) for j in (i, lo, hi)) + " | "
              + " ".join(f(med(j) if j else None) for j in (i, lo, hi)) + " " + f(med(i, 'v1') if i else None))
