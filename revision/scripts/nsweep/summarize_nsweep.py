#!/usr/bin/env python3
"""Patient-count sweep: networks vs the Monolix-estimated EBE vs the true-model EBE,
all scored on the SAME 1000 test patients of confound_vc00_s4 (medians and means over the
full set, so no row alignment is involved).

    summarize_nsweep.py <sweep_epoch6> [seeds]
      networks N=100/200/400 : results/s4_nsweep/eval/ep<EP>_<arch>_n<N>_s<seed>_on_<reg>.json
      networks N=800         : results/s4/eval, ep006000 (6000 epochs x 2 steps = 12000 steps,
                               matched in optimiser steps to the sweep's ep012000)
      Monolix EBE            : results/s4/ebe/popebe_mlx_n<N>_n1000.csv   (N = training size)
      true-model EBE         : results/s4/ebe/popebe_true_n1000.csv
Metrics on the V2 counterfactual: median |AUC error| and mean curve nRMSE (harness definition).
"""
import os, sys, json, numpy as np, pandas as pd
ROOT = os.environ.get("NSWEEP_ROOT", "results/s4_nsweep")
EP = sys.argv[1]; SEEDS = [int(s) for s in (sys.argv[2].split() if len(sys.argv) > 2 else ["1"])]
SIZES = (100, 200, 400, 800)

def net(arch, n, reg):
    files = ([f"{ROOT}/eval/ep{EP}_{arch}_n{n}_s{s}_on_{reg}.json" for s in SEEDS] if n != 800 else
             [f"results/s4/eval/ep006000_{arch}_s{s}_vc00_on_vc00{'' if reg == 'in' else reg}.json" for s in (1, 2, 3, 4)])
    out = []
    for f in files:
        if not os.path.exists(f): continue
        j = json.load(open(f)); t = np.array(j["per_patient"]["true_auc_v2"]); p = np.array(j["per_patient"]["pred_auc_v2"])
        tv1 = np.array(j["per_patient"]["true_auc_v1"]); pv1 = np.array(j["per_patient"]["pred_auc_v1"])
        out.append((100 * np.median(np.abs(pv1 / tv1 - 1)), 100 * np.median(np.abs(p / t - 1)), j["v2"]["nrmse_pct"]))
    return np.mean(out, 0) if out else None, len(out)

def ebe(f, reg):
    if not os.path.exists(f): return None
    d = pd.read_csv(f)
    return (100 * np.median(np.abs(d.auc_v1 / d.auc_true_v1 - 1)), 100 * np.median(np.abs(d[f"auc_{reg}"] / d[f"auc_true_{reg}"] - 1)),
            100 * d[f"nrmse_{reg}"].mean())

fmt = lambda r: "      —      " if r is None else f"{r[1]:5.1f} / {r[2]:5.1f}"
for reg, lab in (("in", "V2 in range (1-8 mg)"), ("hi", "V2 above range (10-12 mg)"), ("lo", "V2 below range (0.25-0.5 mg)")):
    tr = ebe("results/s4/ebe/popebe_true_n1000.csv", reg)
    print(f"\n{lab}: median |AUC err| / mean nRMSE  (%)   true-model EBE floor: {fmt(tr)}")
    print(f"  {'N train':>8}  {'Monolix EBE':>14}  {'OT-FiLM':>14}  {'dose-cond':>14}")
    for n in SIZES:
        m = ebe(f"results/s4/ebe/popebe_mlx_n{n}_n1000.csv", reg); fi, kf = net("film", n, reg); dc, kd = net("dc", n, reg)
        print(f"  {n:>8}  {fmt(m):>14}  {fmt(fi):>14}  {fmt(dc):>14}   (seeds {kf}/{kd})")
tr = ebe("results/s4/ebe/popebe_true_n1000.csv", "in")
print(f"\nV1 (factual) median |AUC err|: true-model EBE {tr[0]:.1f}%" if tr else "")
for n in SIZES:
    m = ebe(f"results/s4/ebe/popebe_mlx_n{n}_n1000.csv", "in"); fi, _ = net("film", n, "in"); dc, _ = net("dc", n, "in")
    print(f"  N={n:<4} Monolix EBE {m[0] if m else float('nan'):5.1f}   FiLM {fi[0] if fi is not None else float('nan'):5.1f}   dose-cond {dc[0] if dc is not None else float('nan'):5.1f}")
