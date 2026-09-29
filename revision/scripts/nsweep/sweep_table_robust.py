#!/usr/bin/env python3
"""Patient-count sweep, robust read-out (scenario 4, control cohort, same 1000 test patients).
Networks: NSWEEP_ROOT (default results/s4_nsweep2) ep<EP> seed 1 for N=100/200/400; N=800 from
results/s4 at ep006000 (same epochs) and ep003000 (same optimiser steps as N<=400 ep6000), seed 1
and the 4-seed mean.  Monolix EBE (true covariate model, and L1 = no HT on Km, no Km-Vc correlation) / true-model EBE
from results/s4/ebe.
Per cell: median |AUC err| %, median signed err (bias) %, mean nRMSE %, n(|err| > 100%).
    sweep_table_robust.py [EP]"""
import os, sys, json, numpy as np, pandas as pd
EP = sys.argv[1] if len(sys.argv) > 1 else "006000"
ROOT = os.environ.get("NSWEEP_ROOT", "results/s4_nsweep2")
def stats_js(f, reg):
    if not os.path.exists(f): return None
    j = json.load(open(f)); t = np.array(j["per_patient"]["true_auc_v2"]); p = np.array(j["per_patient"]["pred_auc_v2"])
    r = p / t - 1; a = np.abs(r)
    tv = np.array(j["per_patient"]["true_auc_v1"]); pv = np.array(j["per_patient"]["pred_auc_v1"])
    return dict(med=100*np.median(a), bias=100*np.median(r), nrmse=j["v2"]["nrmse_pct"], blow=int(np.sum(a > 1)),
                v1=100*np.median(np.abs(pv/tv-1)))
def stats_csv(f, reg):
    if not os.path.exists(f): return None
    d = pd.read_csv(f); r = d[f"auc_{reg}"] / d[f"auc_true_{reg}"] - 1
    return dict(med=100*np.median(np.abs(r)), bias=100*np.median(r), nrmse=100*d[f"nrmse_{reg}"].mean(),
                blow=int(np.sum(np.abs(r) > 1)), v1=100*np.median(np.abs(d.auc_v1/d.auc_true_v1-1)))
def net(arch, n, reg, which="s1"):
    tag = "" if reg == "in" else reg
    if n != 800: return stats_js(f"{ROOT}/eval/ep{EP}_{arch}_n{n}_s1_on_{reg}.json", reg)
    ep = "006000" if which in ("s1", "4s") else "003000"
    seeds = (1, 2, 3, 4) if which == "4s" else (1,)
    ss = [stats_js(f"results/s4/eval/ep{ep}_{arch}_s{s}_vc00_on_vc00{tag}.json", reg) for s in seeds]
    ss = [s for s in ss if s]
    if not ss: return None
    return {k: (np.mean([s[k] for s in ss]) if k != "blow" else int(round(np.mean([s[k] for s in ss])))) for k in ss[0]}
f = lambda s: f"{'—':>22}" if s is None else f"{s['med']:5.1f} ({s['bias']:+5.1f}) {s['nrmse']:6.1f} [{s['blow']:>2}]"
print("cell = median |AUC err| (median signed err)  mean nRMSE  [n |err|>100%], V2, 1000 test patients")
for reg, lab in (("in", "in range 1-8 mg"), ("hi", "above range 10-12 mg"), ("lo", "below range 0.25-0.5 mg")):
    print(f"\n{lab}.  true-model EBE floor: {f(stats_csv('results/s4/ebe/popebe_true_n1000.csv', reg))}")
    print(f"  {'N':>14} {'Monolix EBE (true)':>22} {'Monolix EBE (L1)':>22} {'OT-FiLM':>22} {'dose-cond':>22}")
    for n, which, lbl in ((100,"s1","100"),(200,"s1","200"),(400,"s1","400"),(800,"s1","800 ep6000"),(800,"st","800 ep3000*"),(800,"4s","800 ep6000 4s")):
        m = stats_csv(f"results/s4/ebe/popebe_mlx_n{n}_n1000.csv", reg) if which == "s1" else None
        l1 = stats_csv(f"results/s4/ebe/popebe_mlx_l1_n{n}_n1000.csv", reg) if which == "s1" else None
        print(f"  {lbl:>14} {f(m)} {f(l1)} {f(net('film', n, reg, which))} {f(net('dc', n, reg, which))}")
print("\n* same optimiser steps (6000) as the N<=400 runs at ep6000; 4s = mean of 4 seeds. Networks N<=400: seed 1 only.")
print("\nV1 (factual) median |AUC err| %:")
for n in (100, 200, 400, 800):
    m = stats_csv(f"results/s4/ebe/popebe_mlx_n{n}_n1000.csv", "in"); a = net("film", n, "in"); b = net("dc", n, "in")
    print(f"  N={n:<4} Monolix EBE {m['v1'] if m else float('nan'):4.1f}   FiLM {a['v1'] if a else float('nan'):4.1f}   dose-cond {b['v1'] if b else float('nan'):4.1f}")
