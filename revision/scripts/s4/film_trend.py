#!/usr/bin/env python3
"""FiLM (or dose-cond) trend across epochs, seeds 1 and 4, scenario 4.
ep6000 in-support from results/s4 (the 12k run reproduces it, handoff §6); later epochs from
results/s4_n12000.  V2 nRMSE 2x2, V1 nRMSE, median |AUC err|, and OOD (lo/hi) where scored.
    film_trend.py [arch] [epochs6...]"""
import json, os, sys, numpy as np
arch = sys.argv[1] if len(sys.argv) > 1 else "film"
eps = sys.argv[2:] or ["006000", "007500", "009000", "010500", "012000"]
def J(ep, s, tr, te, reg=""):
    for root in (("results/s4",) if ep == "006000" and not reg else ()) + ("results/s4_n12000", "results/s4"):
        f = f"{root}/eval/ep{ep}_{arch}_s{s}_{tr}_on_{te}{reg}.json"
        if os.path.exists(f): return json.load(open(f))
def med(j, v):
    t = np.array(j['per_patient'][f'true_auc_{v}']); p = np.array(j['per_patient'][f'pred_auc_{v}'])
    return 100 * np.median(abs(p / t - 1))
print(f"{arch}, seeds 1 and 4 (per seed in brackets).  V2 nRMSE %: first = trained on, second = tested on")
print(f"{'epoch':>6} {'ctrl>ctrl':>18} {'ctrl>conf':>9} {'conf>ctrl':>18} {'conf>conf':>9} {'V1 nRMSE':>9} {'medAUC V1/V2':>13} {'lo nRMSE':>9} {'hi nRMSE':>9}")
for ep in eps:
    c = {s: [J(ep, s, tr, te) for tr, te in (("vc00","vc00"),("vc00","vc09"),("vc09","vc00"),("vc09","vc09"))] for s in (1, 4)}
    if any(x is None for s in c for x in c[s]): print(f"{int(ep):>6}   --"); continue
    v = lambda i: [c[s][i]['v2']['nrmse_pct'] for s in (1, 4)]
    ood = {r: [J(ep, s, "vc00", "vc00", r) for s in (1, 4)] for r in ("lo", "hi")}
    oo = {r: (f"{np.mean([x['v2']['nrmse_pct'] for x in ood[r]]):9.2f}" if all(ood[r]) else f"{'--':>9}") for r in ood}
    print(f"{int(ep):>6} {np.mean(v(0)):6.2f} ({v(0)[0]:5.2f}/{v(0)[1]:5.2f}) {np.mean(v(1)):9.2f} "
          f"{np.mean(v(2)):6.2f} ({v(2)[0]:5.2f}/{v(2)[1]:5.2f}) {np.mean(v(3)):9.2f} "
          f"{np.mean([c[s][0]['v1']['nrmse_pct'] for s in (1,4)]):9.2f} "
          f"{np.mean([med(c[s][0],'v1') for s in (1,4)]):6.2f}/{np.mean([med(c[s][0],'v2') for s in (1,4)]):5.2f} {oo['lo']} {oo['hi']}")
