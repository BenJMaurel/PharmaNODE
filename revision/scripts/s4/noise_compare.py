#!/usr/bin/env python3
"""Low noise (0.03 + 0.03*C) vs paper noise (0.71 + 0.113*C), N=100, scenario-4 control cohort,
same 1000 test patients. EBE rows from results/s4/ebe; networks seeds 1-3 at ep6000 from
results/s4_nsweep2 (low) and results/s4_pnoise (paper).
Cell: median |AUC err| % (median signed err) [n |err| > 100%]; nRMSE = mean over patients."""
import os, json, numpy as np, pandas as pd
def ebe(tag, reg):
    f = f"results/s4/ebe/popebe_{tag}_n1000.csv"
    if not os.path.exists(f): return None
    d = pd.read_csv(f)
    if f"auc_{reg}" not in d: return None
    r = d[f"auc_{reg}"] / d[f"auc_true_{reg}"] - 1
    return [100*np.median(np.abs(r)), 100*np.median(r), 100*d[f"nrmse_{reg}"].mean(), int((np.abs(r) > 1).sum()),
            100*np.median(np.abs(d.auc_v1 / d.auc_true_v1 - 1))]
def net(root, arch, reg, seeds=(1, 2, 3)):
    out = []
    for s in seeds:
        f = f"{root}/eval/ep006000_{arch}_n100_s{s}_on_{reg}.json"
        if not os.path.exists(f): continue
        j = json.load(open(f)); t = np.array(j["per_patient"]["true_auc_v2"]); p = np.array(j["per_patient"]["pred_auc_v2"])
        r = p / t - 1; tv = np.array(j["per_patient"]["true_auc_v1"]); pv = np.array(j["per_patient"]["pred_auc_v1"])
        out.append([100*np.median(np.abs(r)), 100*np.median(r), j["v2"]["nrmse_pct"], int((np.abs(r) > 1).sum()),
                    100*np.median(np.abs(pv / tv - 1))])
    return (np.mean(out, 0), [o[0] for o in out]) if out else (None, [])
fmt = lambda c: f"{'—':>26}" if c is None else f"{c[0]:5.1f} ({c[1]:+5.1f}) {c[2]:6.1f} [{int(round(c[3])):>3}]"
rows = [("EBE, true model", "true", "true_pnoise"), ("Monolix, true covariates", "mlx_n100", "mlx_pnoise_n100"),
        ("Monolix, L1", "mlx_l1_n100", "mlx_l1_pnoise_n100")]
for reg, lab in (("in", "V2 in range (1-8 mg)"), ("hi", "V2 above range (10-12 mg)"), ("lo", "V2 below range (0.25-0.5 mg)")):
    print(f"\n{lab}.  cell = median |AUC err| (bias)  mean nRMSE  [n>100%]")
    print(f"  {'':26}{'low noise':>28}{'paper noise':>28}")
    for name, lo, hi in rows:
        print(f"  {name:26}  {fmt(ebe(lo, reg))}  {fmt(ebe(hi, reg))}")
    for arch, name in (("film", "OT-FiLM (3 seeds)"), ("dc", "dose-cond (3 seeds)")):
        a, sa = net("results/s4_nsweep2", arch, reg); b, sb = net("results/s4_pnoise", arch, reg)
        print(f"  {name:26}  {fmt(a)}  {fmt(b)}   per seed {' '.join(f'{x:.1f}' for x in sa)} | {' '.join(f'{x:.1f}' for x in sb)}")
print("\nV1 (observed visit) median |AUC err| %:  low -> paper")
for name, lo, hi in rows:
    a, b = ebe(lo, "in"), ebe(hi, "in"); print(f"  {name:26} {a[4]:5.1f} -> {b[4]:5.1f}")
for arch, name in (("film", "OT-FiLM"), ("dc", "dose-cond")):
    a, _ = net("results/s4_nsweep2", arch, "in"); b, _ = net("results/s4_pnoise", arch, "in")
    print(f"  {name:26} {a[4]:5.1f} -> {b[4]:5.1f}")
