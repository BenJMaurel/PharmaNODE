#!/usr/bin/env python3
"""OT-FiLM vs dose-cond at LOW noise (scenario 4, 0.03 + 0.03*C), same 1000 test patients.
Control-trained sweep N=100/200/400 (results/s4_nsweep2, seeds 1-3, ep6000) and N=800 (results/s4, seeds
1-4, ep6000; ep3000 = same optimiser steps as the sweep).  Per cell: mean over seeds, and the per-seed
difference FiLM - dose-cond (negative = FiLM better) with how many seeds favour FiLM.
Metrics: median |AUC err| (V2 unless stated), mean nRMSE (curve), n patients with |AUC err| > 100%."""
import os, json, numpy as np
def one(f):
    if not os.path.exists(f): return None
    j = json.load(open(f)); pp = j["per_patient"]
    r2 = np.array(pp["pred_auc_v2"]) / np.array(pp["true_auc_v2"]) - 1
    r1 = np.array(pp["pred_auc_v1"]) / np.array(pp["true_auc_v1"]) - 1
    return dict(med=100*np.median(np.abs(r2)), nrmse=j["v2"]["nrmse_pct"], blow=int((np.abs(r2) > 1).sum()),
                v1=100*np.median(np.abs(r1)), v1n=j["v1"]["nrmse_pct"])
def cell(arch, n, reg, ep="006000"):
    if n == 800:
        tag = "" if reg == "in" else reg
        fs = {s: f"results/s4/eval/ep{ep}_{arch}_s{s}_vc00_on_vc00{tag}.json" for s in (1, 2, 3, 4)}
    else:
        fs = {s: f"results/s4_nsweep2/eval/ep{ep}_{arch}_n{n}_s{s}_on_{reg}.json" for s in (1, 2, 3)}
    return {s: one(f) for s, f in fs.items() if one(f) is not None}
def show(metric, lab, regs=("in", "hi", "lo")):
    print(f"\n{lab}")
    print(f"  {'N':>11} {'':4}" + "".join(f"{r:>34}" for r in regs))
    for n, ep, nl in ((100, "006000", "100"), (200, "006000", "200"), (400, "006000", "400"), (800, "003000", "800 ep3000"), (800, "006000", "800 ep6000")):
        line_f, line_d, line_x = [], [], []
        for reg in regs:
            F, D = cell("film", n, reg, ep), cell("dc", n, reg, ep)
            common = sorted(set(F) & set(D))
            if not common: line_f.append(f"{'—':>34}"); line_d.append(f"{'—':>34}"); line_x.append(f"{'—':>34}"); continue
            fv = np.array([F[s][metric] for s in common]); dv = np.array([D[s][metric] for s in common]); d = fv - dv
            line_f.append(f"{fv.mean():9.1f} ({' '.join(f'{x:.1f}' for x in fv)})".rjust(34))
            line_d.append(f"{dv.mean():9.1f} ({' '.join(f'{x:.1f}' for x in dv)})".rjust(34))
            line_x.append(f"{d.mean():+8.1f}  FiLM better {int((d < 0).sum())}/{len(d)}".rjust(34))
        print(f"  {nl:>11} FiLM" + "".join(line_f)); print(f"  {'':>11} dc  " + "".join(line_d)); print(f"  {'':>11} diff" + "".join(line_x))
show("med", "V2 median |AUC err| %  (in range / above 10-12 mg / below 0.25-0.5 mg)")
show("nrmse", "V2 mean curve nRMSE %")
show("blow", "V2 patients with |AUC err| > 100% (of 1000)")
show("v1", "V1 (observed visit) median |AUC err| %", regs=("in",))
show("v1n", "V1 mean curve nRMSE %", regs=("in",))
