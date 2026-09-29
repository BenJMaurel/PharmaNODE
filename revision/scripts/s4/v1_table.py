#!/usr/bin/env python3
"""V1 (observed visit) AUC accuracy of the networks in every scenario-4 context we have, with the EBE
reference. Per run: median |err|, RMSPE, level bias (median signed err) and spread (median |err - bias|);
mean over seeds (per-seed medians in brackets)."""
import os, json, numpy as np, pandas as pd
def run(f):
    if not os.path.exists(f): return None
    pp = json.load(open(f))["per_patient"]; r = np.array(pp["pred_auc_v1"]) / np.array(pp["true_auc_v1"]) - 1
    b = np.median(r); return [100*np.median(np.abs(r)), 100*np.sqrt(np.mean(r**2)), 100*b, 100*np.median(np.abs(r - b))]
def ctx(files):
    v = [x for x in (run(f) for f in files) if x is not None]
    if not v: return None
    a = np.array(v); return a.mean(0), a[:, 0]
def ebe(tag):
    d = pd.read_csv(f"results/s4/ebe/popebe_{tag}_n1000.csv"); r = (d.auc_v1 / d.auc_true_v1 - 1).values; b = np.median(r)
    return 100*np.median(np.abs(r)), 100*np.sqrt(np.mean(r**2)), 100*b, 100*np.median(np.abs(r - b))
C = [("low", "100", "σ0.05", "results/s4_nsweep2/eval/ep006000_{a}_n100_s{s}_on_in.json", (1,2,3), "mlx_n100"),
     ("low", "200", "σ0.05", "results/s4_nsweep2/eval/ep006000_{a}_n200_s{s}_on_in.json", (1,2,3), "mlx_n200"),
     ("low", "400", "σ0.05", "results/s4_nsweep2/eval/ep006000_{a}_n400_s{s}_on_in.json", (1,2,3), "mlx_n400"),
     ("low", "800", "σ0.05", "results/s4/eval/ep006000_{a}_s{s}_vc00_on_vc00.json", (1,2,3,4), "mlx_n800"),
     ("paper", "100", "σ0.05", "results/s4_pnoise/eval/ep006000_{a}_n100_s{s}_on_in.json", (1,2,3), "mlx_pnoise_n100"),
     ("paper", "100", "σ0.217", "results/s4_pnoise_sig0217/eval/ep006000_{a}_n100_s{s}_on_in.json", (1,2,3), "mlx_pnoise_n100")]
print("V1 AUC error %, 1000 test patients. cells: median |err| (per seed) | RMSPE | bias | spread")
for noise, n, sig, pat, seeds, et in C:
    e = ebe(et)
    print(f"\n{noise} noise, N={n}, {sig}:   Monolix EBE  median {e[0]:4.1f} | RMSPE {e[1]:5.1f} | bias {e[2]:+5.1f} | spread {e[3]:4.1f}")
    for a, nm in (("film", "OT-FiLM"), ("dc", "dose-cond")):
        r = ctx([pat.format(a=a, s=s) for s in seeds])
        if r is None: print(f"   {nm:10} —"); continue
        m, per = r
        print(f"   {nm:10} median {m[0]:4.1f} ({' '.join(f'{x:.1f}' for x in per)}) | RMSPE {m[1]:5.1f} | bias {m[2]:+5.1f} | spread {m[3]:4.1f}")
