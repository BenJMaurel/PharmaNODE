#!/usr/bin/env python3
"""Training-set recalibration of every control-trained scenario-4 run.
For each run: c = 1 / median(pred/true) of V1 AUC over its OWN training patients (first N rows of the
--eval-split all JSON, checked against the training CSV). All test predictions (V1, V2 in/hi/lo) are
multiplied by c. No test data is used to choose c. Reports median |AUC err| before -> after (mean over
seeds, per-seed in brackets) with the EBE reference of the same context."""
import os, json, numpy as np, pandas as pd
CTX = [("low_n100", 100, "confound_vc00_s4_n100", "results/s4_nsweep2/eval/ep006000_{a}_n100_s{s}_on_{r}.json", (1,2,3), "mlx_n100"),
       ("low_n200", 200, "confound_vc00_s4_n200", "results/s4_nsweep2/eval/ep006000_{a}_n200_s{s}_on_{r}.json", (1,2,3), "mlx_n200"),
       ("low_n400", 400, "confound_vc00_s4_n400", "results/s4_nsweep2/eval/ep006000_{a}_n400_s{s}_on_{r}.json", (1,2,3), "mlx_n400"),
       ("low_n800", 800, "confound_vc00_s4", "results/s4/eval/ep006000_{a}_s{s}_vc00_on_vc00{t}.json", (1,2,3,4), "mlx_n800"),
       ("paper_sig0.05", 100, "confound_vc00_s4_pnoise_n100", "results/s4_pnoise/eval/ep006000_{a}_n100_s{s}_on_{r}.json", (1,2,3), "mlx_pnoise_n100"),
       ("paper_sig0.217", 100, "confound_vc00_s4_pnoise_n100", "results/s4_pnoise_sig0217/eval/ep006000_{a}_n100_s{s}_on_{r}.json", (1,2,3), "mlx_pnoise_n100"),
       ("paper_sig0.01", 100, "confound_vc00_s4_pnoise_n100", "results/s4_pnoise_sig001/eval/ep006000_{a}_n100_s{s}_on_{r}.json", (1,2,3), "mlx_pnoise_n100")]
def med(p, t): return 100*np.median(np.abs(p / t - 1))
def test_json(pat, a, s, r):
    f = pat.format(a=a, s=s, r=r, t="" if r == "in" else r)
    return json.load(open(f))["per_patient"] if os.path.exists(f) else None
rows = []
for ctx, n, tr, pat, seeds, et in CTX:
    auc_tr = np.sort(pd.read_csv(f"results/exp_film_run/{tr}/virtual_cohort_film_train.csv").query("VISIT == 1").groupby("ID").AUC.first().values)
    e = pd.read_csv(f"results/s4/ebe/popebe_{et}_n1000.csv")
    eref = {"V1": med(e.auc_v1, e.auc_true_v1), **{f"V2 {r}": med(e[f"auc_{r}"], e[f"auc_true_{r}"]) for r in ("in", "hi", "lo")}}
    for a in ("film", "dc"):
        acc = {k: ([], []) for k in eref}; cs = []
        for s in seeds:
            fa = f"results/recal/{ctx}/{a}_s{s}_all.json"
            if not os.path.exists(fa): continue
            pa = json.load(open(fa))["per_patient"]; tt = np.array(pa["true_auc_v1"]); pp = np.array(pa["pred_auc_v1"])
            assert np.allclose(np.sort(tt[:n]), auc_tr, rtol=1e-3), f"{fa}: first {n} rows are not the training set"
            c = 1.0 / np.median(pp[:n] / tt[:n]); cs.append(c)
            tin = test_json(pat, a, s, "in")
            if tin is None: continue
            for k, (tj, key) in {"V1": (tin, "v1"), "V2 in": (tin, "v2"), "V2 hi": (test_json(pat, a, s, "hi"), "v2"),
                                 "V2 lo": (test_json(pat, a, s, "lo"), "v2")}.items():
                if tj is None: continue
                t = np.array(tj[f"true_auc_{key}"]); p = np.array(tj[f"pred_auc_{key}"])
                acc[k][0].append(med(p, t)); acc[k][1].append(med(c * p, t))
        rows.append((ctx, a, cs, acc, eref))
print("median |AUC err| %, before -> after training-set recalibration (mean over seeds); EBE reference")
for ctx, a, cs, acc, eref in rows:
    print(f"\n{ctx:15} {'FiLM' if a == 'film' else 'dose-cond':9} factors c = {' '.join(f'{c:.3f}' for c in cs)}")
    for k in ("V1", "V2 in", "V2 hi", "V2 lo"):
        b, af = acc[k]
        if not b: continue
        print(f"   {k:6} {np.mean(b):6.1f} -> {np.mean(af):6.1f}   (after, per seed: {' '.join(f'{x:.1f}' for x in af)})   EBE {eref[k]:5.1f}")
