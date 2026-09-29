#!/usr/bin/env python3
"""Job list for the training-set recalibration: score every control-trained scenario-4 run on its OWN
training cohort (--eval-split all; the first N rows are the training patients) so a global level factor
can be estimated without the test set. Prints one shell command per line (skips existing outputs)."""
import os
PY = "/opt/miniconda3/bin/python3.12"
jobs = []
def add(ctx, arch, s, ckpt, tr, n):
    out = f"results/recal/{ctx}/{arch}_s{s}_all.json"
    if os.path.exists(out) or not os.path.exists(ckpt): return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sc = "test_film_matched.py" if arch == "film" else "test_dose_cond.py"
    jobs.append(f"OMP_NUM_THREADS=1 nice -n 10 {PY} {sc} --experiment {tr} --scale-from {tr} --eval-split all "
                f"--ckpt {ckpt} --label {ctx}_{arch}_s{s} --out-json {out} > logs/recal_{ctx}_{arch}_s{s}.log 2>&1")
F = "__noz0_sig{sig}_sc{sc}_dech50_sel-mse_v2_ep006000.ckpt"; D = "__sig-{sig}_ep006000.ckpt"
for n in (100, 200, 400):
    tr = f"confound_vc00_s4_n{n}"
    for s in (1, 2, 3):
        add(f"low_n{n}", "film", s, f"results/s4_nsweep2/film_n{n}_s{s}/exp_film_run/{tr}/traj/experiment_film_{tr}" + F.format(sig="0.05", sc="141.27"), tr, n)
        add(f"low_n{n}", "dc", s, f"results/s4_nsweep2/dc_n{n}_s{s}/exp_dosecond_run/{tr}/traj/experiment_dosecond_{tr}" + D.format(sig="0.05"), tr, n)
tr = "confound_vc00_s4"
for s in (1, 2, 3, 4):
    add("low_n800", "film", s, f"results/s4/film_s{s}/exp_film_run/{tr}/traj/experiment_film_{tr}" + F.format(sig="0.05", sc="141.27"), tr, 800)
    add("low_n800", "dc", s, f"results/s4/dc_s{s}/exp_dosecond_run/{tr}/traj/experiment_dosecond_{tr}" + D.format(sig="0.05"), tr, 800)
tr = "confound_vc00_s4_pnoise_n100"
for root, ctx, sig, sc in (("results/s4_pnoise", "paper_sig0.05", "0.05", "141.27"), ("results/s4_pnoise_sig0217", "paper_sig0.217", "0.217", "7.5"),
                          ("results/s4_pnoise_sig001", "paper_sig0.01", "0.01", "3531.68")):
    for s in (1, 2, 3):
        add(ctx, "film", s, f"{root}/film_n100_s{s}/exp_film_run/{tr}/traj/experiment_film_{tr}" + F.format(sig=sig, sc=sc), tr, 100)
        add(ctx, "dc", s, f"{root}/dc_n100_s{s}/exp_dosecond_run/{tr}/traj/experiment_dosecond_{tr}" + D.format(sig=sig), tr, 100)
print("\n".join(jobs))
