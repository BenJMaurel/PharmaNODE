#!/usr/bin/env python3
"""Non-Gaussian scenario (Student-t etas + 30% slow absorbers, paper noise), N=100, 1000 test patients:
networks (3 seeds, ep6000 / final) vs EBEs. Raw RMSPE, RMSPE without the worst 1%, median |err|, n > 100%."""
import json, numpy as np, pandas as pd
R = "results/s4_tslow"
tr = pd.read_csv("results/exp_film_run/confound_vc00_s4_tslow/confound_truth.csv").set_index("ID")
def m(r):
    a = np.abs(r); k = a <= np.quantile(a, 0.99)
    return np.array([100*np.sqrt(np.mean(r**2)), 100*np.sqrt(np.mean(r[k]**2)), 100*np.median(a), (a > 1).sum()])
def net(arch, reg, v):
    out = []
    for s in (1, 2, 3):
        pp = json.load(open(f"{R}/eval/ep006000_{arch}_n100_s{s}_on_{reg}.json"))["per_patient"]
        out.append(m(np.array(pp[f"pred_auc_{v}"]) / np.array(pp[f"true_auc_{v}"]) - 1))
    return np.mean(out, 0), [o[0] for o in out]
def lode():
    out = []
    for s in (1, 2, 3):
        ps = json.load(open(f"{R}/eval/lode_s{s}_final.json"))["per_series"]; ids = np.array(ps["id"]); k = np.array([i.endswith("_1") for i in ids])
        out.append(m(np.array(ps["pred_auc"])[k] / np.array(ps["true_auc"])[k] - 1))
    return np.mean(out, 0), [o[0] for o in out]
def ebe(f, reg):
    d = pd.read_csv(f); return m((d[f"auc_{reg}"] / d[f"auc_true_{reg}"] - 1).values)
fmt = lambda x: f"{x[0]:5.1f} / {x[1]:5.1f} / {x[2]:5.1f} [{x[3]:3.0f}]"
print("cell = raw RMSPE / RMSPE w/o worst 1% / median |err| [n > 100%]; networks = mean of 3 seeds")
print(f"{'':32}{'V1':>28}{'V2 in range':>28}{'V2 above':>28}{'V2 below':>30}")
for lab, f in (("Monolix EBE (log-normal, N=100)", "results/s4/ebe/popebe_mlx_tslow_n100_n1000.csv"),
               ("EBE, nominal Gaussian prior", "results/s4/ebe/popebe_true_tslow_n1000.csv")):
    print(f"{lab:32}" + "".join(f"{fmt(ebe(f, r)):>28}" for r in ("v1", "in", "hi")) + f"{fmt(ebe(f, 'lo')):>30}")
for arch, lab in (("film", "FiLM, residual decoder"), ("dc", "dose-cond, residual decoder")):
    cells = [net(arch, "in", "v1")[0], net(arch, "in", "v2")[0], net(arch, "hi", "v2")[0], net(arch, "lo", "v2")[0]]
    print(f"{lab:32}" + "".join(f"{fmt(c):>28}" for c in cells[:3]) + f"{fmt(cells[3]):>30}")
    print(f"{'   V1 raw RMSPE per seed':32}{' / '.join(f'{x:.1f}' for x in net(arch, 'in', 'v1')[1]):>28}")
L = lode(); print(f"{'plain latent ODE, linear decoder':32}{fmt(L[0]):>28}")
print(f"{'   V1 raw RMSPE per seed':32}{' / '.join(f'{x:.1f}' for x in L[1]):>28}")
