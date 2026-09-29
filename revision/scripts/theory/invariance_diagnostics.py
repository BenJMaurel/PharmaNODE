#!/usr/bin/env python3
"""Diagnostics for the FiLM-vs-dose-cond learnability question (scenario 4, control).

1. Excess |AUC error| over the true-model EBE, binned by dose ratio d2/d1, V1 and V2.
   The dose-invariance argument predicts a COUNTERFACTUAL-specific dose-cond penalty
   that grows with |log(d2/d1)|; a generic fit penalty shows on V1 too and is flat.
2. Seed disagreement: per-patient SD over seeds of log(pred AUC), V1 vs V2.
3. Convergence: mean V2/V1 nRMSE by epoch.
Row order recovered from curves_s4's ids and CHECKED against true AUCs (as in
scripts/ident/compare_ebe_networks.py)."""
import json, os, numpy as np, pandas as pd
E = pd.read_csv('results/s4/ebe/popebe_true_n1000.csv').set_index('ID')
ids = np.load('results/s4/curves/ep006000_s1_vc00_on_vc00.npz')['ids']
E = E.loc[[int(i) for i in ids]]
d1, d2 = E.dose_v1.values, E.dose_in.values
lr = np.abs(np.log(d2 / d1))
bins = [0, np.log(1.5), np.log(2.5), np.log(4), 10]
lab = ['<1.5x', '1.5-2.5x', '2.5-4x', '>4x']
b = np.digitize(lr, bins) - 1

def load(root, ep, arch, s):
    f = f'{root}/eval/ep{ep}_{arch}_s{s}_vc00_on_vc00.json'
    if not os.path.exists(f): return None
    j = json.load(open(f))['per_patient']
    out = {}
    for v, col in (('v1', 'auc_true_v1'), ('v2', 'auc_true_in')):
        t = np.array(j[f'true_auc_{v}']); p = np.array(j[f'pred_auc_{v}'])
        assert np.allclose(t, E[col].values, rtol=2e-3), f'{f} {v}: order mismatch'
        out[v] = p
    return out

ebe = {'v1': E.auc_v1.values, 'v2': E.auc_in.values}
tru = {'v1': E.auc_true_v1.values, 'v2': E.auc_true_in.values}
err = lambda p, v: np.abs(p / tru[v] - 1) * 100

M = {'FiLM ep6000 (4 seeds)': [load('results/s4', '006000', 'film', s) for s in (1, 2, 3, 4)],
     'dc   ep6000 (4 seeds)': [load('results/s4', '006000', 'dc', s) for s in (1, 2, 3, 4)],
     'FiLM ep6000 (s1,s4)':   [load('results/s4', '006000', 'film', s) for s in (1, 4)],
     'dc   ep12000 (s1,s4)':  [load('results/s4_n12000', '012000', 'dc', s) for s in (1, 4)],
     'dc   ep9000 (s1,s4)':   [load('results/s4_n12000', '009000', 'dc', s) for s in (1, 4)],
     'FiLM ep9000 (s1,s4)':   [load('results/s4_n12000', '009000', 'film', s) for s in (1, 4)]}
M = {k: v for k, v in M.items() if all(x is not None for x in v)}

print('1. median |AUC err| %, by dose ratio (n per bin: ' + ', '.join(f'{l} {np.sum(b==i)}' for i, l in enumerate(lab)) + ')')
for v in ('v1', 'v2'):
    print(f'\n  {v.upper()}' + ''.join(f'{l:>10}' for l in lab) + f'{"all":>8}')
    print(f'  {"EBE true model":22}' + ''.join(f'{np.median(err(ebe[v], v)[b==i]):10.2f}' for i in range(4)) + f'{np.median(err(ebe[v], v)):8.2f}')
    for k, runs in M.items():
        # median over patients, then mean over seeds
        row = [np.mean([np.median(err(r[v], v)[b == i]) for r in runs]) for i in range(4)]
        allm = np.mean([np.median(err(r[v], v)) for r in runs])
        print(f'  {k:22}' + ''.join(f'{x:10.2f}' for x in row) + f'{allm:8.2f}')

print('\n2. seed disagreement: median over patients of SD_seeds[log pred AUC] (x100 ~ %)')
for k in ('FiLM ep6000 (4 seeds)', 'dc   ep6000 (4 seeds)', 'FiLM ep6000 (s1,s4)', 'dc   ep12000 (s1,s4)'):
    if k not in M: continue
    r = M[k]
    sd = {v: 100 * np.std(np.log(np.stack([x[v] for x in r])), axis=0, ddof=1) for v in ('v1', 'v2')}
    byb = ' '.join(f'{np.median(sd["v2"][b==i]):5.2f}' for i in range(4))
    print(f'  {k:22} V1 {np.median(sd["v1"]):5.2f}  V2 {np.median(sd["v2"]):5.2f}  ratio {np.median(sd["v2"])/np.median(sd["v1"]):4.2f}   V2 by ratio bin: {byb}')

print('\n3. convergence, ctrl>ctrl nRMSE % (mean over seeds)  V1 / V2')
for root, eps, seeds in (('results/s4', ['001800','002400','003000','003600','004200','006000'], (1,2,3,4)),
                         ('results/s4_n12000', ['006000','009000','012000'], (1,4))):
    for arch in ('film', 'dc'):
        cells = []
        for ep in eps:
            js = [f'{root}/eval/ep{ep}_{arch}_s{s}_vc00_on_vc00.json' for s in seeds]
            if not all(os.path.exists(f) for f in js): cells.append(f'{int(ep):>6}:   --'); continue
            j = [json.load(open(f)) for f in js]
            cells.append(f'{int(ep):>6}: {np.mean([x["v1"]["nrmse_pct"] for x in j]):5.2f}/{np.mean([x["v2"]["nrmse_pct"] for x in j]):5.2f}')
        print(f'  {root.split("/")[1]:10} {arch:5} seeds {seeds}: ' + ' '.join(cells))

print('\n4. excess over EBE by dose-ratio bin, and the trend test (patient bootstrap, 2000 draws)')
rng = np.random.default_rng(0)
def excess_by_bin(runs, idx):
    # per patient: mean over seeds of |err| minus EBE |err|; then median per bin
    e = np.mean([err(r['v2'], 'v2') for r in runs], axis=0) - err(ebe['v2'], 'v2')
    return np.array([np.median(e[idx][b[idx] == i]) for i in range(4)])
for fk, dk in (('FiLM ep6000 (4 seeds)', 'dc   ep6000 (4 seeds)'), ('FiLM ep9000 (s1,s4)', 'dc   ep9000 (s1,s4)'),
               ('FiLM ep6000 (s1,s4)', 'dc   ep12000 (s1,s4)')):
    if fk not in M or dk not in M: continue
    all_idx = np.arange(len(b))
    ef, ed = excess_by_bin(M[fk], all_idx), excess_by_bin(M[dk], all_idx)
    did = []
    for _ in range(2000):
        ix = rng.integers(0, len(b), len(b))
        f_, d_ = excess_by_bin(M[fk], ix), excess_by_bin(M[dk], ix)
        did.append((d_[3] - f_[3]) - (d_[0] - f_[0]))
    lo, hi = np.percentile(did, [2.5, 97.5])
    print(f'  {fk} vs {dk}')
    print(f'     FiLM excess ' + ' '.join(f'{x:6.2f}' for x in ef) + '   dc excess ' + ' '.join(f'{x:6.2f}' for x in ed))
    print(f'     (dc-FiLM)[>4x] - (dc-FiLM)[<1.5x] = {(ed[3]-ef[3])-(ed[0]-ef[0]):+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]')
