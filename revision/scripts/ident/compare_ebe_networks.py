#!/usr/bin/env python3
"""EBE under the true population model vs the two networks, same patients.

Network per-patient AUCs come from the harness JSONs, whose row order is the harness's
data_eval order; that order is recovered from curves_s4.py's .npz (built identically)
and CHECKED by requiring the JSON's true AUCs to equal the EBE file's for every ID.

    compare_ebe_networks.py <ebe.csv> <ids.npz> <root> <epoch6> [seeds]
"""
import sys, json, numpy as np, pandas as pd
ebe_csv, npz, root, ep = sys.argv[1:5]
seeds = [int(s) for s in (sys.argv[5].split() if len(sys.argv) > 5 else "1 2 3 4".split())]
d = pd.read_csv(ebe_csv).set_index('ID')
ids = np.load(npz)['ids']; pos = {int(i): k for k, i in enumerate(ids)}
sel = [pos[int(i)] for i in d.index]

def stats(true, pred):
    rel = pred / true - 1; a = np.abs(rel); keep = a <= np.quantile(a, 0.99)
    return 100 * np.median(a), 100 * np.sqrt(np.mean(rel[keep] ** 2)), 100 * np.median(rel)

rows = []
for v in (1, 2):
    t = d[f'auc{v}_true'].values
    for tag, name in (('replay', 'true parameters (check)'), ('ebe', 'EBE, true pop. model'),
                      ('pm', 'posterior mean, true pop. model'), ('prior', 'population only (no data)')):
        if f'auc{v}_{tag}' not in d: continue
        nr = 100 * d[f'nrmse{v}_{tag}'].median() if f'nrmse{v}_{tag}' in d else np.nan
        rows.append((v, name, *stats(t, d[f'auc{v}_{tag}'].values), nr))
    for arch, name in (('film', 'OT-FiLM'), ('dc', 'dose-cond')):
        per = []
        for s in seeds:
            j = json.load(open(f'{root}/eval/ep{ep}_{arch}_s{s}_vc00_on_vc00.json'))['per_patient']
            jt = np.array(j[f'true_auc_v{v}'])[sel]; jp = np.array(j[f'pred_auc_v{v}'])[sel]
            assert np.allclose(jt, t, rtol=2e-3), f'{arch} s{s} v{v}: row order does not match the EBE ids'
            per.append(stats(t, jp))
        m = np.mean(per, axis=0)
        rows.append((v, f'{name} (mean of {len(seeds)} seeds)', *m, np.nan))
print(f"n = {len(d)} test patients of confound_vc00_s4 (control-trained, in support)\n")
print(f"{'':4}{'estimator':38}{'med |AUC err|':>15}{'tRMSPE':>9}{'bias':>8}{'curve nRMSE':>13}")
for v, name, med, tr, bias, nr in rows:
    nrs = f"{nr:13.2f}" if not np.isnan(nr) else f"{'—':>13}"
    print(f"{'V'+str(v):4}{name:38}{med:15.2f}{tr:9.2f}{bias:+8.2f}{nrs}")
    if name.startswith('dose-cond') and v == 1: print()
