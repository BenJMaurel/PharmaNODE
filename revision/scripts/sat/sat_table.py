#!/usr/bin/env python3
"""Hidden-saturation experiment (handoff 11.3): visit-1 AUC error on the wide test population, stratified by each patient's
saturation scale k (training population: k = 30; lower k = more saturated = further out of distribution).
Arms: MAP-BE linear (BIC-selected), MAP-BE MM (estimated on the same k = 30 training cohort), plain latent ODE seeds 1-3
(metrics per seed, then mean over seeds). Relative error r = pred/true - 1."""
import json, os, numpy as np, pandas as pd
R = "results/sat"; TE = "results/exp_film_run/confound_vc00_s4_satwide_win"
truth = pd.read_csv(f"{TE}/confound_truth.csv").set_index("ID"); truth = truth[truth.split == "test"]
BINS = [1, 2, 4, 8, 16, 30.0001]; LAB = ["k 1-2", "k 2-4", "k 4-8", "k 8-16", "k 16-30"]
kbin = pd.cut(truth.sat_k, BINS, right=False, labels=LAB)
def metrics(r):
    r = r.dropna(); a = np.abs(r.values); n = len(a)
    if n == 0: return dict(n=0)
    keep = np.sort(a)[: max(1, int(np.floor(0.99 * n)))]
    return dict(n=n, med=100 * np.median(a), rmspe=100 * np.sqrt(np.mean(r.values ** 2)),
                trim=100 * np.sqrt(np.mean(keep ** 2)), mpe=100 * np.mean(r.values), gt100=int((a > 1).sum()))
arms = {}
for arm in ("lin", "mm"):
    f = f"{R}/ebe/popebe_{arm}_satwide.csv"
    if os.path.exists(f):
        x = pd.read_csv(f).set_index("ID"); arms[f"MAP-BE {arm}"] = [x.auc_v1 / x.auc_true_v1 - 1]
lode = []
for s in (1, 2, 3):
    f = f"{R}/eval/lode_s{s}.json"
    if not os.path.exists(f): continue
    p = json.load(open(f))["per_series"]; d = pd.DataFrame(p)
    d = d[d.id.astype(str).str.endswith("_1")]; d["ID"] = d.id.astype(str).str.split("_").str[0].astype(int)
    d = d.set_index("ID"); lode.append(d.pred_auc / d.true_auc - 1)
if lode: arms[f"latent ODE ({len(lode)} seeds)"] = lode
print(f"test patients: {len(truth)}; per bin: " + ", ".join(f"{l} {int((kbin == l).sum())}" for l in LAB))
for key, lab in (("med", "median |AUC err| %"), ("trim", "RMSPE w/o worst 1% %"), ("mpe", "MPE % (bias)"), ("gt100", "n |err| > 100%")):
    print(f"\n{lab:<24}" + "".join(f"{l:>10}" for l in LAB) + f"{'all':>10}")
    for name, rs in arms.items():
        row = []
        for l in LAB + ["all"]:
            vals = [metrics(r.reindex(truth.index[kbin == l] if l != "all" else truth.index)).get(key, np.nan) for r in rs]
            row.append(np.nanmean(vals))
        print(f"{name:<24}" + "".join(f"{v:10.1f}" if key != "gt100" else f"{v:10.1f}" for v in row))
