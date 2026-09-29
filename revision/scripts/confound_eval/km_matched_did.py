#!/usr/bin/env python3
"""Difference-in-differences for the matched Km confounding rerun (scripts/confound_eval/chain_km_matched.sh).

E(tr -> ev): error of a model trained on cohort tr, evaluated on test set ev (09 = confounded, 00 = control).
  removed = E(09 -> 00) - E(00 -> 00)   cost of confounded training where the dose-Km link is withdrawn
  held    = E(09 -> 09) - E(00 -> 09)   cost where the link still holds
  DiD     = removed - held              (published definition; positive = the learned association hurts)
Two test-set families: 'own' = the 200-patient test splits of confound_km09 / km00 (the published protocol),
'big' = confound_km09_big / km00_big (1000 new patients). Metrics on V2 (counterfactual) unless stated:
  nrmse  curve nRMSE (mean over patients)       med   median |AUC err| (%)
  trm    AUC RMSPE, worst 1% removed            raw   AUC RMSPE (the published metric; outlier-dominated, 1.3)
Per seed the cell value is the mean over the late checkpoints (ep 2400/2700/3000); ep3000 alone is also shown.
Paired FiLM - dose-cond and FiLM - Lu differences are over seeds (n = 3: a t-test carries little weight).
Lu = Lu et al. port, "same task" variant (chain_km_matched_lu.sh); rows show n/a until its evals exist.
"""
import os, json, glob, itertools
import numpy as np
from scipy import stats

E = os.environ.get("KM_EVAL", "results/km_matched/eval")
ARCHS = ("film", "dc", "lu"); SEEDS = (1, 2, 3)
EPS = tuple(os.environ.get("KM_EPS", "002400 002700 003000").split())
P = os.environ.get("KM_PFX", "km")   # cohort prefix: km (confound_km09/00) or vc (confound_vc09/00)


def metric(path, visit, m):
    j = json.load(open(path))
    if m == "nrmse":
        return j[visit]["nrmse_pct"]
    t = np.array(j["per_patient"][f"true_auc_{visit}"]); p = np.array(j["per_patient"][f"pred_auc_{visit}"])
    r = p / t - 1; a = np.abs(r)
    if m == "med":
        return 100 * np.median(a)
    if m == "raw":
        return 100 * np.sqrt(np.mean(r ** 2))
    if m == "trm":
        k = a <= np.quantile(a, 0.99)
        return 100 * np.sqrt(np.mean(r[k] ** 2))
    raise ValueError(m)


def cell(arch, s, tr, ev, eps, visit, m):
    vals = []
    for ep in eps:
        f = f"{E}/ep{ep}_{arch}_s{s}_{tr}_on_{ev}.json"
        if not os.path.exists(f):
            return np.nan
        vals.append(metric(f, visit, m))
    return float(np.mean(vals))


def did(arch, s, fam, eps, visit, m):
    c, k = (f"{P}09", f"{P}00") if fam == "own" else (f"{P}09big", f"{P}00big")
    g = lambda tr, ev: cell(arch, s, tr, ev, eps, visit, m)
    removed = g(f"{P}09", k) - g(f"{P}00", k)
    held = g(f"{P}09", c) - g(f"{P}00", c)
    return removed - held, removed, held, {f"{tr}>{ev}": g(tr, ev) for tr, ev in itertools.product((f"{P}09", f"{P}00"), (c, k))}


def ms(x):
    x = np.array(x, float); x = x[~np.isnan(x)]
    return (f"{x.mean():+6.2f} ± {x.std(ddof=1):4.2f}" if len(x) > 1 else f"{x.mean():+6.2f}") if len(x) else "   n/a"


if __name__ == "__main__":
    n = len(glob.glob(f"{E}/*.json"))
    print(f"{n} eval files in {E} (cohorts confound_{P}09 / confound_{P}00)\n")
    for visit in ("v2", "v1"):
        for fam in ("big", "own"):
            for eps, lab in ((EPS, "mean of ep" + "/".join(str(int(e)) for e in EPS)), (EPS[-1:], f"ep{int(EPS[-1])} only")):
                print(f"=== {visit.upper()} | test sets: {fam} | {lab}")
                for m in ("nrmse", "med", "trm", "raw"):
                    row = {a: [did(a, s, fam, eps, visit, m) for s in SEEDS] for a in ARCHS}
                    D = {a: np.array([r[0] for r in row[a]]) for a in ARCHS}
                    print(f"  {m:5s} DiD  FiLM {ms(D['film'])}  dc {ms(D['dc'])}  lu {ms(D['lu'])}"
                          f"   per seed FiLM {' '.join(f'{x:+.2f}' for x in D['film'])} | dc {' '.join(f'{x:+.2f}' for x in D['dc'])}"
                          f" | lu {' '.join(f'{x:+.2f}' for x in D['lu'])}")
                    for comp in ("dc", "lu"):
                        diff = D["film"] - D[comp]
                        ok = ~np.isnan(diff)
                        if not ok.any():
                            continue
                        p = stats.ttest_1samp(diff[ok], 0).pvalue if ok.sum() > 1 else np.nan
                        print(f"        paired FiLM-{comp} {ms(diff)}  (FiLM smaller in {int((diff[ok] < 0).sum())}/{int(ok.sum())} seeds, t-test p={p:.2f})")
                    if m == "med":
                        for a in ARCHS:
                            rem = ms([r[1] for r in row[a]]); hel = ms([r[2] for r in row[a]])
                            cells = {k: np.nanmean([r[3][k] for r in row[a]]) for k in row[a][0][3]}
                            print(f"        {a:4s} removed {rem}  held {hel}   cells " +
                                  "  ".join(f"{k} {v:.2f}" for k, v in cells.items()))
                print()
