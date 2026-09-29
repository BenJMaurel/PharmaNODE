#!/usr/bin/env python3
"""Regenerate every table of the confounding experiment from the cells table.

    analyse_confound.py                                   # every table, both arms, ep3000 + ep6000
    analyse_confound.py --table table4 --epochs 3000 --latex
    analyse_confound.py --table paired --arms narrow --epochs 6000 --exclude 3

Tables
    table4   the paper's Table 4: V1, V2 on each test set, DiD
    damage   the DiD decomposed into the damage on each test set
    metrics  every harness metric (nRMSE, AUC RMSPE, pointwise, bias)
    ood      absolute error in support / below / above the training range, and ratios
    paired   per-seed paired contrasts (ours - dose-cond) with 95% CI and tests
    sweep    DiD and its decomposition at every evaluated epoch

Definitions.  V2 is the counterfactual-visit AUC RMSPE (%), per seed and model;
cc = trained confounded -> tested confounded, ck = confounded -> control,
kc = control -> confounded, kk = control -> control.

    damage on the confounded test = cc - kc
    damage on the control test    = ck - kk
    DiD                           = (ck - kk) - (cc - kc)
    OOD ratio at regime r         = mean(cc, ck, kc, kk at r) / mean(same four, in support)
    V1 in Table 4                 = per seed, the mean over the two test sets
    paired contrast               = ours - dose-cond within a seed.  95% CI is the t
                                    interval on the contrasts; p-values from the paired
                                    t test and the Wilcoxon signed-rank test.

Seeds are chosen per table: by default every trained seed whose required cells
are all present, naming any seed dropped for missing data.  --seeds is strict.
"""
import argparse
import csv
import math
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import registry as reg  # noqa: E402

DEFAULT_CELLS = os.path.join(reg.REPO, "results", "confound_eval", "cells.tsv")
METRICS = ["v1_mpe", "v1_auc", "v1_pw", "v1_nrmse", "v2_mpe", "v2_auc", "v2_pw", "v2_nrmse"]
BAR = "=" * 104


class MissingData(Exception):
    pass


class Cells:
    def __init__(self, path):
        self.d, dups = {}, 0
        with open(path) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                key = (int(r["seed"]), r["epoch"], r["arm"], r["regime"], r["model"], r["trained"], r["evaluated"])
                dups += key in self.d
                self.d[key] = {m: float(r[m]) if r[m] else math.nan for m in METRICS}
        if dups:
            print(f"note: {dups} cell(s) occur more than once in {path}; the last occurrence is used",
                  file=sys.stderr)

    def get(self, seed, epoch, arm, regime, model, trained, base_eval, metric="v2_auc"):
        key = (seed, str(epoch), arm, regime, model, trained, reg.eval_cohort(base_eval, regime))
        return self.d.get(key, {}).get(metric, math.nan)

    def epochs(self, arm):
        e = {k[1] for k in self.d if k[2] == arm}
        return sorted((x for x in e if x.isdigit()), key=int) + (["best"] if "best" in e else [])


# -- per-seed statistics -----------------------------------------------------------
def quad(C, s, ep, arm, model, regime="in", metric="v2_auc"):
    conf, ctrl = reg.cohorts(arm)
    g = lambda tr, ev: C.get(s, ep, arm, regime, model, tr, ev, metric)  # noqa: E731
    return g(conf, conf), g(conf, ctrl), g(ctrl, conf), g(ctrl, ctrl)  # cc, ck, kc, kk


def did(C, s, ep, arm, model, metric="v2_auc"):
    cc, ck, kc, kk = quad(C, s, ep, arm, model, "in", metric)
    return (ck - kk) - (cc - kc)


def ratio(C, s, ep, arm, model, regime):
    return float(np.mean(quad(C, s, ep, arm, model, regime)) / np.mean(quad(C, s, ep, arm, model, "in")))


def complete(C, s, ep, arm, regimes=("in",), metrics=("v2_auc",)):
    return all(not math.isnan(x) for m in reg.MODELS for r in regimes for met in metrics
               for x in quad(C, s, ep, arm, m, r, met))


# -- helpers -----------------------------------------------------------------------
def pick_seeds(arm, explicit, exclude, ok, strict=True):
    trained = list(reg.ARMS[arm][1])
    cand = [s for s in (explicit or trained) if s in trained and s not in exclude]
    use = [s for s in cand if ok(s)]
    miss = [s for s in cand if s not in use]
    if explicit and miss and strict:
        raise MissingData(f"seeds {miss} lack the cells this table needs")
    return use, miss


def msd(a):
    a = np.asarray(a, float)
    return a.mean(), (a.std(ddof=1) if len(a) > 1 else math.nan)


def fmt(a, sign=False):
    m, s = msd(a)
    return f"{m:+.2f} ± {s:.2f}" if sign else f"{m:.2f} ± {s:.2f}"


def paired(ours, dc):
    ours, dc = np.asarray(ours, float), np.asarray(dc, float)
    d = ours - dc
    out = {"d": d, "pos": int((d > 0).sum()), "n": len(d), "ci": (math.nan, math.nan), "p_t": math.nan}
    if len(d) >= 2 and d.std(ddof=1) > 0:
        out["ci"] = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
        out["p_t"] = stats.ttest_rel(ours, dc).pvalue
    try:
        out["p_w"] = stats.wilcoxon(ours, dc).pvalue
    except ValueError:
        out["p_w"] = math.nan
    return out


def header(title, use, miss):
    print(f"\n{BAR}\n{title}\n  seeds: {','.join(map(str, use)) or 'none'}"
          + (f"   (no data for seeds {','.join(map(str, miss))})" if miss else "") + f"\n{BAR}")


def seeds_line(use):
    return "  ".join(f"{s:>7}" for s in use)


# -- tables ------------------------------------------------------------------------
def t_table4(C, arm, ep, seeds, exclude, latex=False):
    conf, ctrl = reg.cohorts(arm)
    use, miss = pick_seeds(arm, seeds, exclude, lambda s: complete(C, s, ep, arm, metrics=("v1_auc", "v2_auc")))
    header(f"TABLE 4 — {arm} arm, epoch {ep}  (AUC RMSPE %, mean ± sd over seeds)", use, miss)
    if len(use) < 2:
        print("  (fewer than two seeds with complete data)")
        return
    rows = []
    for m in reg.MODELS:
        dd = [did(C, s, ep, arm, m) for s in use]
        for tr, tl in ((conf, "confounded"), (ctrl, "control")):
            v1 = [np.mean([C.get(s, ep, arm, "in", m, tr, ev, "v1_auc") for ev in (conf, ctrl)]) for s in use]
            v2c = [C.get(s, ep, arm, "in", m, tr, conf) for s in use]
            v2k = [C.get(s, ep, arm, "in", m, tr, ctrl) for s in use]
            rows.append((m, tl, v1, v2c, v2k, dd if tr == conf else None))
    print(f"  {'model':<11}{'trained on':<12}{'V1 (ordinary)':>16}{'V2 conf. test':>16}{'V2 de-conf.':>16}{'DiD':>16}")
    for m, tl, v1, v2c, v2k, dd in rows:
        name = reg.MODEL_NAME[m] if tl == "confounded" else ""
        print(f"  {name:<11}{tl:<12}{fmt(v1):>16}{fmt(v2c):>16}{fmt(v2k):>16}{fmt(dd, True) if dd else '':>16}")
    if latex:
        tex = lambda a, sign=False: ("${:+.2f} \\pm {:.2f}$" if sign else "${:.2f} \\pm {:.2f}$").format(*msd(a))  # noqa: E731
        print("\n  % LaTeX body rows for tab:dose_placement")
        for i, (m, tl, v1, v2c, v2k, dd) in enumerate(rows):
            name = reg.MODEL_NAME[m] if tl == "confounded" else ""
            last = f"\\multirow{{2}}{{*}}{{{tex(dd, True)}}}" if dd else ""
            print(f"  {name:<9} & {tl:<10} & {tex(v1)} & {tex(v2c)} & {tex(v2k)} & {last} \\\\")
            if i == 1:
                print("  \\addlinespace")


def t_damage(C, arm, ep, seeds, exclude):
    conf, ctrl = reg.cohorts(arm)
    use, miss = pick_seeds(arm, seeds, exclude, lambda s: complete(C, s, ep, arm))
    header(f"DAMAGE DECOMPOSITION — {arm} arm, epoch {ep}  (V2 AUC RMSPE, percentage points)", use, miss)
    if len(use) < 2:
        print("  (fewer than two seeds with complete data)")
        return
    print(f"  {'':<11}{'damage on conf. test':>24}{'damage on ctrl test':>24}{'DiD':>18}")
    print(f"  {'':<11}{'(cc − kc)':>24}{'(ck − kk)':>24}{'':>18}")
    for m in reg.MODELS:
        q = [quad(C, s, ep, arm, m) for s in use]
        dc_ = [cc - kc for cc, ck, kc, kk in q]
        dk_ = [ck - kk for cc, ck, kc, kk in q]
        dd = [b - a for a, b in zip(dc_, dk_)]
        print(f"  {reg.MODEL_NAME[m]:<11}{fmt(dc_, True):>24}{fmt(dk_, True):>24}{fmt(dd, True):>18}")
    print(f"\n  raw cells (mean over seeds):   {'-> conf test':>14}{'-> ctrl test':>14}")
    for m in reg.MODELS:
        for tr, tl in ((conf, "conf-trained"), (ctrl, "ctrl-trained")):
            a = np.mean([C.get(s, ep, arm, "in", m, tr, conf) for s in use])
            b = np.mean([C.get(s, ep, arm, "in", m, tr, ctrl) for s in use])
            print(f"  {reg.MODEL_NAME[m]:<11}{tl:<21}{a:14.2f}{b:14.2f}")


def t_metrics(C, arm, ep, seeds, exclude):
    conf, ctrl = reg.cohorts(arm)
    use, miss = pick_seeds(arm, seeds, exclude, lambda s: complete(C, s, ep, arm))
    header(f"ALL METRICS — {arm} arm, epoch {ep}  (%, mean over seeds; — = not recorded for this cell)", use, miss)
    if not use:
        return
    cols = [("V1 nRMSE", "v1_nrmse"), ("V1 AUC", "v1_auc"), ("V1 bias", "v1_mpe"),
            ("V2 nRMSE", "v2_nrmse"), ("V2 AUC", "v2_auc"), ("V2 pw", "v2_pw"), ("V2 bias", "v2_mpe")]
    print(f"  {'model':<11}{'trained':<12}{'test':<12}" + "".join(f"{c:>10}" for c, _ in cols))
    for m in reg.MODELS:
        for tr, tl in ((conf, "confounded"), (ctrl, "control")):
            for ev, el in ((conf, "confounded"), (ctrl, "control")):
                out = []
                for _, k in cols:
                    a = [C.get(s, ep, arm, "in", m, tr, ev, k) for s in use]
                    spec = "+.2f" if k.endswith("mpe") else ".2f"
                    out.append("—" if any(math.isnan(x) for x in a) else format(float(np.mean(a)), spec))
                print(f"  {reg.MODEL_NAME[m]:<11}{tl:<12}{el:<12}" + "".join(f"{v:>10}" for v in out))
    for k in ("v2_nrmse", "v2_auc"):
        if not all(complete(C, s, ep, arm, metrics=(k,)) for s in use):
            continue
        print(f"\n  DiD on {k}, per seed:")
        for m in reg.MODELS:
            d = [did(C, s, ep, arm, m, k) for s in use]
            print(f"    {reg.MODEL_NAME[m]:<11}{'  '.join(f'{x:+6.2f}' for x in d)}   -> {fmt(d, True)}")


def t_ood(C, arm, ep, seeds, exclude):
    conf, ctrl = reg.cohorts(arm)
    use, miss = pick_seeds(arm, seeds, exclude, lambda s: complete(C, s, ep, arm, regimes=reg.REGIMES))
    header(f"OUT-OF-RANGE DOSES — {arm} arm, epoch {ep}  (V2 AUC RMSPE %, mean ± sd)", use, miss)
    if len(use) < 2:
        print("  (fewer than two seeds with complete data)")
        return
    print(f"  {'model':<11}{'trained':<12}{'test':<7}" + "".join(f"{reg.REGIME_NAME[r]:>18}" for r in reg.REGIMES))
    for m in reg.MODELS:
        for tr, tl in ((conf, "confounded"), (ctrl, "control")):
            for ev, el in ((conf, "conf"), (ctrl, "ctrl")):
                print(f"  {reg.MODEL_NAME[m]:<11}{tl:<12}{el:<7}"
                      + "".join(f"{fmt([C.get(s, ep, arm, r, m, tr, ev) for s in use]):>18}" for r in reg.REGIMES))
    print(f"\n  ratio to in-support error, per seed:   seeds {seeds_line(use)}")
    for r in ("lo", "hi"):
        for m in reg.MODELS:
            v = [ratio(C, s, ep, arm, m, r) for s in use]
            print(f"    {reg.REGIME_NAME[r]:<13}{reg.MODEL_NAME[m]:<11}{'  '.join(f'{x:6.2f}x' for x in v)}   -> {fmt(v)}")
    if all(complete(C, s, ep, arm, regimes=("lo", "hi"), metrics=("v2_mpe",)) for s in use):
        print("\n  bias (MPE) on the out-of-range visit, mean over seeds and the four cells:")
        for r in ("lo", "hi"):
            for m in reg.MODELS:
                b = [x for s in use for x in quad(C, s, ep, arm, m, r, "v2_mpe")]
                print(f"    {reg.REGIME_NAME[r]:<13}{reg.MODEL_NAME[m]:<11}{np.mean(b):+8.2f}%")


def t_paired(C, arm, ep, seeds, exclude):
    print(f"\n{BAR}\nPAIRED CONTRASTS, ours − dose-cond — {arm} arm, epoch {ep}\n{BAR}")
    blocks = [("DiD (pp; Δ > 0 = ours more damaged)", ("in",), lambda s, m: did(C, s, ep, arm, m)),
              ("OOD ratio below range (Δ > 0 = ours worse)", ("in", "lo"), lambda s, m: ratio(C, s, ep, arm, m, "lo")),
              ("OOD ratio above range (Δ > 0 = ours worse)", ("in", "hi"), lambda s, m: ratio(C, s, ep, arm, m, "hi"))]
    for title, regimes, fn in blocks:
        try:
            use, miss = pick_seeds(arm, seeds, exclude, lambda s: complete(C, s, ep, arm, regimes=regimes))
        except MissingData as e:
            print(f"\n  {title}\n    skipped: {e}")
            continue
        print(f"\n  {title}   seeds: {','.join(map(str, use)) or 'none'}"
              + (f"   (no data for {','.join(map(str, miss))})" if miss else ""))
        if len(use) < 2:
            print("    (fewer than two seeds with complete data)")
            continue
        dc, ours = [fn(s, "dc") for s in use], [fn(s, "film") for s in use]
        p = paired(ours, dc)
        print(f"    {'seed':<11}{seeds_line(use)}      mean ± sd")
        print(f"    {'dose-cond':<11}{'  '.join(f'{x:7.2f}' for x in dc)}   {fmt(dc)}")
        print(f"    {'ours':<11}{'  '.join(f'{x:7.2f}' for x in ours)}   {fmt(ours)}")
        print(f"    {'Δ':<11}{'  '.join(f'{x:+7.2f}' for x in p['d'])}   {fmt(p['d'], True)}")
        print(f"    95% CI [{p['ci'][0]:+.2f}, {p['ci'][1]:+.2f}]   paired t p = {p['p_t']:.4f}   "
              f"Wilcoxon p = {p['p_w']:.4f}   Δ > 0 in {p['pos']}/{p['n']}")


def t_sweep(C, arm, seeds, exclude):
    print(f"\n{BAR}\nEPOCH SWEEP — {arm} arm  (V2 AUC RMSPE, mean over seeds; "
          f"ctrl base = control-trained error over both test sets)\n{BAR}")
    print(f"  {'epoch':>6}  {'seeds':<20}| " + " | ".join(f"{reg.MODEL_NAME[m]:^42}" for m in reg.MODELS))
    print(f"  {'':>28}| " + " | ".join(f"{'dmg conf':>10}{'dmg ctrl':>10}{'DiD':>9}{'ctrl base':>13}"
                                         for _ in reg.MODELS))
    for ep in C.epochs(arm):
        use, _ = pick_seeds(arm, seeds, exclude, lambda s: complete(C, s, ep, arm), strict=False)
        if not use:
            continue
        parts = []
        for m in reg.MODELS:
            q = [quad(C, s, ep, arm, m) for s in use]
            dc_ = np.mean([cc - kc for cc, ck, kc, kk in q])
            dk_ = np.mean([ck - kk for cc, ck, kc, kk in q])
            base = np.mean([(kc + kk) / 2 for cc, ck, kc, kk in q])
            parts.append(f"{dc_:+10.2f}{dk_:+10.2f}{dk_ - dc_:+9.2f}{base:13.2f}")
        print(f"  {ep:>6}  {','.join(map(str, use)):<20}| " + " | ".join(parts))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", default=DEFAULT_CELLS)
    ap.add_argument("--table", default="all", choices=["all", "table4", "damage", "metrics", "ood", "paired", "sweep"])
    ap.add_argument("--arms", nargs="+", default=list(reg.ARMS), choices=list(reg.ARMS))
    ap.add_argument("--epochs", nargs="+", default=["3000", "6000"])
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--exclude", nargs="+", type=int, default=[])
    ap.add_argument("--latex", action="store_true", help="also print Table 4 as LaTeX body rows")
    a = ap.parse_args()
    C = Cells(a.cells)
    tables = {"table4": lambda arm, ep: t_table4(C, arm, ep, a.seeds, a.exclude, a.latex),
              "damage": lambda arm, ep: t_damage(C, arm, ep, a.seeds, a.exclude),
              "metrics": lambda arm, ep: t_metrics(C, arm, ep, a.seeds, a.exclude),
              "ood": lambda arm, ep: t_ood(C, arm, ep, a.seeds, a.exclude),
              "paired": lambda arm, ep: t_paired(C, arm, ep, a.seeds, a.exclude)}
    for arm in a.arms:
        if a.table != "sweep":
            names = list(tables) if a.table == "all" else [a.table]
            for ep in a.epochs:
                for n in names:
                    try:
                        tables[n](arm, ep)
                    except MissingData as e:
                        print(f"\n{BAR}\n{n} — {arm} arm, epoch {ep}: skipped, {e}\n{BAR}")
        if a.table in ("all", "sweep"):
            t_sweep(C, arm, a.seeds, a.exclude)


if __name__ == "__main__":
    main()
