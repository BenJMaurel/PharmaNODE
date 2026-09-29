#!/usr/bin/env python3
"""Per-run distributions of the 3-scenario rerun (interim or final): histograms + fitted normal, normal Q-Q plots,
and normality statistics of each method's per-run RMSPE and of the paired differences used by Table 1's t-test.
Base methods on the patients common to the base methods; default-init MAP-BE on the patients it returned, runs where
it completed only (its aborted runs are listed, not imputed)."""
import os, glob, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
sys.path.insert(0, 'scripts/scen_rerun'); from summarise import load
RUNS, OUT = 'results/scen_rerun/runs', 'results/scen_rerun/figures'
METH = [('lode_best', 'Latent ODE (_best)', '#2a78d6'), ('mapbe_var', 'MAP-BE, published init,\nSIGMA fixed', '#eb6834'),
        ('mapbe_definit_sd', 'MAP-BE, default init,\nSIGMA as published (paper)', '#1baf7a'), ('mapbe_definit_var', 'MAP-BE, default init,\nSIGMA fixed', '#eda100')]
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
SCN = {1: 'Scenario 1: correct model', 2: 'Scenario 2: missing covariate', 3: 'Scenario 3: linear model, MM data'}
f3 = lambda x: 100 * np.sqrt(np.mean(np.asarray(x) ** 2))
rows, V = [], {}
for sc in (1, 2, 3):
    dirs = sorted(d for d in glob.glob(f'{RUNS}/s{sc}_seed[0-9][0-9][0-9]') if os.path.exists(f'{d}/DONE'))
    for d in dirs:
        r, _ = load(d)
        for k in ('lode_best', 'mapbe_var'): V.setdefault((sc, k), {})[d] = f3(r[k])
        if os.path.exists(f'{d}/DONE_definit'):
            rd, _ = load(d, True)
            for k in ('lode_best', 'mapbe_definit_sd', 'mapbe_definit_var'): V.setdefault((sc, k + '@def'), {})[d] = f3(rd[k])
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': SURF, 'axes.facecolor': SURF})
def series(sc, k):
    src = V.get((sc, k if not k.startswith('mapbe_definit') else k + '@def'), {})
    return np.array(list(src.values()))
for kind in ('hist', 'qq'):
    fig, ax = plt.subplots(3, 4, figsize=(13, 8.6), constrained_layout=True)
    for i, sc in enumerate((1, 2, 3)):
        allv = np.concatenate([series(sc, k) for k, _, _ in METH if len(series(sc, k))])
        lo, hi = np.floor(allv.min()), np.ceil(np.percentile(allv, 99.5))
        for j, (k, lab, col) in enumerate(METH):
            a = ax[i, j]; x = series(sc, k)
            a.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True)
            if len(x) < 3: a.text(0.5, 0.5, 'no runs yet', transform=a.transAxes, ha='center', color=INK2); continue
            W = stats.shapiro(x).pvalue; sk = stats.skew(x); ku = stats.kurtosis(x)
            if kind == 'hist':
                bins = np.linspace(lo, max(hi, x.max()), 26)
                a.hist(x, bins=bins, color=col, edgecolor=SURF, linewidth=1.0)
                g = np.linspace(bins[0], bins[-1], 300); bw = bins[1] - bins[0]
                a.plot(g, len(x) * bw * stats.norm.pdf(g, x.mean(), x.std(ddof=1)), color=INK2, lw=1.5, ls='--')
                a.set_xlabel('per-run RMSPE (%)')
                if j == 0: a.set_ylabel(f'{SCN[sc]}\n\nruns')
            else:
                (osm, osr), (sl, ic, _) = stats.probplot(x, dist='norm')
                a.plot(osm, sl * osm + ic, color=INK2, lw=1.5, ls='--')
                a.plot(osm, osr, 'o', ms=4, color=col, markeredgecolor=SURF, markeredgewidth=0.8)
                a.set_xlabel('normal quantile')
                if j == 0: a.set_ylabel(f'{SCN[sc]}\n\nper-run RMSPE (%)')
            a.set_title(f'{lab}   n={len(x)}', fontsize=8.5, color=INK, loc='left')
            yt, va = (0.95, 'top') if kind == 'hist' else (0.04, 'bottom')   # Q-Q: lower right is empty
            a.text(0.98, yt, f'mean {x.mean():.1f}  median {np.median(x):.1f}\nskew {sk:+.2f}  ex.kurt {ku:+.1f}\nShapiro p = {W:.2g}',
                   transform=a.transAxes, ha='right', va=va, fontsize=7.5, color=INK2)
    fig.suptitle(('Per-run RMSPE distributions (dashed: normal with the same mean and s.d.)' if kind == 'hist' else
                  'Normal Q-Q plots of per-run RMSPE (points on the dashed line = Gaussian)') +
                 '   -   interim: default-init MAP-BE on the runs completed so far', color=INK, fontsize=10.5, x=0.01, ha='left')
    fig.savefig(f'{OUT}/rmspe_{kind}.png', dpi=150); plt.close(fig)
# statistics table, incl. the paired differences Table 1 tests
print(f"{'scen':4s} {'quantity':44s} {'n':>4s} {'mean':>7s} {'median':>7s} {'sd':>6s} {'skew':>6s} {'exkurt':>7s} {'Shapiro p':>10s}")
def line(sc, name, x):
    print(f"{sc:<4d} {name:44s} {len(x):4d} {x.mean():7.2f} {np.median(x):7.2f} {x.std(ddof=1):6.2f} {stats.skew(x):+6.2f} {stats.kurtosis(x):+7.2f} {stats.shapiro(x).pvalue:10.2g}")
for sc in (1, 2, 3):
    for k, lab, _ in METH:
        x = series(sc, k)
        if len(x) >= 3: line(sc, lab.replace('\n', ' '), x)
    b = V[(sc, 'lode_best')]; m = V[(sc, 'mapbe_var')]; dd = np.array([b[d] - m[d] for d in b])
    line(sc, 'DIFF LODE_best - MAP-BE fixed (Table 1 test)', dd)
    bd = V.get((sc, 'lode_best@def'), {}); md = V.get((sc, 'mapbe_definit_sd@def'), {})
    if len(bd) >= 3: line(sc, 'DIFF LODE_best - MAP-BE default/published', np.array([bd[d] - md[d] for d in bd]))
