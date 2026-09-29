#!/usr/bin/env python3
"""Summary of the NMI 3-scenario rerun (results/scen_rerun/runs/s<sc>_seed<NNN>/, chain_scen_rerun.sh).
Methods: latent ODE final checkpoint (epoch 6000; primary, no test-set selection), latent ODE _best (the paper's
selection on test MSE), MAP-BE with SIGMA as variances (the fix; primary comparator), MAP-BE with SIGMA as SDs
(as published; same Monolix fit). Relative error r = pred/true - 1 on the AUC of the observed dosing interval,
same reference AUC for all methods (checked). Per run (40 test patients): RMSPE and MPE, then mean +- s.d. over
runs and a paired t-test over runs (the published table's format). Pooled over runs: median |r|, RMSPE after
dropping the worst 1 %, share within +-20 %, count |r| > 100 %. Also flags implausible Monolix fits."""
import os, glob, json, numpy as np, pandas as pd
from scipy import stats
RUNS = os.environ.get('SCEN_RUNS', 'results/scen_rerun/runs')
M = [('lode_final', 'Latent ODE, final ckpt'), ('lode_best', 'Latent ODE, _best (test-selected)'),
     ('mapbe_var', 'MAP-BE, SIGMA fixed'), ('mapbe_sd', 'MAP-BE, SIGMA as published')]
# default-init redo (chain_scen_definit.sh): reported only when every finished run of the scenario has it
M_DEF = [('mapbe_definit_var', 'MAP-BE, default init, SIGMA fixed'), ('mapbe_definit_sd', 'MAP-BE, default init, SIGMA as publ.')]

def load(d, definit=False):
    out = {}
    for c in ('final', 'best'):
        j = json.load(open(f'{d}/lode_{c}.json'))['per_series']
        out[f'lode_{c}'] = pd.Series(np.array(j['pred_auc']) / np.array(j['true_auc']) - 1, index=[int(i) for i in j['id']])
    for m in ('var', 'sd'):
        x = pd.read_csv(f'{d}/tacro_mapbayest_auc_sig{m}.csv')
        out[f'mapbe_{m}'] = pd.Series((x.auc_ipred / x.AUC_observed - 1).values, index=x.ID.values)
        if definit:
            x = pd.read_csv(f'{d}/tacro_mapbayest_auc_definit_sig{m}.csv')
            out[f'mapbe_definit_{m}'] = pd.Series((x.auc_ipred / x.AUC_observed - 1).values, index=x.ID.values)
    ids = sorted(set.intersection(*[set(s.index) for s in out.values()]))
    return {k: v.loc[ids] for k, v in out.items()}, len(ids)

def main():
    for sc in (1, 2, 3):
        dirs = sorted(d for d in glob.glob(f'{RUNS}/s{sc}_seed[0-9][0-9][0-9]') if os.path.exists(f'{d}/DONE'))
        nfail = len(glob.glob(f'{RUNS}/s{sc}_seed[0-9][0-9][0-9]/FAILED_[a-z]*')) - len(glob.glob(f'{RUNS}/s{sc}_seed[0-9][0-9][0-9]/FAILED_definit_*'))
        print('=' * 100); print(f'SCENARIO {sc}: {len(dirs)} runs done, {nfail} failed')
        if not dirs:
            continue
        # default-init redo: reported on the runs where it completed; its failures (MAP-BE aborted) counted separately
        ddirs = [d for d in dirs if os.path.exists(f'{d}/DONE_definit')]
        dfail = [d for d in dirs if not os.path.exists(f'{d}/DONE_definit') and glob.glob(f'{d}/FAILED_definit_*')]
        MM = M + (M_DEF if ddirs else [])
        # base methods on the patients common to the base methods (unchanged); default-init methods, and every
        # comparison with them, on the patients the default-init MAP-BE returned (it drops patients whose simulation crashed)
        per_run = {k: {} for k, _ in MM}; per_run_def = {k: {} for k, _ in MM}; pooled = {k: [] for k, _ in MM}; bad = []; bad_def = []; dropped = []
        f3 = lambda x: (100 * np.sqrt(np.mean(x ** 2)), 100 * np.mean(x), 100 * np.median(np.abs(x)))
        for d in dirs:
            r, n = load(d)
            if n < 40: print(f'  WARNING {os.path.basename(d)}: only {n} patients matched across methods')
            for k, _ in M:
                per_run[k][d] = f3(r[k]); pooled[k].append(r[k].values)
            if d in ddirs:
                rd, nd = load(d, True)
                if nd < n: dropped.append(f'{os.path.basename(d)} ({n - nd})')
                for k, _ in MM:
                    per_run_def[k][d] = f3(rd[k])
                for k, _ in M_DEF:
                    per_run[k][d] = f3(rd[k]); pooled[k].append(rd[k].values)
            pp = pd.read_csv(f'{d}/populationParameters.txt').set_index('parameter').value
            if not (100 < pp['Vc_pop'] < 2000 and 1 < pp['KTR_pop'] < 10): bad.append(f"{os.path.basename(d)} (Vc {pp['Vc_pop']:.3g}, KTR {pp['KTR_pop']:.3g})")
            if d in ddirs:
                pq = pd.read_csv(f'{d}/populationParameters_definit.txt').set_index('parameter').value
                if not (100 < pq['Vc_pop'] < 2000 and 1 < pq['KTR_pop'] < 10): bad_def.append(f"{os.path.basename(d)} (Vc {pq['Vc_pop']:.3g}, KTR {pq['KTR_pop']:.3g})")
        print(f'  per run (40 test patients), mean +- s.d. over runs  |  pooled over the runs\' patients')
        for k, lab in MM:
            v = np.array(list(per_run[k].values())); a = np.abs(np.concatenate(pooled[k])); rr = np.concatenate(pooled[k]); keep = a <= np.quantile(a, 0.99)
            tag = '' if len(per_run[k]) == len(dirs) else f'  [{len(per_run[k])} runs]'
            print(f'  {lab:36s} RMSPE {v[:, 0].mean():6.2f} +- {v[:, 0].std(ddof=1):5.2f}  '
                  f'MPE {v[:, 1].mean():+6.2f} +- {v[:, 1].std(ddof=1):5.2f}  | '
                  f'median|r| {100 * np.median(a):5.1f}  RMSPE99 {100 * np.sqrt(np.mean(rr[keep] ** 2)):5.1f}  '
                  f'within20 {100 * np.mean(a <= .2):5.1f}  n>100% {int((a > 1).sum())}' + tag)
        pairs = [('lode_final', 'mapbe_var'), ('lode_best', 'mapbe_var'), ('lode_final', 'mapbe_sd'), ('mapbe_sd', 'mapbe_var')]
        if ddirs:
            pairs += [('lode_best', 'mapbe_definit_sd'), ('lode_best', 'mapbe_definit_var'), ('mapbe_definit_var', 'mapbe_var'), ('mapbe_definit_sd', 'mapbe_definit_var')]
        for x, y in pairs:
            src = per_run_def if 'definit' in x + y else per_run
            common = [d for d in dirs if d in src[x] and d in src[y]]
            if len(common) < 2: continue
            for mi, met in ((0, 'rmspe'), (2, 'med')):
                px = np.array([src[x][d][mi] for d in common]); py = np.array([src[y][d][mi] for d in common]); dx = px - py
                t = stats.ttest_rel(px, py); w = stats.wilcoxon(dx) if np.any(dx) else None
                print(f'  paired {x} - {y} [{met}]: {dx.mean():+6.2f} +- {dx.std(ddof=1):5.2f} (first lower in {int((dx < 0).sum())}/{len(dx)} runs), '
                      f't-test p = {t.pvalue:.2g}' + (f', Wilcoxon p = {w.pvalue:.2g}' if w else ''))
        print(f'  implausible Monolix fits (Vc_pop outside 100-2000 L or KTR_pop outside 1-10): {len(bad)}' + (': ' + ', '.join(bad) if bad else ''))
        if ddirs or dfail:
            pend = len(dirs) - len(ddirs) - len(dfail)
            print(f'  DEFAULT-INIT redo: {len(ddirs)} done, {len(dfail)} failed (MAP-BE aborted: ' + ', '.join(os.path.basename(d) for d in dfail) + f'), {pend} not run yet')
            print(f'  DEFAULT-INIT MAP-BE dropped patients (simulation crashed) in {len(dropped)} runs' + (': ' + ', '.join(dropped) if dropped else ''))
            print(f'  implausible DEFAULT-INIT Monolix fits (among done): {len(bad_def)}/{len(ddirs)}' + (': ' + ', '.join(bad_def[:20]) + (' ...' if len(bad_def) > 20 else '') if bad_def else ''))
        # form-fixed MAP-BE (combined1 residual error; chain_scen_c1*.sh): each on the runs where it completed, compared
        # with the latent ODE on the same patients; its failures (MAP-BE aborted) counted
        # auto-init (chain_scen_autoinit.sh, 28/09): ref 'csv:sigc1' = published init + form fix read from the run's CSV
        # (isolates the initialisation: same error form, same data)
        for key, lab, done, tag, ref, fpat in (('mapbe_c1', 'MAP-BE, published init, FORM FIXED', 'DONE_c1', 'sigc1', 'mapbe_var', 'FAILED_c1_*'),
                                               ('mapbe_definit_c1', 'MAP-BE, default init, FORM FIXED', 'DONE_definit_c1', 'definit_sigc1', None, 'FAILED_definit_c1_*'),
                                               ('mapbe_autoinit_c1', 'MAP-BE, auto init, FORM FIXED', 'DONE_autoinit', 'autoinit_sigc1', 'csv:sigc1', 'FAILED_autoinit_*')):
            cd = [d for d in dirs if os.path.exists(f'{d}/{done}')]
            if ref and ref.startswith('csv:'):
                cd = [d for d in cd if os.path.exists(f'{d}/tacro_mapbayest_auc_{ref[4:]}.csv')]
            cf = [d for d in dirs if not os.path.exists(f'{d}/{done}') and glob.glob(f'{d}/{fpat}')]
            if not cd and not cf: continue
            pr, pl, pv, pool, drop = [], [], [], [], []
            for d in cd:
                x = pd.read_csv(f'{d}/tacro_mapbayest_auc_{tag}.csv'); rc = pd.Series((x.auc_ipred / x.AUC_observed - 1).values, index=x.ID.values)
                rb, n = load(d); rl = rb['lode_best']; ids = sorted(set(rc.index) & set(rl.index))
                if len(ids) < n: drop.append(f'{os.path.basename(d)} ({n - len(ids)})')
                pr.append(f3(rc.loc[ids].values)); pl.append(f3(rl.loc[ids].values)); pool.append(rc.loc[ids].values)
                if ref and ref.startswith('csv:'):
                    y = pd.read_csv(f'{d}/tacro_mapbayest_auc_{ref[4:]}.csv'); ry = pd.Series((y.auc_ipred / y.AUC_observed - 1).values, index=y.ID.values)
                    ids = sorted(set(ids) & set(ry.index)); pr[-1] = f3(rc.loc[ids].values); pl[-1] = f3(rl.loc[ids].values); pool[-1] = rc.loc[ids].values
                    pv.append(f3(ry.loc[ids].values))
                elif ref: pv.append(f3(rb[ref].loc[ids].values))
            if cd:
                v = np.array(pr); a = np.abs(np.concatenate(pool)); rr = np.concatenate(pool); keep = a <= np.quantile(a, 0.99)
                print(f'  {lab:36s} RMSPE {v[:, 0].mean():6.2f} +- {v[:, 0].std(ddof=1) if len(v) > 1 else 0:5.2f}  MPE {v[:, 1].mean():+6.2f} +- {v[:, 1].std(ddof=1) if len(v) > 1 else 0:5.2f}  | '
                      f'median|r| {100 * np.median(a):5.1f}  RMSPE99 {100 * np.sqrt(np.mean(rr[keep] ** 2)):5.1f}  within20 {100 * np.mean(a <= .2):5.1f}  n>100% {int((a > 1).sum())}  [{len(cd)} runs]')
                comps = [('lode_best', np.array(pl))] + ([(ref.replace('csv:sigc1', 'mapbe_c1'), np.array(pv))] if ref else [])
                for cname, cv in comps:
                    if len(cd) < 2: break
                    for mi, met in ((0, 'rmspe'), (2, 'med')):
                        dx = cv[:, mi] - v[:, mi]; t = stats.ttest_rel(cv[:, mi], v[:, mi]); w = stats.wilcoxon(dx) if np.any(dx) else None
                        print(f'  paired {cname} - {key} [{met}]: {dx.mean():+6.2f} +- {dx.std(ddof=1):5.2f} (first lower in {int((dx < 0).sum())}/{len(dx)} runs), '
                              f't-test p = {t.pvalue:.2g}' + (f', Wilcoxon p = {w.pvalue:.2g}' if w else ''))
            print(f'  {lab}: {len(cd)} done, {len(cf)} failed' + (' (' + ', '.join(os.path.basename(d) for d in cf) + ')' if cf else '') +
                  (f'; patients dropped in {len(drop)} runs: ' + ', '.join(drop) if drop else ''))
            if key == 'mapbe_autoinit_c1' and cd:
                bad_ai = []
                for d in cd:
                    pa = pd.read_csv(f'{d}/populationParameters_autoinit.txt').set_index('parameter').value
                    if not (100 <= pa['Vc_pop'] <= 2000 and 1 <= pa['KTR_pop'] <= 10):
                        bad_ai.append(f"{os.path.basename(d)} (Vc {pa['Vc_pop']:.3g}, KTR {pa['KTR_pop']:.3g})")
                print(f'  implausible AUTO-INIT Monolix fits (among done): {len(bad_ai)}/{len(cd)}' + (': ' + ', '.join(bad_ai[:20]) + (' ...' if len(bad_ai) > 20 else '') if bad_ai else ''))

if __name__ == '__main__':
    main()
