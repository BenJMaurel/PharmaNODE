#!/usr/bin/env python3
"""MAP-BE on the scenario-4 test patients under a GIVEN population model -- the true one,
or one estimated by Monolix (SAEM) from a training subset -- then the Visit-2 counterfactual.

One fit to the 3 Visit-1 points of confound_vc00_s4 (0, 1, 3 h; the encoder's inputs), then
predictions for the same patient's Visit-2 dose in range (confound_vc00_s4), below range
(_lo) and above range (_hi) -- identical physiology across the three cohorts.
Same simulator and Levenberg-Marquardt as scripts/ident/ebe_s4_batch.py; only the
population parameters change, so any gap to the true-model EBE is the cost of ESTIMATING
the population model.

    ebe_popmodel.py true            <n> <seed> <out.csv>
    ebe_popmodel.py <estimates.json> <n> <seed> <out.csv>
"""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, 'scripts/ident'); sys.path.insert(0, '.')
from ebe_s4_batch import simulate, FIT, ENC_T
from gen_tacro_film import POPULATION_PARAMS as P, IPV_OMEGA as OM, RESIDUAL_ERROR_ADD_SD, RESIDUAL_ERROR_PROP_SD
from mle_km_s4 import replay
import os
COHORT = os.environ.get('EBE_COHORT', 'confound_vc00_s4')   # e.g. confound_vc00_s4_pnoise
REL = np.array([0., 0.33, 0.67, 1., 1.5, 2., 3., 4., 6., 9., 12., 24.])

def spec_true():
    # residual error of the cohort actually scored: its noise.json if present (paper-noise cohorts),
    # else the simulator module's values (0.03 + 0.03*C for every cohort generated before 2026-09-21)
    a_sd, b_sd = RESIDUAL_ERROR_ADD_SD, RESIDUAL_ERROR_PROP_SD
    nf = f'results/exp_film_run/{COHORT}/noise.json'
    if os.path.exists(nf):
        nz = json.load(open(nf)); a_sd, b_sd = nz['add_sd'], nz['prop_sd']
    om_km = float(np.sqrt(max(OM['Km'] ** 2 - (0.1683 * P['theta_Km_HT']) ** 2, 1e-6)))
    return dict(Ktr=P['theta1_Ktr'], bKtr=np.log(P['theta2_Ktr_study']), Vc=P['theta6_Vc'],
                bVc=np.log(P['theta7_Vc_study']), Q=P['Q'], Vp=P['Vp'], Vmax=P['theta_Vmax'],
                bVmaxHT=P['theta4_CL_HT'], bVmaxCYP=np.log(P['theta5_CL_CYP']), Km=P['theta_Km'],
                bKmHT=P['theta_Km_HT'], om=np.array([OM['Ktr'], OM['Q'], OM['Vc'], OM['Vp'], OM['Vmax'], om_km]),
                corr=P['rho_Km_Vc'], a=a_sd, b=b_sd)

def spec_monolix(path):
    e = json.load(open(path))['estimates']
    return dict(Ktr=e['Ktr_pop'], bKtr=e['beta_Ktr_ST_1'], Vc=e['Vc_pop'], bVc=e['beta_Vc_ST_1'],
                Q=e['Q_pop'], Vp=e['Vp_pop'], Vmax=e['Vmax_pop'], bVmaxHT=e['beta_Vmax_tHT'],
                bVmaxCYP=e['beta_Vmax_CYP_1'], Km=e['Km_pop'], bKmHT=e.get('beta_Km_tHT', 0.0),
                om=np.array([e['omega_Ktr'], e['omega_Q'], e['omega_Vc'], e['omega_Vp'], e['omega_Vmax'], e['omega_Km']]),
                corr=e.get('corr_Vc_Km', 0.0), a=e['a'], b=e['b'])  # absent in variant l1

def typical(S, st, cyp, ht):
    lh = np.log(ht / 35.0)
    return np.array([S['Ktr'] * np.exp(S['bKtr'] * st), S['Q'], S['Vc'] * np.exp(S['bVc'] * st), S['Vp'],
                     S['Vmax'] * np.exp(S['bVmaxHT'] * lh + S['bVmaxCYP'] * cyp), S['Km'] * np.exp(S['bKmHT'] * lh)])

def prec(S):
    Om = np.diag(S['om'] ** 2); iv, ik = FIT.index('Vc'), FIT.index('Km')
    Om[iv, ik] = Om[ik, iv] = S['corr'] * S['om'][iv] * S['om'][ik]
    return np.linalg.inv(Om)

def fit_group(y, tv, cl, dose, form, PR, a, b, iters=40, h=1e-3):
    K = y.shape[0]; eta = np.zeros((K, 6)); lam = np.full(K, 1e-2)
    def objective(e):
        f = simulate(tv * np.exp(e), cl, dose, form, ENC_T).T; sd = a + b * f
        return (((y - f) / sd) ** 2).sum(1) + np.einsum('ki,ij,kj->k', e, PR, e)
    obj = objective(eta)
    for it in range(iters):
        pert = eta[:, None, :] + h * np.concatenate([np.zeros((1, 6)), np.eye(6)])[None]
        F = simulate((tv[:, None, :] * np.exp(pert)).reshape(-1, 6), np.repeat(cl, 7), np.repeat(dose, 7),
                     form, ENC_T).T.reshape(K, 7, 3)
        f0 = F[:, 0]; J = (F[:, 1:] - f0[:, None]) / h; W = 1.0 / (a + b * f0)
        Jw = J * W[:, None, :]; rw = (y - f0) * W
        A = np.einsum('kip,kjp->kij', Jw, Jw) + PR[None]; g = np.einsum('kip,kp->ki', Jw, rw) - eta @ PR
        step = np.linalg.solve(A + lam[:, None, None] * np.eye(6)[None], g[..., None])[..., 0]
        new = eta + step; nobj = objective(new); ok = nobj < obj
        eta[ok] = new[ok]; obj[ok] = nobj[ok]; lam = np.where(ok, lam * 0.3, lam * 10.0)
        if np.max(np.abs(step[ok]), initial=0) < 1e-5 and it > 5: break
    return eta

def load(c):
    t = pd.read_csv(f'results/exp_film_run/{c}/confound_truth.csv').set_index('ID')
    o = pd.read_csv(f'results/exp_film_run/{c}/virtual_cohort_film_test.csv')
    o['DVn'] = pd.to_numeric(o.DV, errors='coerce'); o['Tn'] = pd.to_numeric(o.TIME, errors='coerce')
    return t, o

def curve(o, pid, v):
    g = o[(o.ID == pid) & (o.VISIT == v) & o.DVn.notna()].sort_values('Tn'); return g.DVn.values, float(g.AUC.iloc[0])

def main():
    which, n, seed, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    S = spec_true() if which == 'true' else spec_monolix(which); PR = prec(S)
    _t = pd.read_csv(f'results/exp_film_run/{COHORT}/confound_truth.csv').set_index('ID')
    if all(c in _t.columns for c in FIT):
        # cohorts generated with non-default parameter distributions record the true parameters and
        # covariates; replaying the default sampler would not reproduce them
        TP = _t.rename(columns={'DRUG': 'form'}).copy()
        TP['cyp'] = (TP['CYP'] == 'expresser').astype(int)
        print('true parameters read from confound_truth.csv', flush=True)
    else:
        TP = replay(max(1800, int(_t.index.max())), 4)   # windowed cohorts draw past ID 1800 (handoff 1.12)
    C = {r: load(c) for r, c in (('in', COHORT), ('lo', COHORT + '_lo'), ('hi', COHORT + '_hi'))
         if os.path.exists(f'results/exp_film_run/{c}/confound_truth.csv')}   # regimes not generated yet are skipped
    print(f'cohort {COHORT}: regimes {list(C)}', flush=True)
    T0, O0 = C['in']
    te = T0[T0.split == 'test'].index.values.copy(); np.random.RandomState(seed).shuffle(te); ids = te[:n]
    rows = []
    for form in ('Advagraf', 'Prograf'):
        g = [i for i in ids if TP.loc[i].form == form]
        st = 1.0 if form == 'Prograf' else 0.0
        TV = np.stack([typical(S, st, int(TP.loc[i].cyp), float(TP.loc[i].HT)) for i in g])
        CL = np.full(len(g), 20.0)                                   # inert under MM: never enters the ODE
        d1 = np.array([T0.loc[i].d1 for i in g])
        y3 = np.array([[curve(O0, i, 1)[0][int(np.argmin(np.abs(REL - t)))] for t in ENC_T] for i in g])
        eta = fit_group(y3, TV, CL, d1, form, PR, S['a'], S['b']); th = TV * np.exp(eta)
        hi = 12.0 if form == 'Prograf' else 24.0; ta = np.round(np.arange(0.0, hi + 1e-9, 0.02), 4)
        # visit 1 (factual) + the three counterfactual targets
        preds = {'v1': (d1, [curve(O0, i, 1) for i in g])}
        for r, (T, O) in C.items():
            preds[r] = (np.array([T.loc[i].d2 for i in g]), [curve(O, i, 2) for i in g])
        for k, i in enumerate(g):
            row = dict(ID=i, form=form)
            rows.append(row)
        for tag, (dd, obs) in preds.items():
            auc = np.trapezoid(simulate(th, CL, dd, form, ta), ta, axis=0); c12 = simulate(th, CL, dd, form, REL).T
            for k in range(len(g)):
                y12, a_true = obs[k]; row = rows[len(rows) - len(g) + k]
                row[f'dose_{tag}'] = dd[k]; row[f'auc_true_{tag}'] = a_true; row[f'auc_{tag}'] = float(auc[k])
                row[f'nrmse_{tag}'] = float(np.sqrt(np.mean((c12[k] - y12) ** 2)) / np.mean(y12))
        for k, i in enumerate(g):
            row = rows[len(rows) - len(g) + k]
            for j, nm in enumerate(FIT):
                row[f'log_{nm}_true'] = float(np.log(TP.loc[i][nm])); row[f'log_{nm}_ebe'] = float(np.log(TV[k, j]) + eta[k, j])
        print(f'  {form}: {len(g)} patients', flush=True)
    pd.DataFrame(rows).to_csv(out, index=False); print('POPEBE_DONE', out)

if __name__ == '__main__':
    main()
