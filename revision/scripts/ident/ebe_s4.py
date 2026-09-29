#!/usr/bin/env python3
"""MAP-BE (EBE) on scenario 4 under the TRUE population model, from the same 3 points
the encoder sees, then the Visit-2 counterfactual at the new dose.

This is what a pharmacometrician with a perfectly fitted population model would do:
known structural ODE, known typical values and covariate models (Ht on Km, Ht and CYP
on Vmax, formulation on Ktr and Vc), known IIV including the Km-Vc omega block, known
residual error.  Only the individual random effects are estimated.

Fitted: Ktr, Q, Vc, Vp, Vmax, Km (6 etas).  CL is inert under Michaelis-Menten
(landmine 1.5) -- it never enters the ODE, so its MAP eta is 0 and it is not fitted.
Objective: -log p(y | eta) - log p(eta), residual sd = 0.03 + 0.03*C (the generator's).
Also reports the true-parameter replay (validation) and the prior-only prediction.

    ebe_s4.py <cohort> <n_patients> <seed> <out.csv> [shard i/n]
"""
import os, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch
from scipy.optimize import minimize
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE))); sys.path.insert(0, HERE)
from gen_tacro_film import POPULATION_PARAMS as P, IPV_OMEGA as OM, RESIDUAL_ERROR_ADD_SD as EA, RESIDUAL_ERROR_PROP_SD as EB
from oracle_floor import BatchPK, NBR_SS, OBS_REL, ENC_T
from mle_km_s4 import replay
torch.set_num_threads(1)

FIT = ['Ktr', 'Q', 'Vc', 'Vp', 'Vmax', 'Km']
OM_KM = float(np.sqrt(max(OM['Km'] ** 2 - (0.1683 * P['theta_Km_HT']) ** 2, 1e-6)))
om = np.array([OM['Ktr'], OM['Q'], OM['Vc'], OM['Vp'], OM['Vmax'], OM_KM])
Omega = np.diag(om ** 2)
iv, ik = FIT.index('Vc'), FIT.index('Km')
Omega[iv, ik] = Omega[ik, iv] = P['rho_Km_Vc'] * om[iv] * om[ik]
PREC = torch.tensor(np.linalg.inv(Omega), dtype=torch.float64)


def typical(form, cyp, ht):
    study = 1.0 if form == 'Prograf' else 0.0
    he = (ht / 35.0) ** P['theta4_CL_HT']
    return {'Ktr': P['theta1_Ktr'] * P['theta2_Ktr_study'] ** study,
            'CL': P['theta3_CL'] * he * P['theta5_CL_CYP'] ** cyp,
            'Q': P['Q'], 'Vc': P['theta6_Vc'] * P['theta7_Vc_study'] ** study, 'Vp': P['Vp'],
            'Vmax': P['theta_Vmax'] * he * P['theta5_CL_CYP'] ** cyp,
            'Km': P['theta_Km'] * (ht / 35.0) ** P['theta_Km_HT']}


def simulate(params, form, ht, cyp, dose, rel_times, K):
    """Concentration (ng/mL) at rel_times after the last dose, for K parameter sets."""
    pk = BatchPK(K, formulation=form, hematocrit=ht, cyp_status='expresser' if cyp else 'non_expresser', scenario=4)
    pk.individual_params = params
    pk.dose_mg = float(dose)
    dos = ([j * 24 for j in range(NBR_SS + 1)] if form == 'Advagraf'
           else [24 * NBR_SS - 12 * (NBR_SS - j) for j in range(NBR_SS + 1)])
    rel = torch.as_tensor(rel_times, dtype=torch.float32)
    tp = torch.unique(torch.cat([torch.arange(1.0, 24.0 * NBR_SS, 2.0), rel + 24 * NBR_SS]))
    c = pk.simulate(dos, tp, resample=False) * 1000.0
    idx = [int(torch.argmin(torch.abs(tp - (t + 24 * NBR_SS)))) + 1 for t in rel.tolist()]  # +1: row 0 is the t=0 placeholder
    return c[idx, :]


def params_from(eta, tv, K=1):
    e = eta if torch.is_tensor(eta) else torch.tensor(eta)
    out = {n: (tv[n] * torch.exp(e[..., i].float())).reshape(K) for i, n in enumerate(FIT)}
    out['CL'] = torch.full((K,), float(tv['CL']))
    return out


def auc(params, form, ht, cyp, dose):
    hi = 12.0 if form == 'Prograf' else 24.0
    t = np.round(np.arange(0.0, hi + 1e-9, 0.05), 4)
    c = simulate(params, form, ht, cyp, dose, t, 1).detach().numpy().ravel()
    return float(np.trapezoid(c, t))


def fit(y, tv, form, ht, cyp, dose):
    yt = torch.tensor(y, dtype=torch.float64)
    def f(x):
        e = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        pred = simulate(params_from(e, tv), form, ht, cyp, dose, ENC_T, 1).double().ravel()
        sd = EA + EB * pred
        obj = 0.5 * (((yt - pred) / sd) ** 2).sum() + torch.log(sd).sum() + 0.5 * e @ PREC @ e
        obj.backward()
        return float(obj), e.grad.numpy().astype(float)
    best = None
    for x0 in (np.zeros(6), 0.5 * om * np.array([1, -1, 1, -1, 1, -1])):   # 2 starts: ridge guard
        r = minimize(f, x0, jac=True, method='L-BFGS-B', options={'maxiter': 200})
        if best is None or r.fun < best.fun: best = r
    return best


def main():
    cohort, n, seed, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    shard = sys.argv[5] if len(sys.argv) > 5 else "0/1"
    si, sn = (int(v) for v in shard.split('/'))
    R = f'results/exp_film_run/{cohort}'
    truth = pd.read_csv(f'{R}/confound_truth.csv'); ob = pd.read_csv(f'{R}/virtual_cohort_film_test.csv')
    ob['DVn'] = pd.to_numeric(ob.DV, errors='coerce'); ob['Tn'] = pd.to_numeric(ob.TIME, errors='coerce')
    TP = replay(len(truth), 4)
    m = TP.join(truth.set_index('ID')[['CL_base', 'confound_par', 'HT']], rsuffix='_rec')
    assert np.allclose(m.CL, m.CL_base) and np.allclose(m.Vc, m.confound_par) and np.allclose(m.HT, m.HT_rec), 'replay mismatch'
    te = truth[truth.split == 'test'].ID.values.copy()
    np.random.RandomState(seed).shuffle(te); ids = te[:n][si::sn]
    rows = []
    for k, pid in enumerate(ids, 1):
        t0 = time.time()
        tr = truth[truth.ID == pid].iloc[0]; Pt = TP.loc[pid]
        form, ht, cyp = Pt.form, float(Pt.HT), int(Pt.cyp)
        tv = typical(form, cyp, ht)
        g = {v: ob[(ob.ID == pid) & (ob.VISIT == v) & ob.DVn.notna()].sort_values('Tn') for v in (1, 2)}
        rel = {v: g[v].Tn.values - g[v].Tn.values[0] for v in (1, 2)}
        y12 = {v: g[v].DVn.values for v in (1, 2)}
        y3 = np.array([y12[1][np.argmin(np.abs(rel[1] - t))] for t in ENC_T])
        r = fit(y3, tv, form, ht, cyp, tr.d1)
        eta = torch.tensor(r.x)
        true_p = {n_: torch.tensor([float(Pt[n_])]) for n_ in FIT + ['CL']}
        row = dict(ID=pid, form=form, ht=ht, cyp=cyp, d1=tr.d1, d2=tr.d2, converged=bool(r.success), nit=r.nit,
                   auc1_true=float(g[1].AUC.iloc[0]), auc2_true=float(g[2].AUC.iloc[0]))
        for tag, pp in (('ebe', params_from(eta, tv)), ('prior', params_from(torch.zeros(6), tv)), ('replay', true_p)):
            row[f'auc1_{tag}'] = auc(pp, form, ht, cyp, tr.d1); row[f'auc2_{tag}'] = auc(pp, form, ht, cyp, tr.d2)
            for v, d in ((1, tr.d1), (2, tr.d2)):
                q = simulate(pp, form, ht, cyp, d, rel[v], 1).detach().numpy().ravel()
                row[f'nrmse{v}_{tag}'] = float(np.sqrt(np.mean((q - y12[v]) ** 2)) / np.mean(y12[v]))
        for i, n_ in enumerate(FIT):
            row[f'eta_{n_}'] = float(eta[i]); row[f'log_{n_}_true'] = float(np.log(Pt[n_]))
            row[f'log_{n_}_ebe'] = float(np.log(tv[n_]) + eta[i]); row[f'log_{n_}_tv'] = float(np.log(tv[n_]))
        rows.append(row)
        print(f'  [{si}] {k}/{len(ids)} ID {pid} {time.time() - t0:.1f}s conv={r.success}', flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)
    print('EBE_DONE', out)


if __name__ == '__main__':
    main()
