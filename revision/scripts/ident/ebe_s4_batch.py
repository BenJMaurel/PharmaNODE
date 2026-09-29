#!/usr/bin/env python3
"""Batched MAP-BE (EBE) on scenario 4 under the TRUE population model.

Same estimand as ebe_s4.py, fitted for a whole formulation group per ODE solve:
Levenberg-Marquardt on the weighted residuals of the 3 encoder observations plus the
prior (omega block included), with a finite-difference Jacobian whose perturbed
trajectories share the batch -- so they share the adaptive step sequence and the
solver error cancels in the difference.  Residual weights use sd = 0.03 + 0.03*C at the
current prediction (the usual WLS form of MAP-BE; the log-sd term is dropped).

Validation built in: the true parameters (RNG replay) must reproduce the generator's
AUC -- this is what caught a dosing-grid bug in the first version.

    ebe_s4_batch.py <cohort> <n_patients> <seed> <out.csv>
"""
import os, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE))); sys.path.insert(0, HERE)
from gen_tacro_film import POPULATION_PARAMS as P, IPV_OMEGA as OM, RESIDUAL_ERROR_ADD_SD as EA, RESIDUAL_ERROR_PROP_SD as EB
from oracle_floor import BatchPK, NBR_SS, ENC_T
from mle_km_s4 import replay
torch.set_num_threads(2)

FIT = ['Ktr', 'Q', 'Vc', 'Vp', 'Vmax', 'Km']            # CL inert under MM (landmine 1.5)
OM_KM = float(np.sqrt(max(OM['Km'] ** 2 - (0.1683 * P['theta_Km_HT']) ** 2, 1e-6)))
om = np.array([OM['Ktr'], OM['Q'], OM['Vc'], OM['Vp'], OM['Vmax'], OM_KM])
Omega = np.diag(om ** 2); iv, ik = FIT.index('Vc'), FIT.index('Km')
Omega[iv, ik] = Omega[ik, iv] = P['rho_Km_Vc'] * om[iv] * om[ik]
PREC = np.linalg.inv(Omega)
LPREC = np.linalg.cholesky(PREC).T                       # eta' PREC eta = ||LPREC eta||^2


def typical(form, cyp, ht):
    s = 1.0 if form == 'Prograf' else 0.0; he = (ht / 35.0) ** P['theta4_CL_HT']
    return np.array([P['theta1_Ktr'] * P['theta2_Ktr_study'] ** s, P['Q'],
                     P['theta6_Vc'] * P['theta7_Vc_study'] ** s, P['Vp'],
                     P['theta_Vmax'] * he * P['theta5_CL_CYP'] ** cyp,
                     P['theta_Km'] * (ht / 35.0) ** P['theta_Km_HT']]), P['theta3_CL'] * he * P['theta5_CL_CYP'] ** cyp


def dosing(form):
    return ([j * 24.0 for j in range(NBR_SS + 1)] if form == 'Advagraf'
            else [24.0 * NBR_SS - 12.0 * (NBR_SS - j) for j in range(NBR_SS + 1)])


def simulate(theta, cl, dose, form, rel):
    """theta [M,6] natural-scale params, cl [M], dose [M] -> conc [len(rel), M] in ng/mL.
    The grid CONTAINS every dosing time, so the state is carried exactly to each dose."""
    M = theta.shape[0]
    pk = BatchPK(M, formulation=form, hematocrit=35.0, cyp_status='non_expresser', scenario=4)
    pk.individual_params = {n: torch.as_tensor(theta[:, i], dtype=torch.float32) for i, n in enumerate(FIT)}
    pk.individual_params['CL'] = torch.as_tensor(cl, dtype=torch.float32)
    pk.dose_mg = torch.as_tensor(dose, dtype=torch.float32)
    dos = dosing(form)
    tgt = torch.as_tensor(np.asarray(rel) + 24.0 * NBR_SS, dtype=torch.float32)
    tp = torch.unique(torch.cat([torch.arange(1.0, 24.0 * NBR_SS, 1.0), torch.tensor(dos[1:]), tgt]))
    tp = tp[tp > 0]
    with torch.no_grad():
        c = pk.simulate(dos, tp, resample=False) * 1000.0
    idx = [int(torch.argmin(torch.abs(tp - t))) + 1 for t in tgt.tolist()]
    return c[idx, :].double().numpy()


def fit_group(y, tv, cl, dose, form, iters=40, h=1e-3):
    K = y.shape[0]; eta = np.zeros((K, 6)); lam = np.full(K, 1e-2)
    def objective(e):
        f = simulate(tv * np.exp(e), cl, dose, form, ENC_T).T               # [K,3]
        sd = EA + EB * f
        return (((y - f) / sd) ** 2).sum(1) + np.einsum('ki,ij,kj->k', e, PREC, e), f
    obj, f0 = objective(eta)
    for it in range(iters):
        pert = eta[:, None, :] + h * np.concatenate([np.zeros((1, 6)), np.eye(6)])[None]   # [K,7,6]
        F = simulate((tv[:, None, :] * np.exp(pert)).reshape(-1, 6), np.repeat(cl, 7),
                     np.repeat(dose, 7), form, ENC_T).T.reshape(K, 7, 3)
        f0 = F[:, 0]; J = (F[:, 1:] - f0[:, None]) / h                        # [K,6,3]
        W = 1.0 / (EA + EB * f0)
        Jw = J * W[:, None, :]; rw = (y - f0) * W                            # [K,6,3],[K,3]
        A = np.einsum('kip,kjp->kij', Jw, Jw) + PREC[None]
        g = np.einsum('kip,kp->ki', Jw, rw) - eta @ PREC
        step = np.linalg.solve(A + lam[:, None, None] * np.eye(6)[None], g[..., None])[..., 0]
        new = eta + step; nobj, _ = objective(new)
        ok = nobj < obj
        eta[ok] = new[ok]; obj[ok] = nobj[ok]
        lam = np.where(ok, lam * 0.3, lam * 10.0)
        if np.max(np.abs(step[ok]), initial=0) < 1e-5 and it > 5: break
    return eta, obj, it + 1


def main():
    cohort, n, seed, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    R = f'results/exp_film_run/{cohort}'
    truth = pd.read_csv(f'{R}/confound_truth.csv'); ob = pd.read_csv(f'{R}/virtual_cohort_film_test.csv')
    ob['DVn'] = pd.to_numeric(ob.DV, errors='coerce'); ob['Tn'] = pd.to_numeric(ob.TIME, errors='coerce')
    TP = replay(len(truth), 4)
    m = TP.join(truth.set_index('ID')[['CL_base', 'confound_par', 'HT']], rsuffix='_rec')
    assert np.allclose(m.CL, m.CL_base) and np.allclose(m.Vc, m.confound_par) and np.allclose(m.HT, m.HT_rec)
    te = truth[truth.split == 'test'].ID.values.copy(); np.random.RandomState(seed).shuffle(te); ids = te[:n]
    rel12 = np.array([0., 0.33, 0.67, 1., 1.5, 2., 3., 4., 6., 9., 12., 24.])
    rows = []
    for form in ('Advagraf', 'Prograf'):
        g_ids = [i for i in ids if TP.loc[i].form == form]
        if not g_ids: continue
        t0 = time.time()
        TV, CLt, D1, D2, Y3, Y12, TRUE = [], [], [], [], [], [], []
        for pid in g_ids:
            Pt = TP.loc[pid]; tr = truth[truth.ID == pid].iloc[0]
            tv, clt = typical(form, int(Pt.cyp), float(Pt.HT)); TV.append(tv); CLt.append(clt)
            D1.append(tr.d1); D2.append(tr.d2); TRUE.append([float(Pt[k]) for k in FIT] + [float(Pt.CL)])
            yy = {}
            for v in (1, 2):
                gv = ob[(ob.ID == pid) & (ob.VISIT == v) & ob.DVn.notna()].sort_values('Tn')
                yy[v] = (gv.DVn.values, float(gv.AUC.iloc[0]))
            Y12.append(yy); Y3.append([yy[1][0][int(np.argmin(np.abs(rel12 - t)))] for t in ENC_T])
        TV, CLt, D1, D2, Y3, TRUE = map(np.array, (TV, CLt, D1, D2, Y3, TRUE))
        eta, obj, nit = fit_group(Y3, TV, CLt, D1, form)
        hi = 12.0 if form == 'Prograf' else 24.0
        ta = np.round(np.arange(0.0, hi + 1e-9, 0.02), 4)
        sets = {'ebe': (TV * np.exp(eta), CLt), 'prior': (TV, CLt), 'replay': (TRUE[:, :6], TRUE[:, 6])}
        pred = {}
        for tag, (th, cl) in sets.items():
            for v, d in ((1, D1), (2, D2)):
                pred[(tag, v, 'auc')] = np.trapezoid(simulate(th, cl, d, form, ta), ta, axis=0)
                pred[(tag, v, 'c12')] = simulate(th, cl, d, form, rel12).T
        for k, pid in enumerate(g_ids):
            r = dict(ID=pid, form=form, ht=float(TP.loc[pid].HT), cyp=int(TP.loc[pid].cyp), d1=D1[k], d2=D2[k], lm_iters=nit)
            for v in (1, 2):
                y, auc_true = Y12[k][v]; r[f'auc{v}_true'] = auc_true
                for tag in sets:
                    r[f'auc{v}_{tag}'] = float(pred[(tag, v, 'auc')][k])
                    q = pred[(tag, v, 'c12')][k]
                    r[f'nrmse{v}_{tag}'] = float(np.sqrt(np.mean((q - y) ** 2)) / np.mean(y))
            for i, nm in enumerate(FIT):
                r[f'log_{nm}_true'] = float(np.log(TRUE[k, i])); r[f'log_{nm}_ebe'] = float(np.log(TV[k, i]) + eta[k, i])
                r[f'log_{nm}_tv'] = float(np.log(TV[k, i]))
            rows.append(r)
        print(f'  {form}: {len(g_ids)} patients, {nit} LM iterations, {time.time() - t0:.0f}s', flush=True)
    pd.DataFrame(rows).to_csv(out, index=False); print('EBE_DONE', out)


if __name__ == '__main__':
    main()
