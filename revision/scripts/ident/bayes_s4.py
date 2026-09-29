#!/usr/bin/env python3
"""Bayes-optimal V2 prediction (posterior MEAN) under the true scenario-4 model.

The EBE is the posterior mode; for prediction the optimal point estimate under squared
error is the posterior mean, and it is the floor for ANY estimator given the same 3
points.  Importance sampling with a Laplace proposal centred on each patient's EBE:
q = N(eta_hat, c^2 * (Jw'Jw + PREC)^-1), c = 1.5 (heavier than the posterior, so the
weights stay bounded), exact log-likelihood (log-sd term included) and exact prior.
Reports the effective sample size so a collapsed estimate is visible.

    bayes_s4.py <ebe.csv> <out.csv> [S]
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'scripts/ident'); sys.path.insert(0, '.')
from ebe_s4_batch import simulate, typical, FIT, PREC, EA, EB, ENC_T
ebe_csv, out = sys.argv[1], sys.argv[2]; S = int(sys.argv[3]) if len(sys.argv) > 3 else 200
d = pd.read_csv(ebe_csv)
obs = pd.read_csv('results/exp_film_run/confound_vc00_s4/virtual_cohort_film_test.csv')
obs['DVn'] = pd.to_numeric(obs.DV, errors='coerce'); obs['Tn'] = pd.to_numeric(obs.TIME, errors='coerce')
rel12 = np.array([0., 0.33, 0.67, 1., 1.5, 2., 3., 4., 6., 9., 12., 24.])
rng = np.random.RandomState(0); c = 1.5; h = 1e-3
res = []
for form in ('Advagraf', 'Prograf'):
    g = d[d.form == form].reset_index(drop=True); K = len(g)
    if K == 0: continue
    TV = np.stack([typical(form, int(r.cyp), float(r.ht))[0] for r in g.itertuples()])
    CL = np.array([typical(form, int(r.cyp), float(r.ht))[1] for r in g.itertuples()])
    ETA = np.stack([g[f'log_{n}_ebe'] - g[f'log_{n}_tv'] for n in FIT], 1)
    Y3, Y12 = [], []
    for pid in g.ID:
        y1 = obs[(obs.ID == pid) & (obs.VISIT == 1) & obs.DVn.notna()].sort_values('Tn').DVn.values
        y2 = obs[(obs.ID == pid) & (obs.VISIT == 2) & obs.DVn.notna()].sort_values('Tn').DVn.values
        Y3.append([y1[int(np.argmin(np.abs(rel12 - t)))] for t in ENC_T]); Y12.append(y2)
    Y3 = np.array(Y3)
    # Laplace covariance at the EBE
    pert = ETA[:, None, :] + h * np.concatenate([np.zeros((1, 6)), np.eye(6)])[None]
    F = simulate((TV[:, None, :] * np.exp(pert)).reshape(-1, 6), np.repeat(CL, 7), np.repeat(g.d1.values, 7), form, ENC_T).T.reshape(K, 7, 3)
    J = (F[:, 1:] - F[:, :1]) / h; W = 1 / (EA + EB * F[:, 0])
    Jw = J * W[:, None, :]
    Hs = np.einsum('kip,kjp->kij', Jw, Jw) + PREC[None]
    Sig = np.linalg.inv(Hs) * c ** 2; Lc = np.linalg.cholesky(Sig)
    Z = rng.normal(size=(K, S, 6)); E = ETA[:, None, :] + np.einsum('kij,ksj->ksi', Lc, Z)   # [K,S,6]
    th = (TV[:, None, :] * np.exp(E)).reshape(-1, 6); clr = np.repeat(CL, S)
    f = simulate(th, clr, np.repeat(g.d1.values, S), form, ENC_T).T.reshape(K, S, 3)
    sd = EA + EB * f
    loglik = -0.5 * (((Y3[:, None, :] - f) / sd) ** 2).sum(-1) - np.log(sd).sum(-1)
    logpri = -0.5 * np.einsum('ksi,ij,ksj->ks', E, PREC, E)
    dz = E - ETA[:, None, :]
    logq = -0.5 * np.einsum('ksi,kij,ksj->ks', dz, np.linalg.inv(Sig), dz)
    lw = loglik + logpri - logq; lw -= lw.max(1, keepdims=True); w = np.exp(lw); w /= w.sum(1, keepdims=True)
    ess = 1 / (w ** 2).sum(1)
    hi = 12.0 if form == 'Prograf' else 24.0; ta = np.round(np.arange(0.0, hi + 1e-9, 0.05), 4)
    for v, dose in ((1, g.d1.values), (2, g.d2.values)):
        dd = np.repeat(dose, S)
        a = np.trapezoid(simulate(th, clr, dd, form, ta), ta, axis=0).reshape(K, S)
        g[f'auc{v}_pm'] = (w * a).sum(1)
        if v == 2:
            cc = simulate(th, clr, dd, form, rel12).T.reshape(K, S, 12)
            q = (w[..., None] * cc).sum(1)
            g['nrmse2_pm'] = [float(np.sqrt(np.mean((q[k] - Y12[k]) ** 2)) / np.mean(Y12[k])) for k in range(K)]
    g['ess'] = ess; res.append(g)
    print(f'  {form}: K={K} S={S}  ESS median {np.median(ess):.0f}  min {ess.min():.0f}', flush=True)
pd.concat(res).to_csv(out, index=False); print('BAYES_DONE', out)
