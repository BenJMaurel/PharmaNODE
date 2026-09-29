#!/usr/bin/env python3
"""EBE on TWO observed visits (3 points each, at two different doses) under the true
scenario-4 model -- does a second dose break the Vmax-Km ridge?

The fit uses V1 (d1) and V2 (d2) of the in-range cohort confound_vc00_s4: 3 points each at
the encoder's times (0, 1, 3 h), so each visit carries the same information the encoder gets.
Predictions:
  in-range V2      NOT a counterfactual (its own points were fitted) -- the ceiling
  below / above    GENUINE counterfactuals: the same patients' out-of-range visit-2 dose from
                   confound_vc00_s4_lo / _hi (identical physiology, checked by the pairing script)
Compared, patient by patient, with the one-visit EBE already on disk.

    ebe_s4_2visits.py <n_patients> <seed> <out.csv>
"""
import sys, time, numpy as np, pandas as pd
sys.path.insert(0, 'scripts/ident'); sys.path.insert(0, '.')
from ebe_s4_batch import simulate, typical, FIT, PREC, EA, EB, ENC_T
from mle_km_s4 import replay
REL = np.array([0., 0.33, 0.67, 1., 1.5, 2., 3., 4., 6., 9., 12., 24.])

def obs(ob, pid, v):
    g = ob[(ob.ID == pid) & (ob.VISIT == v) & ob.DVn.notna()].sort_values('Tn')
    return g.DVn.values, float(g.AUC.iloc[0])

def load(c):
    t = pd.read_csv(f'results/exp_film_run/{c}/confound_truth.csv').set_index('ID')
    o = pd.read_csv(f'results/exp_film_run/{c}/virtual_cohort_film_test.csv')
    o['DVn'] = pd.to_numeric(o.DV, errors='coerce'); o['Tn'] = pd.to_numeric(o.TIME, errors='coerce')
    return t, o

def fit2(Y, D, tv, cl, form, iters=40, h=1e-3):
    """Y [K,2,3] observations at doses D [K,2]; LM on weighted residuals + prior."""
    K = Y.shape[0]; eta = np.zeros((K, 6)); lam = np.full(K, 1e-2)
    def pred(e):                                            # [K,2,3]
        return np.stack([simulate(tv * np.exp(e), cl, D[:, v], form, ENC_T).T for v in (0, 1)], 1)
    def objective(e):
        f = pred(e); sd = EA + EB * f
        return (((Y - f) / sd) ** 2).sum((1, 2)) + np.einsum('ki,ij,kj->k', e, PREC, e)
    obj = objective(eta)
    for it in range(iters):
        pert = eta[:, None, :] + h * np.concatenate([np.zeros((1, 6)), np.eye(6)])[None]
        th = (tv[:, None, :] * np.exp(pert)).reshape(-1, 6)
        F = np.stack([simulate(th, np.repeat(cl, 7), np.repeat(D[:, v], 7), form, ENC_T).T.reshape(K, 7, 3)
                      for v in (0, 1)], 2).reshape(K, 7, 6)          # [K,7,6 obs]
        f0 = F[:, 0]; J = (F[:, 1:] - f0[:, None]) / h                 # [K,6,6]
        W = 1.0 / (EA + EB * f0); Jw = J * W[:, None, :]; rw = (Y.reshape(K, 6) - f0) * W
        A = np.einsum('kip,kjp->kij', Jw, Jw) + PREC[None]
        g = np.einsum('kip,kp->ki', Jw, rw) - eta @ PREC
        step = np.linalg.solve(A + lam[:, None, None] * np.eye(6)[None], g[..., None])[..., 0]
        new = eta + step; nobj = objective(new); ok = nobj < obj
        eta[ok] = new[ok]; obj[ok] = nobj[ok]; lam = np.where(ok, lam * 0.3, lam * 10.0)
        if np.max(np.abs(step[ok]), initial=0) < 1e-5 and it > 5: break
    return eta, it + 1

def main():
    n, seed, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    TP = replay(1800, 4)
    T0, O0 = load('confound_vc00_s4'); TL, OL = load('confound_vc00_s4_lo'); TH, OH = load('confound_vc00_s4_hi')
    te = T0[T0.split == 'test'].index.values.copy(); np.random.RandomState(seed).shuffle(te); ids = te[:n]
    rows = []
    for form in ('Advagraf', 'Prograf'):
        g = [i for i in ids if TP.loc[i].form == form]; t0 = time.time()
        TV = np.stack([typical(form, int(TP.loc[i].cyp), float(TP.loc[i].HT))[0] for i in g])
        CL = np.array([typical(form, int(TP.loc[i].cyp), float(TP.loc[i].HT))[1] for i in g])
        D = np.array([[T0.loc[i].d1, T0.loc[i].d2] for i in g])
        Y = np.stack([[[obs(O0, i, v)[0][int(np.argmin(np.abs(REL - t)))] for t in ENC_T] for v in (1, 2)] for i in g])
        eta, nit = fit2(Y, D, TV, CL, form)
        th = TV * np.exp(eta)
        hi_t = 12.0 if form == 'Prograf' else 24.0; ta = np.round(np.arange(0.0, hi_t + 1e-9, 0.02), 4)
        targets = {'in': (D[:, 1], [obs(O0, i, 2) for i in g]),
                   'lo': (np.array([TL.loc[i].d2 for i in g]), [obs(OL, i, 2) for i in g]),
                   'hi': (np.array([TH.loc[i].d2 for i in g]), [obs(OH, i, 2) for i in g])}
        res = {}
        for tag, (dd, ys) in targets.items():
            res[tag] = (np.trapezoid(simulate(th, CL, dd, form, ta), ta, axis=0), simulate(th, CL, dd, form, REL).T, dd, ys)
        for k, i in enumerate(g):
            r = dict(ID=i, form=form, d1=D[k, 0], d2_in=D[k, 1], lm_iters=nit)
            for tag, (auc, c12, dd, ys) in res.items():
                y12, a_true = ys[k]
                r[f'd2_{tag}'] = dd[k]; r[f'auc_true_{tag}'] = a_true; r[f'auc_2v_{tag}'] = float(auc[k])
                r[f'nrmse_2v_{tag}'] = float(np.sqrt(np.mean((c12[k] - y12) ** 2)) / np.mean(y12))
            for j, nm in enumerate(FIT):
                r[f'log_{nm}_true'] = float(np.log(TP.loc[i][nm])); r[f'log_{nm}_2v'] = float(np.log(TV[k, j]) + eta[k, j])
            rows.append(r)
        print(f'  {form}: {len(g)} patients, {nit} LM iterations, {time.time() - t0:.0f}s', flush=True)
    pd.DataFrame(rows).to_csv(out, index=False); print('EBE2_DONE', out)

if __name__ == '__main__':
    main()
