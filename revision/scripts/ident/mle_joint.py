"""Are Vmax and Km JOINTLY identifiable from the observations?

Km alone is sharply determined (R^2 = 0.92 from 3 points). This profiles the 2-D likelihood
over (log Vmax, log Km) with the remaining parameters at truth, which is the classic test of
whether the pair trades off along a ridge.

Reports, per observation design:
  R^2 of the joint MLE and of the EBE (conditional mode with the true log-normal priors)
  the correlation of the two estimates implied by the Hessian at the MLE -- the ridge
  eta-shrinkage of the EBE, the standard NLME diagnostic (>30% means unreliable)
"""
import argparse, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch

sys.path.insert(0, '/Users/benjaminmaurel/Documents/PharmaNODE')
sys.path.insert(0, '/private/tmp/claude-501/-Users-benjaminmaurel-Documents-PharmaNODE/a847a5f5-d0e8-430d-965d-120f8364055e/scratchpad')
from gen_tacro_film import RESIDUAL_ERROR_ADD_SD, RESIDUAL_ERROR_PROP_SD
from oracle_floor import BatchPK, NBR_SS, OBS_REL, ENC_T, time_grid

KM_TV, KM_OM = 0.01, np.sqrt(0.10)
VMAX_TV_BASE, VMAX_OM = 0.212, np.sqrt(0.08)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-patients', type=int, default=40)
    ap.add_argument('--n', type=int, default=41, help='grid points per axis')
    ap.add_argument('--cohort', default='confound_vc00_10k')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    root = f'/Users/benjaminmaurel/Documents/PharmaNODE/results/exp_film_run/{a.cohort}'
    truth = pd.read_csv(f'{root}/confound_truth.csv'); truth = truth[truth.split == 'test']
    obs = pd.read_csv(f'{root}/virtual_cohort_film_test.csv')
    obs['DVn'] = pd.to_numeric(obs['DV'], errors='coerce')
    obs['Tn'] = pd.to_numeric(obs['TIME'], errors='coerce')
    TP = pd.read_csv('/tmp/true_params_10k.csv').set_index('ID')

    rng = np.random.RandomState(a.seed)
    ids = truth.ID.values.copy(); rng.shuffle(ids); ids = ids[:a.n_patients]

    gk = np.linspace(-4 * KM_OM, 4 * KM_OM, a.n)      # offsets in log space
    gv = np.linspace(-4 * VMAX_OM, 4 * VMAX_OM, a.n)
    KK, VV = np.meshgrid(gk, gv, indexing='ij')
    flat_k, flat_v = KK.ravel(), VV.ravel()
    K = flat_k.size

    recs = []
    for n, pid in enumerate(ids, 1):
        tr = truth[truth.ID == pid].iloc[0]
        g = obs[(obs.ID == pid) & (obs.VISIT == 1) & obs.DVn.notna()].sort_values('Tn')
        t = g.Tn.values - g.Tn.values[0]; y = g.DVn.values
        P = TP.loc[pid]
        form = g.DRUG.iloc[0]; ht = float(g.HT.iloc[0]); cyp = int(g.CYP.iloc[0])
        tv_v = VMAX_TV_BASE * (2.0 ** cyp)

        pk = BatchPK(K, formulation=form, hematocrit=ht,
                     cyp_status='expresser' if cyp else 'non_expresser', scenario=3)
        pk.individual_params = {
            'Ktr': torch.full((K,), float(P.Ktr)), 'CL': torch.full((K,), float(P.CL)),
            'Q': torch.full((K,), float(P.Q)), 'Vc': torch.full((K,), float(P.Vc)),
            'Vp': torch.full((K,), float(P.Vp)),
            'Vmax': torch.tensor(tv_v * np.exp(flat_v), dtype=torch.float32),
            'Km': torch.tensor(KM_TV * np.exp(flat_k), dtype=torch.float32)}
        pk.dose_mg = float(tr['d1'])
        dosing = ([i * 24 for i in range(NBR_SS + 1)] if form == 'Advagraf'
                  else [24 * NBR_SS - 12 * (NBR_SS - i) for i in range(NBR_SS + 1)])
        tpg = time_grid(12.0 if form == 'Prograf' else 24.0)
        with torch.no_grad():
            conc = pk.simulate(dosing, tpg, resample=False) * 1000.0
        trel = tpg - 24 * NBR_SS

        rec = {'ID': pid, 'lk_true': float(np.log(P.Km / KM_TV)),
               'lv_true': float(np.log(P.Vmax / tv_v))}
        for tag, times in (('3obs', ENC_T), ('12obs', list(OBS_REL.numpy()))):
            idx = [int(torch.argmin(torch.abs(trel - float(tt)))) for tt in times]
            yy = torch.tensor([float(y[np.argmin(np.abs(t - float(tt)))]) for tt in times]).unsqueeze(1)
            pred = conc[idx, :]
            sd = RESIDUAL_ERROR_ADD_SD + RESIDUAL_ERROR_PROP_SD * pred
            ll = (-0.5 * ((yy - pred) / sd) ** 2 - torch.log(sd)).sum(dim=0).numpy()
            lp = -0.5 * (flat_k / KM_OM) ** 2 - 0.5 * (flat_v / VMAX_OM) ** 2
            i_mle = int(np.argmax(ll)); i_ebe = int(np.argmax(ll + lp))
            rec[f'mle_k_{tag}'] = flat_k[i_mle]; rec[f'mle_v_{tag}'] = flat_v[i_mle]
            rec[f'ebe_k_{tag}'] = flat_k[i_ebe]; rec[f'ebe_v_{tag}'] = flat_v[i_ebe]
            # ridge orientation: correlation implied by the likelihood Hessian at the MLE
            L = ll.reshape(a.n, a.n)
            ik, iv = np.unravel_index(i_mle, (a.n, a.n))
            ik = min(max(ik, 1), a.n - 2); iv = min(max(iv, 1), a.n - 2)
            hk = gk[1] - gk[0]; hv = gv[1] - gv[0]
            fkk = (L[ik + 1, iv] - 2 * L[ik, iv] + L[ik - 1, iv]) / hk ** 2
            fvv = (L[ik, iv + 1] - 2 * L[ik, iv] + L[ik, iv - 1]) / hv ** 2
            fkv = (L[ik + 1, iv + 1] - L[ik + 1, iv - 1] - L[ik - 1, iv + 1] + L[ik - 1, iv - 1]) / (4 * hk * hv)
            H = -np.array([[fkk, fkv], [fkv, fvv]])
            try:
                C = np.linalg.inv(H)
                rec[f'corr_{tag}'] = float(C[0, 1] / np.sqrt(C[0, 0] * C[1, 1]))
                rec[f'se_k_{tag}'] = float(np.sqrt(abs(C[0, 0])))
            except Exception:
                rec[f'corr_{tag}'] = np.nan; rec[f'se_k_{tag}'] = np.nan
        recs.append(rec)
        if n % 10 == 0: print(f'  {n}/{len(ids)}', flush=True)

    d = pd.DataFrame(recs)
    def r2(true, est): return 1 - np.mean((true - est) ** 2) / np.var(true)

    print(f'\nn = {len(d)} | Vmax and Km BOTH unknown, other parameters at truth')
    print(f'prior sd: log Km {KM_OM:.3f}, log Vmax {VMAX_OM:.3f}\n')
    print(f"{'design':8s} {'R2 Km(MLE)':>11s} {'R2 Km(EBE)':>11s} {'R2 Vmax(EBE)':>13s} "
          f"{'ridge corr':>11s} {'med SE(lKm)':>12s} {'Km shrinkage':>13s}")
    print('-' * 86)
    for tag in ('3obs', '12obs'):
        shr = 1 - np.std(d[f'ebe_k_{tag}']) / KM_OM
        print(f'{tag:8s} {r2(d.lk_true.values, d[f"mle_k_{tag}"].values):11.3f} '
              f'{r2(d.lk_true.values, d[f"ebe_k_{tag}"].values):11.3f} '
              f'{r2(d.lv_true.values, d[f"ebe_v_{tag}"].values):13.3f} '
              f'{np.nanmedian(d[f"corr_{tag}"]):11.3f} {np.nanmedian(d[f"se_k_{tag}"]):12.3f} '
              f'{shr:12.1%}')
    d.to_csv('/tmp/mle_joint.csv', index=False)
    print('\nwrote /tmp/mle_joint.csv')


if __name__ == '__main__':
    main()
