"""How well is Km determined, by maximum likelihood, with the true ODE and every other
parameter known?

For each patient: fix Ktr, Q, Vc, Vp, Vmax to their TRUE values, leave Km the only unknown,
and profile the likelihood of the visit-1 observations over a grid of Km. The residual model
is the generator's own: sd = 0.03 + 0.03*f(t).

  MLE  argmax of the likelihood
  EBE  argmax of likelihood + log-normal prior (the NLME conditional mode)

Reported per observation design (3 points, as the encoder gets, vs all 12):
  R^2 of log Km_hat against log Km_true, relative SE from the profile curvature,
  and the shrinkage of the EBE toward the prior.
"""
import argparse, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch

sys.path.insert(0, '/Users/benjaminmaurel/Documents/PharmaNODE')
sys.path.insert(0, '/private/tmp/claude-501/-Users-benjaminmaurel-Documents-PharmaNODE/a847a5f5-d0e8-430d-965d-120f8364055e/scratchpad')
from gen_tacro_film import TacrolimusPK, RESIDUAL_ERROR_ADD_SD, RESIDUAL_ERROR_PROP_SD
from oracle_floor import BatchPK, NBR_SS, OBS_REL, ENC_T, time_grid

KM_TV, KM_OMEGA = 0.01, np.sqrt(0.10)
VMAX_OMEGA = np.sqrt(0.08)


def sim_grid(theta_fixed, grid_name, grid_vals, form, ht, dose, K):
    """One batched ODE solve over K parameter settings."""
    pk = BatchPK(K, formulation=form, hematocrit=ht,
                 cyp_status='expresser' if theta_fixed['_cyp'] else 'non_expresser', scenario=3)
    p = {k: torch.full((K,), float(v)) for k, v in theta_fixed.items() if not k.startswith('_')}
    for n, v in zip(grid_name, grid_vals):
        p[n] = torch.as_tensor(v, dtype=torch.float32)
    pk.individual_params = p
    pk.dose_mg = float(dose)
    if form == 'Advagraf':
        dosing = [i * 24 for i in range(NBR_SS + 1)]
    else:
        dosing = [24 * NBR_SS - 12 * (NBR_SS - i) for i in range(NBR_SS + 1)]
    ii = 12.0 if form == 'Prograf' else 24.0
    tp = time_grid(ii)
    with torch.no_grad():
        c = pk.simulate(dosing, tp, resample=False) * 1000.0
    return tp - 24 * NBR_SS, c


def loglik(conc_at_obs, y):
    """conc_at_obs [n_obs, K], y [n_obs] -> [K]. Generator's residual model."""
    pred = conc_at_obs
    yy = torch.as_tensor(y, dtype=torch.float32).unsqueeze(1)
    sd = RESIDUAL_ERROR_ADD_SD + RESIDUAL_ERROR_PROP_SD * pred
    return (-0.5 * ((yy - pred) / sd) ** 2 - torch.log(sd)).sum(dim=0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', default='confound_vc00_10k')
    ap.add_argument('--n-patients', type=int, default=200)
    ap.add_argument('--n-grid', type=int, default=121)
    ap.add_argument('--joint', action='store_true', help='also profile Vmax and Km jointly')
    ap.add_argument('--n-grid-2d', type=int, default=41)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    root = f'/Users/benjaminmaurel/Documents/PharmaNODE/results/exp_film_run/{a.cohort}'
    truth = pd.read_csv(f'{root}/confound_truth.csv'); truth = truth[truth.split == 'test']
    obs = pd.read_csv(f'{root}/virtual_cohort_film_test.csv')
    obs['DVn'] = pd.to_numeric(obs['DV'], errors='coerce')
    obs['Tn'] = pd.to_numeric(obs['TIME'], errors='coerce')
    tp_true = pd.read_csv('/tmp/true_params_10k.csv').set_index('ID')

    rng = np.random.RandomState(a.seed)
    ids = truth.ID.values.copy(); rng.shuffle(ids); ids = ids[:a.n_patients]

    lg = np.linspace(np.log(KM_TV) - 4 * KM_OMEGA, np.log(KM_TV) + 4 * KM_OMEGA, a.n_grid)
    km_grid = np.exp(lg)

    recs = []
    for n, pid in enumerate(ids, 1):
        tr = truth[truth.ID == pid].iloc[0]
        g = obs[(obs.ID == pid) & (obs.VISIT == 1) & obs.DVn.notna()].sort_values('Tn')
        t = g.Tn.values - g.Tn.values[0]; y = g.DVn.values
        P = tp_true.loc[pid]
        form = g.DRUG.iloc[0]; ht = float(g.HT.iloc[0]); cyp = int(g.CYP.iloc[0])
        # CL is read by forward() but unused under MM elimination; pass it anyway.
        fixed = {'Ktr': P.Ktr, 'CL': P.CL, 'Q': P.Q, 'Vc': P.Vc, 'Vp': P.Vp,
                 'Vmax': P.Vmax, '_cyp': cyp}

        trel, conc = sim_grid(fixed, ['Km'], [km_grid], form, ht, tr['d1'], a.n_grid)
        rec = {'ID': pid, 'Km_true': float(P.Km), 'Vmax_true': float(P.Vmax)}
        for tag, times in (('3obs', ENC_T), ('12obs', list(OBS_REL.numpy()))):
            idx = [int(torch.argmin(torch.abs(trel - float(tt)))) for tt in times]
            yy = [float(y[np.argmin(np.abs(t - float(tt)))]) for tt in times]
            ll = loglik(conc[idx, :], yy)
            lp = -0.5 * ((lg - np.log(KM_TV)) / KM_OMEGA) ** 2
            i_mle = int(np.argmax(ll)); i_ebe = int(np.argmax(ll + lp))
            # profile curvature -> SE on log Km
            d2 = np.gradient(np.gradient(ll, lg), lg)[i_mle]
            se = float(np.sqrt(1.0 / max(-d2, 1e-9)))
            rec[f'mle_{tag}'] = float(lg[i_mle]); rec[f'ebe_{tag}'] = float(lg[i_ebe])
            rec[f'se_{tag}'] = se
            rec[f'edge_{tag}'] = int(i_mle in (0, a.n_grid - 1))
        recs.append(rec)
        if n % 25 == 0: print(f'  {n}/{len(ids)}', flush=True)

    d = pd.DataFrame(recs)
    lk = np.log(d.Km_true.values)
    def r2(p): return 1 - np.mean((lk - p) ** 2) / np.var(lk)

    print(f'\ncohort {a.cohort} | n = {len(d)} | Km the ONLY unknown, all others at truth')
    print(f'prior sd on log Km = {KM_OMEGA:.3f}\n')
    print(f"{'design':10s} {'R2(MLE)':>9s} {'R2(EBE)':>9s} {'med SE(logKm)':>14s} {'SE/prior sd':>12s} {'MLE at edge':>12s}")
    print('-' * 72)
    for tag in ('3obs', '12obs'):
        se = np.median(d[f'se_{tag}'])
        print(f'{tag:10s} {r2(d[f"mle_{tag}"].values):9.3f} {r2(d[f"ebe_{tag}"].values):9.3f} '
              f'{se:14.3f} {se/KM_OMEGA:12.2f} {d[f"edge_{tag}"].mean():11.1%}')
    d.to_csv('/tmp/mle_km.csv', index=False)
    print('\nwrote /tmp/mle_km.csv')


if __name__ == '__main__':
    main()
