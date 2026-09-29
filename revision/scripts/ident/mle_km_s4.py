"""Did the hematocrit-on-Km design actually improve Km identifiability?

Replays the true parameters for a scenario-4 cohort, then profiles the joint (Vmax, Km)
likelihood exactly as for scenario 3, with the Km prior now centred on the covariate value
tv_km(Ht) = theta_Km * (Ht/35)^theta_Km_HT.

Two R^2 are reported and they answer different questions:
  eta_Km   how well the UNEXPLAINED part is recovered -- comparable to the scenario-3 number
  log Km   how well the parameter is known overall, covariate included -- what the model sees
"""
import argparse, os, random, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch

sys.path.insert(0, '/Users/benjaminmaurel/Documents/PharmaNODE')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import os
from gen_tacro_film import (TacrolimusPK, POPULATION_PARAMS, IPV_OMEGA,
                            RESIDUAL_ERROR_ADD_SD as EA, RESIDUAL_ERROR_PROP_SD as EB)
from oracle_floor import BatchPK, NBR_SS, OBS_REL, ENC_T, time_grid

VB = POPULATION_PARAMS['theta_Vmax']
V_OM = IPV_OMEGA['Vmax']
KM_TV = POPULATION_PARAMS['theta_Km']
TH_HT = POPULATION_PARAMS['theta_Km_HT']
RHO = POPULATION_PARAMS['rho_Km_Vc']
USE_BLOCK = os.environ.get('NO_BLOCK','0') != '1'


def replay(n, scenario, seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    rows = []
    for pid in range(1, n + 1):
        form = random.choice(['Prograf', 'Advagraf'])
        cyp = random.choice(['expresser', 'non_expresser'])
        ht = random.uniform(25.0, 45.0) if scenario in (2, 4) else 35.0
        pk = TacrolimusPK(formulation=form, hematocrit=ht, distribution_type='log_normal',
                          cyp_status=cyp, scenario=scenario)
        pk._sample_individual_parameters()
        r = {k: float(v) for k, v in pk.individual_params.items()}
        r.update(ID=pid, HT=ht, form=form, cyp=1 if cyp == 'expresser' else 0)
        rows.append(r)
    return pd.DataFrame(rows).set_index('ID')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', default='pilot_s4_vc00')
    ap.add_argument('--scenario', type=int, default=4)
    ap.add_argument('--n-patients', type=int, default=60)
    ap.add_argument('--n', type=int, default=41)
    a = ap.parse_args()

    root = f'/Users/benjaminmaurel/Documents/PharmaNODE/results/exp_film_run/{a.cohort}'
    truth = pd.read_csv(f'{root}/confound_truth.csv')
    ob = pd.read_csv(f'{root}/virtual_cohort_film_test.csv')
    ob['DVn'] = pd.to_numeric(ob['DV'], errors='coerce')
    ob['Tn'] = pd.to_numeric(ob['TIME'], errors='coerce')

    TP = replay(len(truth), a.scenario)
    m = TP.join(truth.set_index('ID')[['CL_base', 'confound_par', 'HT']], rsuffix='_rec')
    ok_cl = np.allclose(m.CL, m.CL_base); ok_vc = np.allclose(m.Vc, m.confound_par)
    ok_ht = np.allclose(m.HT, m.HT_rec)
    print(f'replay validation: CL {ok_cl}, Vc {ok_vc}, Ht {ok_ht}')
    if not (ok_cl and ok_vc and ok_ht):
        print('   replay does not match the cohort -- aborting'); return
    print(f'Ht range {m.HT.min():.1f}-{m.HT.max():.1f} | '
          f'corr(log Km, log Ht) = {np.corrcoef(np.log(m.Km), np.log(m.HT))[0,1]:+.3f}')

    te = truth[truth.split == 'test']
    ids = te.ID.values[:a.n_patients]
    km_om_res = float(np.sqrt(max(IPV_OMEGA['Km'] ** 2 - (0.1683 * TH_HT) ** 2, 1e-6)))
    gk = np.linspace(-4 * km_om_res, 4 * km_om_res, a.n)
    gv = np.linspace(-4 * V_OM, 4 * V_OM, a.n)
    KK, VV = np.meshgrid(gk, gv, indexing='ij')
    fk, fv = KK.ravel(), VV.ravel(); K = fk.size

    rec = []
    for i, pid in enumerate(ids, 1):
        t0 = te[te.ID == pid].iloc[0]
        g = ob[(ob.ID == pid) & (ob.VISIT == 1) & ob.DVn.notna()].sort_values('Tn')
        if g.empty: continue
        t = g.Tn.values - g.Tn.values[0]; y = g.DVn.values
        P = TP.loc[pid]; form = P.form; ht = float(P.HT); cyp = int(P.cyp)
        tvv = VB * ((ht / 35.0) ** POPULATION_PARAMS['theta4_CL_HT']) * (2.0 ** cyp)
        tvk = KM_TV * ((ht / 35.0) ** TH_HT)
        tv_vc = POPULATION_PARAMS['theta6_Vc'] * (POPULATION_PARAMS['theta7_Vc_study'] ** (1.0 if form=='Prograf' else 0.0))
        eta_vc = float(np.log(P.Vc / tv_vc))

        pk = BatchPK(K, formulation=form, hematocrit=ht,
                     cyp_status='expresser' if cyp else 'non_expresser', scenario=a.scenario)
        pk.individual_params = {
            'Ktr': torch.full((K,), float(P.Ktr)), 'CL': torch.full((K,), float(P.CL)),
            'Q': torch.full((K,), float(P.Q)), 'Vc': torch.full((K,), float(P.Vc)),
            'Vp': torch.full((K,), float(P.Vp)),
            'Vmax': torch.tensor(tvv * np.exp(fv), dtype=torch.float32),
            'Km': torch.tensor(tvk * np.exp(fk), dtype=torch.float32)}
        pk.dose_mg = float(t0['d1'])
        dos = ([j * 24 for j in range(NBR_SS + 1)] if form == 'Advagraf'
               else [24 * NBR_SS - 12 * (NBR_SS - j) for j in range(NBR_SS + 1)])
        tg = time_grid(12.0 if form == 'Prograf' else 24.0)
        with torch.no_grad():
            c = pk.simulate(dos, tg, resample=False) * 1000.0
        trel = tg - 24 * NBR_SS
        r = {'ID': pid, 'eta_true': float(np.log(P.Km / tvk)),
             'lkm_true': float(np.log(P.Km)), 'lkm_cov': float(np.log(tvk))}
        for tag, times in (('3obs', ENC_T), ('12obs', list(OBS_REL.numpy()))):
            idx = [int(torch.argmin(torch.abs(trel - float(x)))) for x in times]
            yy = torch.tensor([float(y[np.argmin(np.abs(t - float(x)))]) for x in times]).unsqueeze(1)
            pr = c[idx, :]; sd = EA + EB * pr
            ll = (-0.5 * ((yy - pr) / sd) ** 2 - torch.log(sd)).sum(0).numpy()
            # eta_Km | eta_Vc ~ N(rho*(om_km/om_vc)*eta_Vc, om_km^2 (1-rho^2)).
            # Vc is at truth here, so eta_Vc is known exactly.
            if USE_BLOCK:
                mu_k = RHO * (km_om_res / IPV_OMEGA['Vc']) * eta_vc
                sd_k = km_om_res * np.sqrt(1 - RHO ** 2)
            else:
                mu_k, sd_k = 0.0, km_om_res
            lp = -0.5 * ((fk - mu_k) / sd_k) ** 2 - 0.5 * (fv / V_OM) ** 2
            r[f'ebe_{tag}'] = float(fk[int(np.argmax(ll + lp))])
        rec.append(r)
        if i % 20 == 0: print(f'  {i}/{len(ids)}', flush=True)

    d = pd.DataFrame(rec)
    print(f'\nn = {len(d)} | scenario {a.scenario} | residual omega_Km = {km_om_res:.3f} '
          f'(scenario-3 value {IPV_OMEGA["Km"]:.3f})')
    print(f"{'design':8s} {'R2 eta_Km':>11s} {'R2 log Km (total)':>19s}")
    print('-' * 42)
    for tag in ('3obs', '12obs'):
        e = d[f'ebe_{tag}'].values
        r2e = 1 - np.mean((d.eta_true.values - e) ** 2) / np.var(d.eta_true.values)
        tot = d.lkm_cov.values + e
        r2t = 1 - np.mean((d.lkm_true.values - tot) ** 2) / np.var(d.lkm_true.values)
        print(f'{tag:8s} {r2e:11.3f} {r2t:19.3f}')
    print('\nscenario-3 reference (same procedure): R2 eta_Km = R2 log Km = 0.242 (3obs), 0.290 (12obs)')


if __name__ == '__main__':
    main()
