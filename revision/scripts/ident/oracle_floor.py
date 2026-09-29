"""Information floor on the visit-2 counterfactual, given what the encoder actually sees.

The encoder gets 3 noisy concentrations (t = 0, 1, 3 h) plus dose / formulation / CYP.
The individual has 7 free PK parameters. This computes the Bayes-optimal prediction of
the visit-2 AUC under the TRUE generative model (exact log-normal prior, exact residual
error), by self-normalised importance sampling with the prior as proposal.

No estimator using the same inputs can beat this in expectation, so its error is a floor.

Reported:
  oracle      posterior mean of AUC given the 3 observations
  prior-only  same, ignoring the observations  (population model, no individualisation)
The gap between them is what those 3 points actually buy.
"""
import argparse, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch

sys.path.insert(0, '/Users/benjaminmaurel/Documents/PharmaNODE')
from gen_tacro_film import (TacrolimusPK, POPULATION_PARAMS, IPV_OMEGA,
                            RESIDUAL_ERROR_ADD_SD, RESIDUAL_ERROR_PROP_SD)

NBR_SS = 6
OBS_REL = torch.tensor([0., 0.33, 0.67, 1., 1.5, 2., 3., 4., 6., 9., 12., 24.])
ENC_T = [0.0, 1.0, 3.0]          # what the encoder is given
FIT_PARAMS = ['Ktr', 'CL', 'Q', 'Vc', 'Vp', 'Vmax', 'Km']


class BatchPK(TacrolimusPK):
    """TacrolimusPK with [K]-shaped individual parameters: one ODE solve for K draws."""
    def __init__(self, K, **kw):
        super().__init__(**kw)
        self.K = K

    def get_initial_state(self):
        t0 = torch.tensor([0.0])
        return t0, tuple(torch.zeros(self.K) for _ in range(6))

    def simulate(self, dosing_times, time_points, resample=False):
        # Identical to the generator's, except the t=0 placeholder is [1, K] not [[0.0]],
        # so the K draws concatenate. Value is irrelevant: t=0 is outside the AUC window
        # and is not one of the encoder's observation times.
        t0, state = self.get_initial_state()
        if 0.0 in dosing_times:
            state = self.state_update(state)
        all_conc = [torch.zeros(1, self.K)]
        dosing_times = sorted(dosing_times)
        last_time = t0
        for event_t in dosing_times:
            if event_t > last_time:
                ts = time_points[(time_points > last_time) & (time_points <= event_t)]
                if len(ts) > 0:
                    tt = torch.cat([last_time, ts])
                    sol = self.odeint(self, state, tt, atol=1e-6, rtol=1e-6)
                    all_conc.append(sol[4][1:] / self.individual_params['Vc'])
                    state = tuple(s[-1] for s in sol)
            if event_t > 0.0:
                state = self.state_update(state)
            last_time = torch.tensor([event_t])
        ts = time_points[time_points > last_time]
        if len(ts) > 0:
            tt = torch.cat([last_time, ts])
            sol = self.odeint(self, state, tt, atol=1e-6, rtol=1e-6)
            all_conc.append(sol[4][1:] / self.individual_params['Vc'])
        return torch.cat(all_conc)


def typical_values(formulation, cyp_status, ht):
    p = POPULATION_PARAMS
    study = 1.0 if formulation == 'Prograf' else 0.0
    cypf = 1.0 if cyp_status == 'expresser' else 0.0
    ht_eff = (ht / 35.0) ** p['theta4_CL_HT']
    return {
        'Ktr': p['theta1_Ktr'] * (p['theta2_Ktr_study'] ** study),
        'CL':  p['theta3_CL'] * ht_eff * (p['theta5_CL_CYP'] ** cypf),
        'Q':   p['Q'],
        'Vc':  p['theta6_Vc'] * (p['theta7_Vc_study'] ** study),
        'Vp':  p['Vp'],
        'Vmax': p['theta_Vmax'] * ht_eff * (p['theta5_CL_CYP'] ** cypf),
        'Km':  p['theta_Km'],
    }


def draw_prior(tv, K, rng):
    """Exact generative prior: log-normal with the population IPV omegas."""
    return {n: torch.tensor(tv[n] * np.exp(rng.normal(size=K) * IPV_OMEGA[n]), dtype=torch.float32)
            for n in FIT_PARAMS}


def time_grid(ii):
    """Coarse before the scored interval, fine inside it (AUC is a trapezoid there)."""
    sim_t = OBS_REL + 24 * NBR_SS
    coarse = torch.arange(0.0, 24 * NBR_SS, 0.5)
    fine = torch.arange(24 * NBR_SS, 24 * NBR_SS + 24.0 + 1e-9, 0.01)
    return torch.unique(torch.cat([coarse, fine, sim_t]))


def simulate_batch(K, formulation, cyp, ht, dose, scenario=3):
    """Concentrations (ng/mL) on the grid for all K draws. Returns (times_rel, conc [T,K])."""
    pk = BatchPK(K, formulation=formulation, hematocrit=ht, cyp_status=cyp, scenario=scenario)
    pk.dose_mg = float(dose)
    ii = 12.0 if formulation == 'Prograf' else 24.0
    if formulation == 'Advagraf':
        dosing = [i * 24 for i in range(NBR_SS + 1)]
    else:
        dosing = [24 * NBR_SS - 12 * (NBR_SS - i) for i in range(NBR_SS + 1)]
    tp = time_grid(ii)
    conc = pk.simulate(dosing, tp, resample=False) * 1000.0
    return tp - 24 * NBR_SS, conc.squeeze(1) if conc.dim() == 3 else conc, ii


def auc_from(times_rel, conc, ii):
    """Same window as the generator: 0-12 h for Prograf, 0-24 h for Advagraf."""
    hi = 12.0 if ii == 12.0 else 24.0
    m = (times_rel >= 0) & (times_rel <= hi)
    t = times_rel[m].numpy()
    c = conc[m].numpy()                      # [T, K]
    return np.trapezoid(c, t, axis=0)        # [K]


def run_patient(row, obs_vals, K, rng):
    form = row['DRUG']; cyp = 'expresser' if row['CYP'] == 1 else 'non_expresser'
    ht = float(row['HT'])
    tv = typical_values(form, cyp, ht)
    theta = draw_prior(tv, K, rng)

    out = {}
    for visit, dose in (('v1', row['d1']), ('v2', row['d2'])):
        pk = BatchPK(K, formulation=form, hematocrit=ht, cyp_status=cyp, scenario=3)
        pk.individual_params = dict(theta)
        pk.dose_mg = float(dose)
        if form == 'Advagraf':
            dosing = [i * 24 for i in range(NBR_SS + 1)]
        else:
            dosing = [24 * NBR_SS - 12 * (NBR_SS - i) for i in range(NBR_SS + 1)]
        ii = 12.0 if form == 'Prograf' else 24.0
        tp = time_grid(ii)
        with torch.no_grad():
            conc = pk.simulate(dosing, tp, resample=False) * 1000.0
        if conc.dim() == 3:
            conc = conc.squeeze(1)
        trel = tp - 24 * NBR_SS
        out[visit] = (trel, conc, ii)

    # likelihood of the 3 encoder observations under the visit-1 draws
    trel, conc1, ii1 = out['v1']
    idx = [int(torch.argmin(torch.abs(trel - t))) for t in ENC_T]
    pred = conc1[idx, :]                                   # [3, K]
    y = torch.tensor(obs_vals, dtype=torch.float32).unsqueeze(1)   # [3, 1]
    sd = RESIDUAL_ERROR_ADD_SD + RESIDUAL_ERROR_PROP_SD * pred
    ll = (-0.5 * ((y - pred) / sd) ** 2 - torch.log(sd)).sum(dim=0)  # [K]
    w = torch.softmax(ll, dim=0).numpy()

    auc1 = auc_from(*out['v1'])
    auc2 = auc_from(*out['v2'])
    ess = 1.0 / np.sum(w ** 2)
    return dict(auc1_post=float((w * auc1).sum()), auc1_prior=float(auc1.mean()),
                auc2_post=float((w * auc2).sum()), auc2_prior=float(auc2.mean()),
                ess=float(ess))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', default='confound_vc00_10k')
    ap.add_argument('--n-patients', type=int, default=150)
    ap.add_argument('--K', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='results/vc10k/oracle_floor.csv')
    a = ap.parse_args()

    root = f'/Users/benjaminmaurel/Documents/PharmaNODE/results/exp_film_run/{a.cohort}'
    truth = pd.read_csv(f'{root}/confound_truth.csv')
    df = pd.read_csv(f'{root}/virtual_cohort_film_test.csv')
    truth = truth[truth.split == 'test']

    rng = np.random.RandomState(a.seed)
    ids = truth.ID.values.copy(); rng.shuffle(ids)
    ids = ids[:a.n_patients]

    recs = []
    for n, pid in enumerate(ids, 1):
        tr = truth[truth.ID == pid].iloc[0]
        pdf = df[df.ID == pid]
        v1 = pdf[(pdf.VISIT == 1) & (pdf.DV != '.')].copy()
        v1['TIME'] = pd.to_numeric(v1['TIME']); v1['DV'] = pd.to_numeric(v1['DV'])
        v1 = v1.sort_values('TIME')
        t0 = v1['TIME'].iloc[0]
        obs = [float(v1.iloc[(v1['TIME'] - t0 - t).abs().argmin()]['DV']) for t in ENC_T]
        true1 = float(pdf[pdf.VISIT == 1]['AUC'].iloc[0])
        true2 = float(pdf[pdf.VISIT == 2]['AUC'].iloc[0])

        row = dict(DRUG=v1['DRUG'].iloc[0], CYP=int(v1['CYP'].iloc[0]), HT=float(v1['HT'].iloc[0]),
                   d1=float(tr['d1']), d2=float(tr['d2']))
        r = run_patient(row, obs, a.K, rng)
        r.update(ID=pid, true1=true1, true2=true2, d1=row['d1'], d2=row['d2'], drug=row['DRUG'])
        recs.append(r)
        if n % 10 == 0:
            print(f'  {n}/{len(ids)}', flush=True)

    out = pd.DataFrame(recs)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out.to_csv(a.out, index=False)

    def rmspe(t, p): return float(np.sqrt(np.mean(((t - p) / t) ** 2)) * 100)
    def mpe(t, p):   return float(np.mean((t - p) / t) * 100)

    print(f'\ncohort {a.cohort}  |  n = {len(out)}  |  K = {a.K}  |  median ESS = {out.ess.median():.1f}')
    print(f"{'':28s} {'AUC RMSPE %':>12s} {'MPE %':>9s}")
    for lab, tcol, pcol in (('V1 oracle (3 obs)', 'true1', 'auc1_post'),
                            ('V1 prior only', 'true1', 'auc1_prior'),
                            ('V2 oracle (3 obs)', 'true2', 'auc2_post'),
                            ('V2 prior only', 'true2', 'auc2_prior')):
        print(f'{lab:28s} {rmspe(out[tcol].values, out[pcol].values):12.2f} '
              f'{mpe(out[tcol].values, out[pcol].values):9.2f}')


if __name__ == '__main__':
    main()
