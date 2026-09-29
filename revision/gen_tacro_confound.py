###########################
# Exp 2: confounded dose assignment.
#
# Hypothesis under test: because dose-cond feeds d into the vector field at every
# step, it can use d as a SHORTCUT for elimination rate whenever the two are
# correlated in the training data -- learning "high dose => fast decay" instead of
# reading clearance out of the latent state. OT-FiLM's field is autonomous, so d
# cannot modulate downstream rates mid-course and the shortcut is unavailable.
#
# The confound is the clinically realistic one: clinicians titrate high-clearance
# patients UP to reach target exposure, so in observational data dose correlates
# with clearance.
#
#   dose index = Phi( rho * z_CL + sqrt(1-rho^2) * eps )  mapped onto the 7-level grid
#   with z_CL the patient's standardised log baseline clearance.
#
# rho = 0 reproduces the published independent assignment exactly.
#
# TWO RNG STREAMS: patient physiology is drawn from `seed`, dose assignment from
# `dose_seed`. Two cohorts generated with the same `seed` but different rho contain
# THE SAME PATIENTS and differ only in how doses were assigned -- which is what
# makes the confounded-vs-independent comparison an isolated one.
#
# The test split's visit-2 dose is drawn INDEPENDENTLY of clearance (a randomised
# intervention), so a model that leaned on the shortcut is penalised there.
###########################

import os, argparse, random
import numpy as np, pandas as pd, torch
from scipy.stats import norm
from sklearn.model_selection import train_test_split

from gen_tacro_film import TacrolimusPK, generate_patient_visit

GRID = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

# Which PK parameter the dose is confounded WITH. This choice decides whether the
# confound can bite at all:
#
#   CL   -- sets the LEVEL of the curve. Measured shape range of the dose-response
#           = 0.000: changing CL fourfold does not alter how exposure scales with
#           dose at all, and the level is directly visible in the observed window.
#           Confounding on it is doubly harmless (this was the first design).
#   Km   -- sets the CURVATURE of the dose-response (shape range 0.203, the largest
#           of any parameter). Not identifiable from a single-dose window -- you need
#           >= 2 doses to see it -- and it governs exactly the quantity the
#           counterfactual scores. Clinically: CYP3A5 genotype-guided starting doses
#           are standard practice and genotype also governs metabolic saturation.
#   Vmax -- intermediate (0.131).
#   Vp   -- shapes the 4-24 h tail, which the encoder never observes (its window
#           ends at 3 h). Strong mechanism, weak clinical story.
# Vc is the confounder that makes the experiment a genuine shortcut test: unlike Km
# it IS inferable from a single-dose curve (R^2 = 0.86 from the latent on a control
# cohort), so a legitimate route exists and a model that leans on the dose instead
# has taken a shortcut rather than used its only option.  It also acts on the
# Michaelis-Menten dynamics -- C = A/Vc sets how saturated elimination is -- so it
# changes the dose-exposure CURVATURE (15.9%) and is therefore needed for the
# counterfactual rather than derivable by proportional scaling.
CONFOUND_PARAMS = ('CL', 'Km', 'Vmax', 'Vp', 'Vc')


def assign_dose(z_cl, rho, rng, grid=GRID):
    """Map a standardised log-clearance to a dose level with correlation ~rho."""
    u = rho * z_cl + np.sqrt(max(0.0, 1.0 - rho ** 2)) * rng.normal()
    k = int(np.clip(np.floor(norm.cdf(u) * len(grid)), 0, len(grid) - 1))
    return grid[k]


def build(num_patients, rho, scenario, seed, dose_seed, test_fraction, randomize_test, ood_v2_grid=None,
          test_v2_rho=None,
          nbr_ss=6, confound_param='CL', grid=None, zstats=None, id_offset=0,
          eta_dist='log_normal', t_df=4, ktr_slow_frac=0.0, ktr_slow_factor=1.0, v1_cavg_window=None,
          ood_keep_v1=False, sat_scale=1.0, sat_logunif=None):
    obs = torch.tensor([0, 0.33, 0.67, 1., 1.5, 2., 3., 4., 6., 9., 12., 24.]) + 24 * nbr_ss
    # ---- pass 1: physiology only, so z_CL can be standardised over the cohort ----
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    slow_rng = np.random.RandomState(seed + 4242)
    # saturation reserve: Km AND Vmax scaled by the same per-patient factor k, so Vmax/Km (the low-concentration
    # clearance) is unchanged and only the degree of Michaelis-Menten saturation moves. Fixed k (sat_scale) or
    # log-uniform per patient (sat_logunif) drawn from its OWN stream, so every other stream is consumed as before.
    sat_on = (sat_scale != 1.0) or (sat_logunif is not None)
    assert not sat_on or scenario >= 3, "--sat-scale / --sat-logunif need Michaelis-Menten (scenario >= 3)"
    sat_rng = np.random.RandomState(seed + 7331)
    record_params = (eta_dist != 'log_normal') or ktr_slow_frac > 0 or sat_on
    pats = []
    for pid in range(1, num_patients + 1):
        form = random.choice(['Prograf', 'Advagraf'])
        cyp = random.choice(['expresser', 'non_expresser'])
        ht = random.uniform(25.0, 45.0) if scenario in (2, 4) else 35.0
        pk = TacrolimusPK(formulation=form, hematocrit=ht, distribution_type=eta_dist,
                          cyp_status=cyp, scenario=scenario)
        pk._sample_individual_parameters(t_df=t_df)
        slow = 0
        if ktr_slow_frac > 0:
            # unrecorded slow/delayed-absorber subgroup: Ktr scaled by ktr_slow_factor. The indicator comes
            # from its OWN stream, so the random/torch streams (covariates, etas) are consumed as before.
            slow = int(slow_rng.rand() < ktr_slow_frac)
            if slow:
                pk.individual_params['Ktr'] = pk.individual_params['Ktr'] * ktr_slow_factor
        sat_k = 1.0
        if sat_logunif is not None:
            sat_k = float(np.exp(sat_rng.uniform(np.log(sat_logunif[0]), np.log(sat_logunif[1]))))
        elif sat_scale != 1.0:
            sat_k = float(sat_scale)
        if sat_on:
            pk.individual_params['Km'] = pk.individual_params['Km'] * sat_k
            pk.individual_params['Vmax'] = pk.individual_params['Vmax'] * sat_k
        pats.append(dict(pid=pid, pk=pk, form=form, cyp=cyp, ht=ht, slow=slow, sat_k=sat_k,
                         cl=float(pk.individual_params['CL']),
                         par=float(pk.individual_params[confound_param])))
    grid = grid or GRID
    # standardise the log of whichever parameter the dose is confounded with
    lp = np.log(np.array([p['par'] for p in pats]))
    if zstats is not None:
        # Pin the standardisation to a reference cohort.  z_CL feeds assign_dose, so
        # re-deriving mean/sd on a different cohort would shift every patient's dose
        # and change the realised dose-physiology correlation -- the quantity the
        # whole experiment turns on.  Pinning keeps a freshly generated held-out set
        # on the same dose scale as the cohort the models were trained on.
        zm, zs = zstats
    else:
        zm, zs = lp.mean(), lp.std()
    z_cl = (lp - zm) / zs

    # ---- pass 2: dose assignment on its own stream ----
    rng = np.random.RandomState(dose_seed)
    rng_v2 = np.random.RandomState(dose_seed + 9173)   # only used when test_v2_rho is set
    n_rejected = [0, 0]   # [test, train] patients outside the visit-1 exposure window
    rng_ood = np.random.RandomState(dose_seed + 5151)   # only used with ood_keep_v1
    n_train = int(round(num_patients * (1 - test_fraction)))
    rows, truth = [], []
    for i, p in enumerate(pats):
        is_test = (p['pid'] > n_train)
        d1 = assign_dose(z_cl[i], rho, rng, grid)
        if is_test and ood_v2_grid:
            # OUT-OF-SUPPORT intervention: visit-2 dose drawn from a grid the model
            # never saw in training. visit 1 stays in-distribution so the encoder is
            # given a familiar visit -- this isolates dose extrapolation from an
            # inability to encode the patient at all.
            if ood_keep_v1:
                # consume the dose stream exactly as the in-range cohort does (randomised d2, redrawn while equal
                # to d1), so every later patient's d1 is identical across the in-range and out-of-range cohorts;
                # the out-of-range d2 comes from its own stream.
                _d = grid[rng.randint(len(grid))]
                while _d == d1:
                    _d = grid[rng.randint(len(grid))]
                d2 = ood_v2_grid[rng_ood.randint(len(ood_v2_grid))]
            else:
                d2 = ood_v2_grid[rng.randint(len(ood_v2_grid))]
        elif is_test and randomize_test:
            # randomised intervention: visit-2 dose independent of clearance
            d2 = grid[rng.randint(len(grid))]
            while d2 == d1:
                d2 = grid[rng.randint(len(grid))]
            if test_v2_rho is not None:
                # Re-draw the counterfactual dose with a prescribed correlation to the
                # confounded parameter, using a DEDICATED stream. The randomised draws
                # above are still made and discarded, so `rng` is consumed identically
                # to the default path: the cohort therefore contains the same patients,
                # with the same visit-1 doses, and differs only in the visit-2 rule.
                d2 = assign_dose(z_cl[i], test_v2_rho, rng_v2, grid)
                guard_v2 = 0
                while d2 == d1 and guard_v2 < 50:
                    d2 = assign_dose(z_cl[i], test_v2_rho, rng_v2, grid); guard_v2 += 1
                if d2 == d1:
                    d2 = grid[(grid.index(d1) + 1) % len(grid)]
        else:
            d2 = assign_dose(z_cl[i], rho, rng, grid)
            guard = 0
            while d2 == d1 and guard < 50:
                d2 = assign_dose(z_cl[i], rho, rng, grid); guard += 1
            if d2 == d1:
                d2 = grid[(grid.index(d1) + 1) % len(grid)]
        if v1_cavg_window is not None:
            # clinically plausible exposure on the OBSERVED visit: keep the patient only if the true average
            # concentration over the visit-1 dosing interval (AUC / tau) lies in the window. Doses were already
            # drawn above, so the dose streams are consumed exactly as without the window; the visit-2 dose (the
            # counterfactual) is NOT restricted. Rejected patients are not simulated past visit 1.
            p['pk'].dose_mg = d1
            r1 = generate_patient_visit(p['pk'], 1, p['pid'] + id_offset, nbr_ss, obs, p['form'], p['ht'], p['cyp'])
            tau = 12.0 if p['form'] == 'Prograf' else 24.0
            cavg = float(r1[0]['AUC']) / tau
            if not (v1_cavg_window[0] <= cavg <= v1_cavg_window[1]):
                n_rejected[0 if is_test else 1] += 1
                continue
            p['pk'].dose_mg = d2
            rows += r1 + generate_patient_visit(p['pk'], 2, p['pid'] + id_offset, nbr_ss, obs,
                                                p['form'], p['ht'], p['cyp'])
        else:
          for visit, dose in ((1, d1), (2, d2)):
            p['pk'].dose_mg = dose
            rows += generate_patient_visit(p['pk'], visit, p['pid'] + id_offset, nbr_ss, obs,
                                           p['form'], p['ht'], p['cyp'])
        truth.append(dict(ID=p['pid'] + id_offset, CL_base=p['cl'], confound_par=p['par'],
                          confound_param=confound_param, z_CL=z_cl[i], d1=d1, d2=d2,
                          DRUG=p['form'], CYP=p['cyp'], HT=p['ht'],
                          split='test' if is_test else 'train',
                          **({k: float(p['pk'].individual_params[k]) for k in ('Ktr', 'CL', 'Q', 'Vc', 'Vp', 'Vmax', 'Km')
                              if k in p['pk'].individual_params} | {'slow_absorber': p['slow']}
                             | ({'sat_k': p['sat_k']} if sat_on else {})
                             if record_params else {})))
        if p['pid'] % 50 == 0:
            print(f"  ...{p['pid']}/{num_patients}", flush=True)
    if v1_cavg_window is not None:
        print(f"visit-1 exposure window {v1_cavg_window} ng/mL: rejected {n_rejected[1]} train / {n_rejected[0]} test patients")
    return pd.DataFrame(rows), pd.DataFrame(truth), n_train


def main():
    ap = argparse.ArgumentParser('Confounded dose assignment (Exp 2)')
    ap.add_argument('--exp', required=True)
    ap.add_argument('--num_patients', type=int, default=1000)
    ap.add_argument('--rho', type=float, default=0.9,
                    help="correlation between standardised log-clearance and the dose "
                         "assignment score. 0 = published independent assignment.")
    ap.add_argument('--scenario', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0, help="physiology stream")
    ap.add_argument('--dose-seed', type=int, default=1234, help="dose-assignment stream")
    ap.add_argument('--test-fraction', type=float, default=0.2)
    ap.add_argument('--randomize-test', action='store_true', default=True)
    ap.add_argument('--no-randomize-test', dest='randomize_test', action='store_false')
    ap.add_argument('--confound-param', default='CL', choices=CONFOUND_PARAMS,
                    help="PK parameter the dose is confounded with. CL sets the curve's "
                         "level and has ZERO leverage on the dose-response; Km sets its "
                         "curvature and is the one that can actually mislead the model.")
    ap.add_argument('--ood-v2-grid', default=None,
                    help="Comma-separated doses for the TEST split's visit-2, drawn from "
                         "outside the training grid (e.g. '0.5,1.0' or '7.0,8.0'). Visit 1 "
                         "and the whole train split keep the normal grid. Default None "
                         "reproduces previous behaviour exactly.")
    ap.add_argument('--test-v2-rho', type=float, default=None,
                    help="Correlation between the confounded parameter and the TEST "
                         "split's visit-2 dose. Default None keeps the randomised "
                         "intervention. Setting it (e.g. -0.9) keeps the same patients "
                         "and visit-1 doses and changes only the counterfactual rule, "
                         "which turns the test split into a probe of whether a model "
                         "is exploiting the training-time dose-physiology association.")
    ap.add_argument('--dose-grid', default=None,
                    help="Comma-separated dose levels, e.g. '1,2,3,4,5,6,7,8'. "
                         "Default is the published 2.0-5.0 grid.")
    ap.add_argument('--zstats-from', default=None,
                    help="Experiment name whose confound_truth.csv supplies the mean and sd "
                         "of log(confound_param) used to standardise z_CL. Use when "
                         "generating a fresh held-out cohort that must sit on the same dose "
                         "scale as the cohort the models were trained on. Default None "
                         "re-derives them from the generated cohort, as before.")
    ap.add_argument('--id-offset', type=int, default=0,
                    help="Added to every patient ID, so a freshly generated cohort can be "
                         "evaluated alongside an existing one without ID collisions.")
    ap.add_argument('--out-root', default='./results/exp_film_run')
    ap.add_argument('--eta-dist', default='log_normal', choices=['log_normal', 'log_t'],
                    help="Distribution of the random effects. log_t = Student-t etas scaled to the same "
                         "variance (heavy tails). Default log_normal = every existing cohort.")
    ap.add_argument('--t-df', type=float, default=4, help="Degrees of freedom for --eta-dist log_t.")
    ap.add_argument('--ktr-slow-frac', type=float, default=0.0,
                    help="Fraction of patients in an unrecorded slow-absorber subgroup (default 0 = none).")
    ap.add_argument('--ktr-slow-factor', type=float, default=1.0,
                    help="Ktr multiplier for the slow-absorber subgroup.")
    ap.add_argument('--ood-keep-v1', action='store_true',
                    help="With --ood-v2-grid: keep every patient's visit-1 dose identical to the in-range cohort "
                         "(the default path does not: its dose stream diverges after the first redraw).")
    ap.add_argument('--v1-cavg-window', default=None,
                    help="'lo,hi' in ng/mL: keep only patients whose true average concentration over the visit-1 "
                         "dosing interval (AUC/tau) lies in [lo, hi]. Default: no window (every existing cohort).")
    ap.add_argument('--prop-sd', type=float, default=None,
                    help="Override gen_tacro_film.RESIDUAL_ERROR_PROP_SD for this run only "
                         "(default: leave the module value, currently 0.03). The paper's "
                         "committed simulator used 0.113.")
    ap.add_argument('--add-sd', type=float, default=None,
                    help="Override gen_tacro_film.RESIDUAL_ERROR_ADD_SD (ng/mL) for this run only "
                         "(default: module value, currently 0.03). The paper used 0.71.")
    ap.add_argument('--sat-scale', type=float, default=1.0,
                    help="Scale every patient's Km AND Vmax by this factor (scenario >= 3). Vmax/Km is unchanged, "
                         "so only the degree of saturation moves (k > 1 = less saturated). Default 1 = no change.")
    ap.add_argument('--sat-logunif', default=None,
                    help="'lo,hi': per-patient Km/Vmax scale drawn log-uniform in [lo, hi] from its own stream "
                         "(a population spread in saturation). Overrides --sat-scale. Default: none.")
    a = ap.parse_args()
    import gen_tacro_film as _gtf
    if a.prop_sd is not None: _gtf.RESIDUAL_ERROR_PROP_SD = a.prop_sd
    if a.add_sd is not None: _gtf.RESIDUAL_ERROR_ADD_SD = a.add_sd
    print(f"residual error: sd = {_gtf.RESIDUAL_ERROR_ADD_SD} + {_gtf.RESIDUAL_ERROR_PROP_SD} * C")

    out = os.path.join(a.out_root, str(a.exp)); os.makedirs(out, exist_ok=True)
    grid = [float(x) for x in a.dose_grid.split(',')] if a.dose_grid else None
    ood = [float(x) for x in a.ood_v2_grid.split(',')] if a.ood_v2_grid else None
    zstats = None
    if a.zstats_from:
        _t = pd.read_csv(os.path.join(a.out_root, a.zstats_from, 'confound_truth.csv'))
        _lp = np.log(_t['confound_par'].values)
        zstats = (_lp.mean(), _lp.std())
        print(f"z_CL standardisation pinned to {a.zstats_from}: "
              f"mean={zstats[0]:.6f} sd={zstats[1]:.6f}")
    df, truth, n_train = build(a.num_patients, a.rho, a.scenario, a.seed, a.dose_seed,
                               a.test_fraction, a.randomize_test, ood_v2_grid=ood,
                               zstats=zstats, id_offset=a.id_offset,
                               test_v2_rho=a.test_v2_rho,
                               confound_param=a.confound_param, grid=grid,
                               eta_dist=a.eta_dist, t_df=a.t_df,
                               ktr_slow_frac=a.ktr_slow_frac, ktr_slow_factor=a.ktr_slow_factor,
                               v1_cavg_window=([float(x) for x in a.v1_cavg_window.split(',')]
                                               if a.v1_cavg_window else None),
                               ood_keep_v1=a.ood_keep_v1, sat_scale=a.sat_scale,
                               sat_logunif=([float(x) for x in a.sat_logunif.split(',')]
                                            if a.sat_logunif else None))
    tr = truth[truth.split == 'train'].ID.values
    te = truth[truth.split == 'test'].ID.values
    df[df['ID'].isin(tr)].to_csv(os.path.join(out, 'virtual_cohort_film_train.csv'), index=False)
    df[df['ID'].isin(te)].to_csv(os.path.join(out, 'virtual_cohort_film_test.csv'), index=False)
    truth.to_csv(os.path.join(out, 'confound_truth.csv'), index=False)
    import json as _json
    with open(os.path.join(out, 'noise.json'), 'w') as _f:
        _json.dump({'add_sd': _gtf.RESIDUAL_ERROR_ADD_SD, 'prop_sd': _gtf.RESIDUAL_ERROR_PROP_SD,
                    'v1_cavg_window': a.v1_cavg_window,
                    **({'sat_scale': a.sat_scale, 'sat_logunif': a.sat_logunif}
                       if (a.sat_scale != 1.0 or a.sat_logunif) else {})}, _f)

    trn = truth[truth.split == 'train']; tst = truth[truth.split == 'test']
    print(f"\nWrote {out}\n  train {len(tr)} / test {len(te)} | rho = {a.rho} on {a.confound_param}")
    print(f"  dose grid: {grid or GRID}  ({(grid or GRID)[-1]/(grid or GRID)[0]:.1f}x span)")
    print(f"  realised corr(log {a.confound_param}, d1): train "
          f"{np.corrcoef(np.log(trn.confound_par), trn.d1)[0,1]:+.3f} | test "
          f"{np.corrcoef(np.log(tst.confound_par), tst.d1)[0,1]:+.3f}")
    print(f"  realised corr(log CL, d1): train {np.corrcoef(np.log(trn.CL_base), trn.d1)[0,1]:+.3f}"
          f" | test {np.corrcoef(np.log(tst.CL_base), tst.d1)[0,1]:+.3f}")
    print(f"  realised corr(log CL, d2): train {np.corrcoef(np.log(trn.CL_base), trn.d2)[0,1]:+.3f}"
          f" | test {np.corrcoef(np.log(tst.CL_base), tst.d2)[0,1]:+.3f}"
          f"   <- test d2 is the randomised intervention" if a.randomize_test else "")


if __name__ == '__main__':
    main()
