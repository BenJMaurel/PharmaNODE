###########################
# Longitudinal cohort with EVOLVING physiology (paper revision)
#
# Each patient is followed over several visits. Between visits two things change
# independently:
#   * physiology  -- hematocrit rises as post-transplant anaemia resolves, which
#                    lowers CL/F through the model's own HT exponent (-3.14).
#                    Trajectory: HT(t) = HT_inf - (HT_inf - HT_0) exp(-k t), a
#                    standard saturating recovery; each patient gets its own
#                    HT_0, HT_inf and rate. Time-varying CL driven by hematocrit
#                    is the standard structure in tacrolimus population PK
#                    (e.g. Staatz/Stoerset-type models).
#   * dose        -- reassigned at each visit, independently of physiology.
#
# Because the two vary independently, the latent response to each can be
# separated by regression -- which is what makes this a test of whether removing
# the dose axis yields a representation that tracks physiology only.
#
# PURELY ADDITIVE: imports the published simulator, changes nothing in it.
###########################

import os, argparse, random
import numpy as np
import pandas as pd
import torch

from gen_tacro_film import TacrolimusPK, generate_patient_visit, POPULATION_PARAMS


def set_hematocrit(pk, ht, etas):
    """Update HT and re-derive the HT-dependent typical values, keeping the
    patient's random effects (etas) fixed so only physiology moves."""
    pk.hematocrit = ht
    cyp = 1.0 if pk.cyp_status == 'expresser' else 0.0
    study = 1.0 if pk.formulation == 'Prograf' else 0.0
    tv = {
        'Ktr': POPULATION_PARAMS['theta1_Ktr'] * (POPULATION_PARAMS['theta2_Ktr_study'] ** study),
        'CL':  POPULATION_PARAMS['theta3_CL'] * ((ht / 35.0) ** POPULATION_PARAMS['theta4_CL_HT'])
               * (POPULATION_PARAMS['theta5_CL_CYP'] ** cyp),
        'Q':   POPULATION_PARAMS['Q'],
        'Vc':  POPULATION_PARAMS['theta6_Vc'] * (POPULATION_PARAMS['theta7_Vc_study'] ** study),
        'Vp':  POPULATION_PARAMS['Vp'],
    }
    if pk.scenario == 3:
        tv['Vp2'] = POPULATION_PARAMS['Vp2']; tv['Q2'] = POPULATION_PARAMS['Q2']
        tv['Vmax'] = POPULATION_PARAMS['theta_Vmax'] * ((ht / 35.0) ** POPULATION_PARAMS['theta4_CL_HT']) \
                     * (POPULATION_PARAMS['theta5_CL_CYP'] ** cyp)
        tv['Km'] = POPULATION_PARAMS['theta_Km']
    for k, v in tv.items():
        pk.individual_params[k] = torch.tensor(float(v)) * torch.exp(torch.tensor(float(etas[k])))


def main():
    p = argparse.ArgumentParser('Longitudinal tacrolimus cohort with evolving physiology')
    p.add_argument('--exp', type=str, required=True)
    p.add_argument('--num_patients', type=int, default=60)
    p.add_argument('--n-visits', type=int, default=6)
    p.add_argument('--scenario', type=int, default=3)
    p.add_argument('--doses', type=str, default='2.0,5.0',
                   help="Dose levels available at each visit (mg).")
    p.add_argument('--weeks', type=str, default='1,2,4,8,12,24',
                   help="Weeks post-transplant for each visit.")
    p.add_argument('--out-root', type=str, default='./results/exp_film_run')
    p.add_argument('--attribution', action='store_true',
                   help="Controlled mode for the attribution demo: each visit-to-visit "
                        "transition changes ONLY the dose, ONLY the physiology, BOTH, or "
                        "NEITHER, in random order per patient. The condition is recorded as "
                        "COND, so a decomposition of the latent step can be scored against "
                        "the true cause.")
    p.add_argument('--titrate', action='store_true',
                   help="Assign the dose by titration instead of at random: after the first "
                        "visit the dose is adjusted toward a target exposure using the "
                        "patient's own AUC, then snapped to the nearest available level. This "
                        "makes dose CORRELATED with clearance, so reading the dose becomes a "
                        "shortcut to the phenotype -- which a policy-shift test then exposes.")
    p.add_argument('--target-auc', type=float, default=150.0)
    p.add_argument('--phenotypes', action='store_true',
                   help="Split the cohort into two physiological phenotypes with different "
                        "recovery dynamics: 'fast' (rapid hematocrit recovery -> CL falls "
                        "quickly) and 'slow' (persistent anaemia -> CL stays high). The "
                        "phenotype label is written to the truth file as PHENO.")
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    doses = [float(x) for x in args.doses.split(',')]
    weeks = [float(x) for x in args.weeks.split(',')][:args.n_visits]
    nbr_ss = 6
    obs_t = torch.tensor([0, 0.33, 0.67, 1., 1.5, 2., 3., 4., 6., 9., 12., 24.]) + 24 * nbr_ss

    rows, meta = [], []
    print(f"{args.num_patients} patients x {len(weeks)} visits | scenario {args.scenario} | doses {doses}")
    for pid in range(1, args.num_patients + 1):
        formulation = random.choice(['Prograf', 'Advagraf'])
        cyp = random.choice(['expresser', 'non_expresser'])
        # patient-specific hematocrit recovery curve
        ht0 = random.uniform(25.0, 32.0)          # post-transplant anaemia
        if args.phenotypes:
            pheno = pid % 2                        # 0 = slow, 1 = fast; balanced by construction
            if pheno == 1:                         # fast responder
                htinf = random.uniform(42.0, 46.0); krate = random.uniform(0.30, 0.45)
            else:                                  # slow responder, persistent anaemia
                htinf = random.uniform(33.0, 37.0); krate = random.uniform(0.03, 0.10)
        else:
            pheno = -1
            htinf = random.uniform(38.0, 45.0)    # recovered
            krate = random.uniform(0.08, 0.30)    # per week

        pk = TacrolimusPK(formulation=formulation, hematocrit=ht0,
                          distribution_type='log_normal', cyp_status=cyp, scenario=args.scenario)
        pk._sample_individual_parameters()
        # recover the patient's random effects so physiology can be moved without resampling them
        cypf = 1.0 if cyp == 'expresser' else 0.0
        studyf = 1.0 if formulation == 'Prograf' else 0.0
        tv0 = {'Ktr': POPULATION_PARAMS['theta1_Ktr']*(POPULATION_PARAMS['theta2_Ktr_study']**studyf),
               'CL': POPULATION_PARAMS['theta3_CL']*((ht0/35.0)**POPULATION_PARAMS['theta4_CL_HT'])*(POPULATION_PARAMS['theta5_CL_CYP']**cypf),
               'Q': POPULATION_PARAMS['Q'],
               'Vc': POPULATION_PARAMS['theta6_Vc']*(POPULATION_PARAMS['theta7_Vc_study']**studyf),
               'Vp': POPULATION_PARAMS['Vp']}
        if args.scenario == 3:
            tv0['Vp2']=POPULATION_PARAMS['Vp2']; tv0['Q2']=POPULATION_PARAMS['Q2']
            tv0['Vmax']=POPULATION_PARAMS['theta_Vmax']*((ht0/35.0)**POPULATION_PARAMS['theta4_CL_HT'])*(POPULATION_PARAMS['theta5_CL_CYP']**cypf)
            tv0['Km']=POPULATION_PARAMS['theta_Km']
        etas = {k: float(np.log(float(pk.individual_params[k]) / v)) for k, v in tv0.items()}

        if args.attribution:
            conds = ['dose', 'phys', 'both', 'none']; random.shuffle(conds)
            cur_ht = ht0; cur_dose = random.choice(doses)
            for v, cond in enumerate([None] + conds, start=1):
                if cond in ('dose', 'both'):
                    cur_dose = random.choice([d for d in doses if d != cur_dose])
                if cond in ('phys', 'both'):
                    cur_ht = min(cur_ht + random.uniform(3.0, 6.0), 46.0)
                set_hematocrit(pk, cur_ht, etas); pk.dose_mg = cur_dose
                rows += generate_patient_visit(pk, v, pid, nbr_ss, obs_t, formulation, cur_ht, cyp)
                meta.append({'ID': pid, 'VISIT': v, 'WEEK': float(v), 'HT': cur_ht, 'PHENO': -1,
                             'COND': cond if cond else 'baseline',
                             'CL': float(pk.individual_params['CL']), 'DOSE': cur_dose,
                             'DRUG': formulation, 'CYP': 1 if cyp == 'expresser' else 0})
            if pid % 20 == 0: print(f"  ...{pid}/{args.num_patients}")
            continue

        for v, wk in enumerate(weeks, start=1):
            ht = htinf - (htinf - ht0) * np.exp(-krate * wk)
            set_hematocrit(pk, ht, etas)
            if args.titrate and v > 1:
                # adjust toward the target exposure using the previous visit's AUC,
                # then snap to the nearest available level
                want = prev_dose * (args.target_auc / max(prev_auc, 1e-6))
                pk.dose_mg = min(doses, key=lambda d: abs(d - want))
            else:
                pk.dose_mg = random.choice(doses)
            rec = generate_patient_visit(pk, v, pid, nbr_ss, obs_t, formulation, ht, cyp)
            prev_dose = pk.dose_mg; prev_auc = float(rec[0]['AUC'])
            rows += rec
            meta.append({'ID': pid, 'VISIT': v, 'WEEK': wk, 'HT': ht, 'PHENO': pheno, 'COND': 'traj',
                         'CL': float(pk.individual_params['CL']),
                         'DOSE': pk.dose_mg, 'DRUG': formulation,
                         'CYP': 1 if cyp == 'expresser' else 0})
        if pid % 20 == 0: print(f"  ...{pid}/{args.num_patients}")

    out = os.path.join(args.out_root, str(args.exp)); os.makedirs(out, exist_ok=True)
    df = pd.DataFrame(rows); md = pd.DataFrame(meta)
    df.to_csv(os.path.join(out, 'longitudinal_cohort.csv'), index=False)
    md.to_csv(os.path.join(out, 'longitudinal_truth.csv'), index=False)
    print(f"\nwrote {out}/longitudinal_cohort.csv  ({len(df)} rows)")
    print(f"      {out}/longitudinal_truth.csv    ({len(md)} visit records)")
    print(f"  HT  range across visits: {md.HT.min():.1f} - {md.HT.max():.1f}")
    print(f"  CL  range across visits: {md.CL.min():.1f} - {md.CL.max():.1f}  "
          f"(within-patient CL change: {md.groupby('ID').CL.apply(lambda s: s.max()/s.min()).mean():.2f}x on average)")
    if args.phenotypes:
        g = md.groupby('PHENO')
        print("  phenotypes: " + " | ".join(
            f"{'fast' if k==1 else 'slow'} n={v.ID.nunique()} HT@wk24={v[v.WEEK==v.WEEK.max()].HT.mean():.1f}"
            f" CL@wk24={v[v.WEEK==v.WEEK.max()].CL.mean():.1f}" for k,v in g))
    if args.titrate:
        cc=md.groupby('ID').apply(lambda g: np.corrcoef(g.CL,g.DOSE)[0,1] if g.DOSE.std()>0 else np.nan)
        print(f"  titration: within-patient corr(CL, dose) = {np.nanmean(cc):+.2f}  "
              f"| corr(CL, dose) overall = {np.corrcoef(md.CL, md.DOSE)[0,1]:+.2f}")
    print(f"  dose changes between consecutive visits: "
          f"{md.sort_values(['ID','VISIT']).groupby('ID').DOSE.apply(lambda s: (s.diff()!=0).sum()-1).mean():.1f} per patient")


if __name__ == '__main__':
    main()
