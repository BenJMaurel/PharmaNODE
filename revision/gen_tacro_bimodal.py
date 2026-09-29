###########################
# Bimodal-dose cohort generator (paper revision)
#
# Motivation: the standard generator draws doses from 7 levels 0.5 mg apart, so
# "interpolating between doses" is trivial for any smooth function of dose --
# every model passed that test. Real PK trials usually run a small number of
# well-separated fixed doses. This generator reproduces that: every patient is
# observed at exactly TWO widely separated dose levels, and nothing in between.
#
# It also writes simulated ground truth for a sweep of doses across and beyond
# the training range, for the held-out patients only. That truth cannot be
# derived analytically under scenario 3, where Michaelis-Menten elimination
# makes AUC non-proportional to dose -- which is the point of using it.
#
# PURELY ADDITIVE: imports the published simulator from gen_tacro_film.py and
# changes nothing in it.
###########################

import os
import argparse
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from gen_tacro_film import TacrolimusPK, generate_patient_visit


def build_cohort(num_patients, scenario, dose_low, dose_high, sweep_doses,
                 test_fraction=0.2, nbr_ss=6, dose_grid=None):
    observation_times = torch.tensor(
        [0, 0.33, 0.67, 1., 1.5, 2., 3., 4., 6., 9., 12., 24.]) + 24 * nbr_ss

    # train_test_split(..., shuffle=False) keeps order, so the held-out patients
    # are exactly the last test_fraction of the IDs. Knowing that up front means
    # the (expensive) dose sweep is only simulated for patients we will score.
    n_train = int(round(num_patients * (1 - test_fraction)))
    rows, truth = [], []

    print(f"Generating {num_patients} patients | scenario {scenario} | "
          f"doses {{{dose_low}, {dose_high}}} mg | sweep truth for the last "
          f"{num_patients - n_train} (held-out) patients")

    for pid in range(1, num_patients + 1):
        formulation = random.choice(['Prograf', 'Advagraf'])
        cyp_status = random.choice(['expresser', 'non_expresser'])
        # match gen_tacro_film.py exactly: hematocrit varies ONLY in scenario 2
        # (the covariate-misspecification case). Scenario 3 isolates nonlinear
        # elimination and keeps hematocrit fixed.
        hematocrit = random.uniform(25.0, 45.0) if scenario == 2 else 35.0

        pk = TacrolimusPK(formulation=formulation, hematocrit=hematocrit,
                          distribution_type='log_normal', cyp_status=cyp_status,
                          scenario=scenario)
        pk._sample_individual_parameters()      # once per patient, reused for every dose

        # the two visits sit at the two dose levels; order randomised per patient
        if dose_grid:
            # dense design: both visits drawn from a grid, never the same level.
            # This is the design the published generator uses, and it is what
            # makes the dose axis identifiable (dose R^2 in z0 drops to ~0.03).
            d1 = random.choice(dose_grid)
            d2 = random.choice([x for x in dose_grid if x != d1])
        else:
            d1, d2 = (dose_low, dose_high) if random.random() < 0.5 else (dose_high, dose_low)
        for visit, dose in ((1, d1), (2, d2)):
            pk.dose_mg = dose
            rows += generate_patient_visit(pk, visit, pid, nbr_ss, observation_times,
                                           formulation, hematocrit, cyp_status)

        if pid > n_train:
            for d in sweep_doses:
                pk.dose_mg = d
                r = generate_patient_visit(pk, 99, pid, nbr_ss, observation_times,
                                           formulation, hematocrit, cyp_status)
                truth.append({'ID': pid, 'DOSE': float(d), 'AUC': float(r[0]['AUC']),
                              'DRUG': formulation})
        if pid % 25 == 0:
            print(f"  ...{pid}/{num_patients}")

    return pd.DataFrame(rows), pd.DataFrame(truth)


def main():
    p = argparse.ArgumentParser('Bimodal-dose cohort generator')
    p.add_argument('--exp', type=str, required=True)
    p.add_argument('--num_patients', type=int, default=200)
    p.add_argument('--scenario', type=int, default=3)
    p.add_argument('--dose-low', type=float, default=2.0)
    p.add_argument('--dose-high', type=float, default=5.0)
    p.add_argument('--sweep', type=str,
                   default="1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0",
                   help="Doses (mg) at which to simulate ground truth for held-out patients.")
    p.add_argument('--dose-grid', type=str, default=None,
                   help="Comma-separated dense dose grid. When given, each patient's two "
                        "visits are drawn from this grid instead of {dose-low, dose-high}.")
    p.add_argument('--out-root', type=str, default='./results/exp_film_run')
    p.add_argument('--seed', type=int, default=None,
                   help="Seed the generator. The published generator is unseeded; "
                        "pass this for a reproducible cohort.")
    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    sweep = [float(x) for x in args.sweep.split(',')]
    out_dir = os.path.join(args.out_root, str(args.exp))
    os.makedirs(out_dir, exist_ok=True)

    grid = [float(x) for x in args.dose_grid.split(',')] if args.dose_grid else None
    df, truth = build_cohort(args.num_patients, args.scenario,
                             args.dose_low, args.dose_high, sweep, dose_grid=grid)

    ids = df['ID'].unique()
    train_ids, test_ids = train_test_split(ids, test_size=0.2, shuffle=False)
    df[df['ID'].isin(train_ids)].to_csv(
        os.path.join(out_dir, 'virtual_cohort_film_train.csv'), index=False)
    df[df['ID'].isin(test_ids)].to_csv(
        os.path.join(out_dir, 'virtual_cohort_film_test.csv'), index=False)
    truth.to_csv(os.path.join(out_dir, 'dose_sweep_truth.csv'), index=False)

    print(f"\nWrote {out_dir}")
    print(f"  train {len(train_ids)} patients | test {len(test_ids)} patients")
    print(f"  dose levels seen in training: {sorted(pd.to_numeric(df['AMT'], errors='coerce').dropna().unique())}")
    print(f"  sweep truth rows: {len(truth)}  ({len(sweep)} doses x {len(test_ids)} held-out patients)")


if __name__ == '__main__':
    main()
