###########################
# Scenario A generator: time-varying covariates across occasions.
#
# Each patient is observed at two visits that differ in THREE ways at once:
#   * dose            d1, d2 ~ U(1, 8) mg          (independent draws)
#   * time post-graft days1 ~ U(7, 30), days2 = days1 + U(30, 180)
#   * hematocrit      Ht1 ~ N(28, 4)%, Ht2 ~ N(36, 4)%   (anaemia resolving)
#
# Clearance carries a non-linear dependence on both:
#     CL_i(occasion) = CL_base_i * (Ht/30)^theta_Ht * f(days)
#     f(days) = 1 + A * exp(-days / tau)      (graft stabilisation)
# so the visit-2 profile differs from visit 1 through a dose change AND a
# physiological drift, which is the confound the experiment is about.
#
# CL_base_i (the patient's dose- and occasion-invariant clearance) is written
# to covshift_truth.csv for the disentanglement probe.
#
# PURELY ADDITIVE: imports the published simulator and changes nothing in it.
###########################

import os
import argparse
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from gen_tacro_film import TacrolimusPK, generate_patient_visit

THETA_HT = -3.14          # same exponent the published simulator uses for Ht on CL
GRAFT_A, GRAFT_TAU = 0.5, 60.0


def graft_factor(days):
    """Elimination is elevated early post-graft and stabilises: 1 + A*exp(-days/tau)."""
    return 1.0 + GRAFT_A * float(np.exp(-days / GRAFT_TAU))


def build_cohort(num_patients, nbr_ss=6, seed=None):
    if seed is not None:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    obs = torch.tensor([0, 0.33, 0.67, 1., 1.5, 2., 3., 4., 6., 9., 12., 24.]) + 24 * nbr_ss
    rows, truth = [], []
    for pid in range(1, num_patients + 1):
        form = random.choice(['Prograf', 'Advagraf'])
        cyp = random.choice(['expresser', 'non_expresser'])
        # hematocrit is applied per occasion below, so the constructor gets the
        # neutral value 35.0 and its built-in (Ht/35)^theta term is exactly 1.
        pk = TacrolimusPK(formulation=form, hematocrit=35.0, distribution_type='log_normal',
                          cyp_status=cyp, scenario=2)
        pk._sample_individual_parameters()
        cl_base = float(pk.individual_params['CL'])

        d1, d2 = random.uniform(1.0, 8.0), random.uniform(1.0, 8.0)
        days1 = random.uniform(7.0, 30.0)
        ddays = random.uniform(30.0, 180.0)
        days2 = days1 + ddays
        ht1 = float(np.random.normal(28.0, 4.0))
        ht2 = float(np.random.normal(36.0, 4.0))
        ht1 = float(np.clip(ht1, 15.0, 50.0)); ht2 = float(np.clip(ht2, 15.0, 50.0))

        for visit, dose, days, ht in ((1, d1, days1, ht1), (2, d2, days2, ht2)):
            cl_occ = cl_base * (ht / 30.0) ** THETA_HT * graft_factor(days)
            pk.individual_params['CL'] = torch.tensor(cl_occ, device=pk.device)
            pk.dose_mg = dose
            r = generate_patient_visit(pk, visit, pid, nbr_ss, obs, form, ht, cyp)
            for row in r:
                row['DAYS'] = days
                row['CL_OCC'] = cl_occ
            rows += r
        pk.individual_params['CL'] = torch.tensor(cl_base, device=pk.device)
        truth.append(dict(ID=pid, CL_base=cl_base, DRUG=form, CYP=cyp,
                          d1=d1, d2=d2, days1=days1, days2=days2, ddays=ddays,
                          Ht1=ht1, Ht2=ht2,
                          CL1=cl_base * (ht1/30.)**THETA_HT * graft_factor(days1),
                          CL2=cl_base * (ht2/30.)**THETA_HT * graft_factor(days2)))
        if pid % 50 == 0:
            print(f"  ...{pid}/{num_patients}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(truth)


def main():
    p = argparse.ArgumentParser('Scenario A: inter-occasion covariate shift')
    p.add_argument('--exp', type=str, required=True)
    p.add_argument('--num_patients', type=int, default=1000)
    p.add_argument('--test-fraction', type=float, default=0.2)
    p.add_argument('--out-root', type=str, default='./results/exp_film_run')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()

    out = os.path.join(a.out_root, str(a.exp)); os.makedirs(out, exist_ok=True)
    df, truth = build_cohort(a.num_patients, seed=a.seed)
    ids = df['ID'].unique()
    tr, te = train_test_split(ids, test_size=a.test_fraction, shuffle=False)
    df[df['ID'].isin(tr)].to_csv(os.path.join(out, 'virtual_cohort_film_train.csv'), index=False)
    df[df['ID'].isin(te)].to_csv(os.path.join(out, 'virtual_cohort_film_test.csv'), index=False)
    truth.to_csv(os.path.join(out, 'covshift_truth.csv'), index=False)
    print(f"\nWrote {out}\n  train {len(tr)} / test {len(te)} patients")
    print(f"  d1,d2 ~ U(1,8) | days1 ~ U(7,30), ddays ~ U(30,180) | Ht1~N(28,4), Ht2~N(36,4)")
    print(f"  CL ratio visit2/visit1: median {np.median(truth.CL2/truth.CL1):.3f} "
          f"[{np.percentile(truth.CL2/truth.CL1,5):.2f}, {np.percentile(truth.CL2/truth.CL1,95):.2f}]")


if __name__ == '__main__':
    main()
