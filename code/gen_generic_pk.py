import argparse
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split

from lib.pk_drug import DrugStudyConfig, generate_virtual_cohort
from lib.classic_pk import ClassicPKModel

def main():
    parser = argparse.ArgumentParser("Generate Generic PK Dataset")
    parser.add_argument("--exp", type=str, required=True, help="Experiment ID")
    parser.add_argument("--num_patients", type=int, default=100, help="Number of patients to generate")
    parser.add_argument("--compartments", type=int, default=2, help="Number of compartments (1, 2, or 3)")
    parser.add_argument("--absorption", type=str, default="oral", choices=["oral", "iv"], help="Absorption type")
    parser.add_argument("--distribution", type=str, default="log-normal", choices=["log-normal", "normal"], help="PK parameter distribution")
    
    # Advanced Custom PK Args
    parser.add_argument("--pop_cl", type=float, default=10.5, help="Population Clearance")
    parser.add_argument("--pop_vc", type=float, default=45.0, help="Population Central Volume")
    parser.add_argument("--pop_q", type=float, default=15.0, help="Inter-compartmental Clearance")
    parser.add_argument("--pop_vp", type=float, default=60.0, help="Peripheral Volume")
    parser.add_argument("--res_prop", type=float, default=0.1, help="Proportional Residual Error")
    parser.add_argument("--res_add", type=float, default=0.01, help="Additive Residual Error")
    parser.add_argument("--covariates", type=str, default="WT:50-100", help="Comma separated list of covariates with ranges")
    args = parser.parse_args()

    # Parse covariates safely: format WT:50-100
    covs = []
    covariate_ranges = {}
    for c in args.covariates.split(","):
        c = c.strip()
        if not c: continue
        if ":" in c:
            name, bounds = c.split(":", 1)
            try:
                low, high = map(float, bounds.split("-"))
                covariate_ranges[name.strip()] = (low, high)
            except:
                pass
            covs.append(name.strip())
        else:
            covs.append(c)

    if not covs:
        covs = ["WT"]
        covariate_ranges = {"WT": (50.0, 100.0)}

    # Generic configuration
    config = DrugStudyConfig(
        exp_id=args.exp,
        dosing_interval_h=24.0,
        n_steady_state_cycles=4,
        observation_times=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 18.0, 24.0],
        sparse_times=[0.0, 1.0, 3.0], 
        auc_window=(0.0, 24.0),
        covariate_columns=covs,
        residual_prop_sd=args.res_prop,
        residual_add_sd=args.res_add,
        static_feature_names=["dose_norm"] + [c.lower() + "_norm" for c in covs],
    )
    
    # Inject covariate ranges into config temporarily so generate_virtual_cohort can use them
    config.covariate_ranges = covariate_ranges

    # Initialize the specific math model
    pop_params = {
        "CL": args.pop_cl,
        "Vc": args.pop_vc,
    }
    if args.compartments >= 2:
        pop_params["Q"] = args.pop_q
        pop_params["Vp"] = args.pop_vp
    if args.absorption == "oral":
        pop_params["ka"] = 1.2
        
    model = ClassicPKModel(n_patients=args.num_patients, n_compartments=args.compartments, absorption=args.absorption, pop_params=pop_params, distribution=args.distribution)

    os.makedirs(config.results_path, exist_ok=True)
    config.save_json()

    print(f"Generating generic cohort (Compartments: {args.compartments}, Absorption: {args.absorption}) for {args.num_patients} patients...")
    cohort_df = generate_virtual_cohort(config, model, n_patients=args.num_patients)

    patient_ids = cohort_df["ID"].unique()
    train_ids, test_ids = train_test_split(patient_ids, test_size=0.2, random_state=42)

    train_df = cohort_df[cohort_df["ID"].isin(train_ids)]
    test_df = cohort_df[cohort_df["ID"].isin(test_ids)]

    train_path = os.path.join(config.results_path, "virtual_cohort_train.csv")
    test_path = os.path.join(config.results_path, "virtual_cohort_test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Saved {len(train_ids)} train patients to {train_path}")
    print(f"Saved {len(test_ids)} test patients to {test_path}")

if __name__ == "__main__":
    main()
