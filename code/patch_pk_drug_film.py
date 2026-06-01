import sys

with open("lib/pk_drug.py", "r") as f:
    content = f.read()

film_funcs = """
def extract_gen_pk_film(
    config: DrugStudyConfig,
    file_path: Optional[List[str]] = None,
    exp: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[float]]:
    if exp is not None:
        config = DrugStudyConfig.from_experiment(exp) or config
        config.exp_id = str(exp)

    if file_path is None:
        base = config.results_path
        file_path = [
            os.path.join(base, "virtual_cohort_film_train.csv"),
            os.path.join(base, "virtual_cohort_film_test.csv"),
        ]

    import os
    import pandas as pd
    from scipy import stats
    import torch
    import numpy as np

    dfs = []
    for fp in file_path:
        if os.path.exists(fp):
            dfs.append(pd.read_csv(fp, sep=","))
    if not dfs:
        raise FileNotFoundError(f"No cohort CSVs found under {config.results_path}")
    df = pd.concat(dfs, ignore_index=True)

    df.columns = df.columns.str.strip()
    df["ID_new"] = df["ID"].astype(str)
    df["OUT"] = pd.to_numeric(df["DV"].replace(".", None), errors="coerce")
    df["DOSE"] = pd.to_numeric(df["AMT"].replace(".", None), errors="coerce")
    df["TIME"] = pd.to_numeric(df["TIME"].replace(".", None), errors="coerce")

    dose_max = df["DOSE"].max()
    if pd.isna(dose_max) or dose_max <= 0:
        dose_max = 1.0
    df["DOSE"] = df["DOSE"] / dose_max

    max_out = df["OUT"].max()
    if pd.isna(max_out) or max_out <= 0:
        max_out = 1.0
    df["OUT"] = df["OUT"] / max_out

    mask = (df["OUT"] > 0) & (df["OUT"].notna())
    if mask.sum() < 2:
        raise ValueError("Need at least two positive concentration observations for Box-Cox.")
    boxcox_transformed, best_lambda = stats.boxcox(df.loc[mask, "OUT"])
    df.loc[mask, "OUT"] = boxcox_transformed

    sparse = list(config.sparse_times)
    data_dict: Dict[str, Any] = {}

    for patient_id in df["ID_new"].unique():
        patient_df = df[df["ID_new"] == patient_id]
        data_dict[patient_id] = {}
        
        for visit in [1, 2]:
            visit_df = patient_df[patient_df["VISIT"] == visit]
            if visit_df.empty: continue
            
            visit_clean = visit_df[visit_df["OUT"].notna()].copy()
            if visit_clean.empty:
                continue

            visit_clean["TIME"] = visit_clean["TIME"] - visit_clean["TIME"].iloc[0]
            y_times = visit_clean["TIME"].tolist()
            y_values = visit_clean["OUT"].astype(float).tolist()

            auc_vals = visit_df["AUC"].dropna().unique()
            auc = float(auc_vals[0]) if len(auc_vals) else 0.0

            x_times, x_values = [], []
            for target in sparse:
                idx = (visit_clean["TIME"] - target).abs().idxmin()
                x_times.append(float(visit_clean.loc[idx, "TIME"]))
                x_values.append(float(visit_clean.loc[idx, "OUT"]))

            dose_rows = visit_df[visit_df["DOSE"].notna() & (visit_df["DOSE"] > 0)]
            doses = float(dose_rows["DOSE"].iloc[0]) if len(dose_rows) else 0.0

            meta_row = visit_df.iloc[0].to_dict()
            static = torch.tensor(
                build_static_vector(doses * dose_max, dose_max, meta_row, config),
                dtype=torch.float32,
            )

            others = [-1.0] * 6

            data_dict[patient_id][f'v{visit}'] = {
                'times_val': torch.tensor(y_times),
                'values_val': torch.tensor(y_values),
                'x_times': torch.tensor(x_times),
                'x_values': torch.tensor(x_values),
                'others': torch.tensor(others),
                'doses': torch.tensor(doses),
                'static': static,
                'macro_time': torch.tensor([0.0], dtype=torch.float32),
                'auc_red': torch.tensor(auc / max_out),
                'delta_t': torch.tensor([0.0])
            }
            
    return data_dict, [max_out, best_lambda]

"""

if "def extract_gen_pk_film" not in content:
    with open("lib/pk_drug.py", "a") as f:
        f.write("\n" + film_funcs + "\n")
