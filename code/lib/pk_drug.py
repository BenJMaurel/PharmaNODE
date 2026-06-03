"""
Generic PK cohort generation and NODE-compatible data loading for new drugs.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats

def auc_linuplogdown(conc, time):
    """Linear-up / log-down AUC (same rule as lib.read_tacro)."""
    conc = np.asarray(conc)
    time = np.asarray(time)
    if len(conc) != len(time):
        raise ValueError("Concentration and time arrays must have the same length.")
    total_auc = 0.0
    for i in range(len(time) - 1):
        t1, t2 = time[i], time[i + 1]
        c1, c2 = conc[i], conc[i + 1]
        if t1 == t2:
            continue
        if c2 >= c1 or c1 <= 0 or c2 <= 0:
            total_auc += (c1 + c2) * (t2 - t1) / 2.0
        else:
            total_auc += (c1 - c2) * (t2 - t1) / (np.log(c1) - np.log(c2))
    return total_auc


@dataclass
class DrugStudyConfig:
    """Central configuration for synthetic cohort generation and extraction."""

    exp_id: str = "tutorial_demo"
    output_dir: Optional[str] = None

    population_params: Dict[str, float] = field(default_factory=dict)
    ipv_omega: Dict[str, float] = field(default_factory=dict)
    residual_prop_sd: float = 0.15
    residual_add_sd: float = 0.05
    unit_scale: float = 1.0

    dose_choices: List[float] = field(default_factory=lambda: [50.0, 75.0, 100.0, 150.0])
    dosing_interval_h: float = 24.0
    n_steady_state_cycles: int = 3

    observation_times: List[float] = field(
        default_factory=lambda: [
            0.0, 0.33, 0.67, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 9.0, 12.0, 24.0
        ]
    )
    sparse_times: List[float] = field(default_factory=lambda: [0.0, 2.0, 8.0])
    auc_window: Tuple[float, float] = (0.0, 24.0)

    covariate_columns: List[str] = field(default_factory=lambda: ["WT"])
    wt_range: Tuple[float, float] = (50.0, 100.0)
    static_feature_names: List[str] = field(
        default_factory=lambda: ["dose_norm", "wt_norm", "pad"]
    )

    others_columns: List[str] = field(default_factory=list)

    @property
    def results_path(self) -> str:
        if self.output_dir:
            return self.output_dir
        return os.path.join("./results", str(self.exp_id))

    def dosing_times_absolute(self, n_cycles: Optional[int] = None) -> List[float]:
        n = n_cycles if n_cycles is not None else self.n_steady_state_cycles
        return [i * self.dosing_interval_h for i in range(n + 1)]

    def observation_times_absolute(self) -> List[float]:
        anchor = self.n_steady_state_cycles * self.dosing_interval_h
        return [anchor + t for t in self.observation_times]

    def save_json(self, path: Optional[str] = None) -> str:
        os.makedirs(self.results_path, exist_ok=True)
        out = path or os.path.join(self.results_path, "drug_config.json")
        payload = asdict(self)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return out

    @classmethod
    def load_json(cls, path: str) -> "DrugStudyConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_experiment(cls, exp_id: str) -> Optional["DrugStudyConfig"]:
        path = os.path.join("./results", str(exp_id), "drug_config.json")
        if os.path.isfile(path):
            return cls.load_json(path)
        return None


def default_patient_meta(config: DrugStudyConfig) -> Dict[str, float]:
    meta = {}
    ranges = getattr(config, "covariate_ranges", {})
    for col in config.covariate_columns:
        if col in ranges:
            meta[col] = random.uniform(*ranges[col])
        elif col == "WT":
            meta[col] = random.uniform(*config.wt_range)
        elif col == "HT":
            meta[col] = random.uniform(1.4, 2.0)
        else:
            meta[col] = 1.0
    return meta


def build_static_vector(
    dose_mg: float,
    dose_max: float,
    patient_row: Dict[str, Any],
    config: DrugStudyConfig,
) -> List[float]:
    dose_norm = dose_mg / dose_max if dose_max > 0 else 0.0
    static = [dose_norm]
    ranges = getattr(config, "covariate_ranges", {})
    for col in config.covariate_columns:
        val = float(patient_row.get(col, 1.0))
        # Use midpoint of range for rough normalization if provided
        if col in ranges:
            mid = sum(ranges[col]) / 2.0
            static.append(val / (mid if mid > 0 else 1.0))
        elif col == "WT":
            static.append(val / 70.0)
        elif col == "HT":
            static.append(val / 35.0)
        else:
            static.append(val)
    return static


def generate_virtual_cohort(config: DrugStudyConfig, pk_model, n_patients: int = 100) -> pd.DataFrame:
    """
    Batched generation of the virtual cohort. 
    Expects pk_model to support .simulate() returning (T, N) tensors.
    """
    obs_abs = torch.tensor(config.observation_times_absolute(), dtype=torch.float32)
    sim_times = obs_abs[obs_abs > 0]
    fine = torch.arange(0.0, sim_times.max().item() + 0.1, 0.1)
    all_sim_times = torch.unique(torch.cat([sim_times, fine]))

    dosing_times = config.dosing_times_absolute()
    anchor = config.n_steady_state_cycles * config.dosing_interval_h

    # Simulate batch
    true_all = pk_model.simulate(dosing_times, all_sim_times) * config.unit_scale # shape (T, N)
    
    mask = torch.isin(all_sim_times, sim_times)
    true_conc = true_all[mask, :] # shape (obs_T, N)
    
    tc_np = true_conc.cpu().numpy()
    tc_all_np = true_all.cpu().numpy()
    all_times_np = all_sim_times.cpu().numpy()
    sim_times_np = sim_times.cpu().numpy()

    auc_t0, auc_t1 = config.auc_window
    auc_start = anchor + auc_t0
    auc_end = anchor + auc_t1
    time_mask = (all_times_np >= auc_start) & (all_times_np <= auc_end)
    import scipy.integrate
    auc_np = scipy.integrate.trapezoid(tc_all_np[time_mask, :], all_times_np[time_mask] - anchor, axis=0)

    sd_error = config.residual_add_sd + config.residual_prop_sd * tc_np
    obs_np = np.clip(tc_np + sd_error * np.random.randn(*tc_np.shape), 0.0, None)
    
    for m in range(len(sim_times_np)):
        for n in range(n_patients):
            if obs_np[m, n] == 0:
                obs_np[m, n] = tc_np[m, n]

    wt_np = pk_model.cov_wt.cpu().numpy() if hasattr(pk_model, "cov_wt") else np.full(n_patients, 70.0)
    dose_np = pk_model.dose_mg.cpu().numpy()

    # Generate metadata for all patients
    patient_metas = [default_patient_meta(config) for _ in range(n_patients)]

    all_rows = []
    for n in range(n_patients):
        pid = n + 1
        base = {
            "ID": pid, "PERI": 1, "AUC": auc_np[n], "mdv": 1, "ss": 1, "ST": 0, "nbr_ss": config.n_steady_state_cycles
        }
        for col in config.covariate_columns:
            if col == "WT":
                base["WT"] = patient_metas[n].get("WT", 70.0)
            else:
                base[col] = patient_metas[n].get(col, 1.0)

        all_rows.append({**base, "TIME": 0.0, "DV": ".", "AMT": dose_np[n], "II": config.dosing_interval_h})
        for cycle in range(config.n_steady_state_cycles):
            t_dose = -config.dosing_interval_h * (config.n_steady_state_cycles - cycle)
            all_rows.append({**base, "TIME": t_dose, "DV": ".", "AMT": dose_np[n], "II": config.dosing_interval_h})

        for m, t_obs in enumerate(sim_times_np):
            all_rows.append({**base, "TIME": t_obs - anchor, "DV": obs_np[m, n], "AMT": ".", "II": ".", "mdv": 0, "ss": "."})

    df = pd.DataFrame(all_rows)
    return df.sort_values(by=['ID', 'TIME']).reset_index(drop=True)



def generate_virtual_cohort_film(
    pk_model_factory: Callable[[Dict[str, float]], Any],
    config: DrugStudyConfig,
    num_patients: int = 100,
    patient_meta_fn: Optional[Callable[[DrugStudyConfig], Dict[str, float]]] = None,
) -> pd.DataFrame:
    meta_fn = patient_meta_fn or default_patient_meta
    obs_abs = torch.tensor(config.observation_times_absolute(), dtype=torch.float32)
    sim_times = obs_abs[obs_abs > 0]
    start_time = 0.0
    end_time = sim_times.max().item()
    fine = torch.arange(start_time, end_time + 0.1, 0.1)
    all_sim_times = torch.unique(torch.cat([sim_times, fine]))

    dosing_times = config.dosing_times_absolute()
    anchor = config.n_steady_state_cycles * config.dosing_interval_h
    auc_t0, auc_t1 = config.auc_window
    auc_start = anchor + auc_t0
    auc_end = anchor + auc_t1

    all_rows: List[Dict[str, Any]] = []
    generated = 0
    patient_id = 0

    print(f"Generating FiLM data for {num_patients} virtual patients (2 visits each)...")

    while generated < num_patients:
        meta = meta_fn(config)
        pk_model = pk_model_factory(meta)
        pk_model._sample_individual_parameters()

        # Visit 1
        dose_1 = random.choice(config.dose_choices)
        pk_model.dose_mg = float(dose_1)
        
        true_all_1 = pk_model.simulate(dosing_times, all_sim_times) * config.unit_scale
        true_all_1 = true_all_1.flatten()

        mask = torch.isin(all_sim_times, sim_times)
        true_conc_1 = true_all_1[mask]
        sim_t_list = all_sim_times[mask].tolist()

        prop_sd = config.residual_prop_sd
        add_sd = config.residual_add_sd
        sd_error_1 = add_sd + prop_sd * true_conc_1
        noise_1 = torch.randn_like(true_conc_1)
        concentrations_1 = true_conc_1 + sd_error_1 * noise_1
        concentrations_1 = torch.clamp(concentrations_1, min=1e-6)

        auc_mask = (all_sim_times >= auc_start) & (all_sim_times <= auc_end)
        auc_times = (all_sim_times[auc_mask] - anchor).cpu().numpy()
        auc_conc_1 = true_all_1[auc_mask].cpu().numpy()
        if len(auc_times) < 2:
            continue
        auc_1 = float(auc_linuplogdown(auc_conc_1, auc_times))

        # Visit 2
        dose_2 = random.choice(config.dose_choices)
        while dose_2 == dose_1:
            dose_2 = random.choice(config.dose_choices)
        pk_model.dose_mg = float(dose_2)
        
        true_all_2 = pk_model.simulate(dosing_times, all_sim_times) * config.unit_scale
        true_all_2 = true_all_2.flatten()
        true_conc_2 = true_all_2[mask]
        
        sd_error_2 = add_sd + prop_sd * true_conc_2
        noise_2 = torch.randn_like(true_conc_2)
        concentrations_2 = true_conc_2 + sd_error_2 * noise_2
        concentrations_2 = torch.clamp(concentrations_2, min=1e-6)
        
        auc_conc_2 = true_all_2[auc_mask].cpu().numpy()
        auc_2 = float(auc_linuplogdown(auc_conc_2, auc_times))

        generated += 1
        patient_id = generated

        for visit, dose_mg, auc, conc_arr in [(1, dose_1, auc_1, concentrations_1), (2, dose_2, auc_2, concentrations_2)]:
            base = {
                "ID": patient_id,
                "VISIT": visit,
                "PERI": 1,
                "AUC": auc,
                "mdv": 1,
                "ss": 1,
                "ST": 0,
                "nbr_ss": config.n_steady_state_cycles,
            }
            for col in config.covariate_columns:
                base[col] = meta[col]

            all_rows.append({**base, "TIME": 0.0, "DV": ".", "AMT": dose_mg, "II": config.dosing_interval_h})

            for cycle in range(config.n_steady_state_cycles):
                t_dose = -config.dosing_interval_h * (config.n_steady_state_cycles - cycle)
                all_rows.append({**base, "TIME": t_dose, "DV": ".", "AMT": dose_mg, "II": config.dosing_interval_h})

            for t_obs, conc in zip(sim_t_list, conc_arr.tolist()):
                all_rows.append({**base, "TIME": t_obs - anchor, "DV": float(conc), "AMT": ".", "II": ".", "mdv": 0, "ss": "."})

    print("FiLM Generation complete.")
    return pd.DataFrame(all_rows)


def save_cohort_splits(
    cohort_df: pd.DataFrame,
    config: DrugStudyConfig,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs(config.results_path, exist_ok=True)
    config.save_json()

    unique_ids = cohort_df["ID"].unique()
    n_test = max(1, int(len(unique_ids) * test_size))
    test_ids = unique_ids[-n_test:]
    train_ids = unique_ids[:-n_test]

    train_df = cohort_df[cohort_df["ID"].isin(train_ids)]
    test_df = cohort_df[cohort_df["ID"].isin(test_ids)]

    train_path = os.path.join(config.results_path, "virtual_cohort_train.csv")
    test_path = os.path.join(config.results_path, "virtual_cohort_test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return train_df, test_df


def extract_gen_pk(
    config: DrugStudyConfig,
    file_path: Optional[List[str]] = None,
    plot: bool = False,
    exp: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[float]]:
    if exp is not None:
        config = DrugStudyConfig.from_experiment(exp) or config
        config.exp_id = str(exp)

    if file_path is None:
        base = config.results_path
        file_path = [
            os.path.join(base, "virtual_cohort_train.csv"),
            os.path.join(base, "virtual_cohort_test.csv"),
        ]

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
        patient_clean = patient_df[patient_df["OUT"].notna()].copy()
        if patient_clean.empty:
            continue

        patient_clean["TIME"] = patient_clean["TIME"] - patient_clean["TIME"].iloc[0]
        y_times = patient_clean["TIME"].tolist()
        y_values = patient_clean["OUT"].astype(float).tolist()

        auc_vals = patient_df["AUC"].dropna().unique()
        auc = float(auc_vals[0]) if len(auc_vals) else 0.0

        x_times, x_values = [], []
        for target in sparse:
            idx = (patient_clean["TIME"] - target).abs().idxmin()
            x_times.append(float(patient_clean.loc[idx, "TIME"]))
            x_values.append(float(patient_clean.loc[idx, "OUT"]))

        dose_rows = patient_df[patient_df["DOSE"].notna() & (patient_df["DOSE"] > 0)]
        doses = float(dose_rows["DOSE"].iloc[0]) if len(dose_rows) else 0.0

        meta_row = patient_df.iloc[0].to_dict()
        static = torch.tensor(
            build_static_vector(doses * dose_max, dose_max, meta_row, config),
            dtype=torch.float32,
        )

        others = [-1.0] * 6
        for i, col in enumerate(config.others_columns[:6]):
            if col in patient_df.columns:
                try:
                    others[i] = float(patient_df[col].astype(float).values[0])
                except (TypeError, ValueError):
                    pass
        if "WT" in patient_df.columns and others[5] == -1.0:
            others[5] = float(patient_df["WT"].astype(float).values[0])

        data_dict[patient_id] = {
            "times_val": torch.tensor(y_times, dtype=torch.float32),
            "values_val": torch.tensor(y_values, dtype=torch.float32),
            "y_true_times": torch.tensor(y_times, dtype=torch.float32),
            "x_values": torch.tensor(x_values, dtype=torch.float32),
            "x_times": torch.tensor(x_times, dtype=torch.float32),
            "doses": torch.tensor(doses, dtype=torch.float32),
            "static": static,
            "patient_id": patient_id,
            "dataset_number": torch.tensor(0.0),
            "others": torch.tensor(others, dtype=torch.float32),
            "auc_be": torch.tensor([0.0]),
            "auc_red": torch.tensor(auc / max_out, dtype=torch.float32),
        }

    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        for pid, d in list(data_dict.items())[:5]:
            ax.plot(d["times_val"], d["values_val"], label=str(pid))
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Transformed concentration")
        ax.legend()
        plt.tight_layout()
        plt.show()

    return data_dict, [float(max_out), float(best_lambda)]


class PKDataset(torch.utils.data.Dataset):
    """Same interface as TacroDataset."""

    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.data = []
        self.data_input = []
        self.data_t_input = []
        self.data_t = []
        self.dose = []
        self.static = []
        self.auc_be = []
        self.y_true_times = []
        self.dataset = []
        self.patient_id = []
        self.auc_red = []
        self.others = []

        for patient_id in self.data_dict:
            patient_dict = self.data_dict[patient_id]
            self.data.append(patient_dict["values_val"])
            self.data_t.append(patient_dict["times_val"])
            self.data_t_input.append(patient_dict["x_times"])
            self.data_input.append(patient_dict["x_values"])
            self.dose.append(patient_dict["doses"])
            self.static.append(patient_dict["static"])
            self.auc_be.append(patient_dict["auc_be"])
            self.auc_red.append(patient_dict["auc_red"])
            self.y_true_times.append(patient_dict["y_true_times"])
            self.dataset.append(patient_dict["dataset_number"])
            self.patient_id.append(patient_dict["patient_id"])
            self.others.append(patient_dict["others"])

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        return (
            self.data_t[idx],
            self.data[idx],
            self.data_t_input[idx],
            self.data_input[idx],
            self.static[idx],
            self.auc_be[idx],
            self.y_true_times[idx],
            self.dataset[idx],
            self.patient_id[idx],
            self.others[idx],
            self.auc_red[idx],
            self.dose[idx],
        )


def collate_fn_pk(batch, config: DrugStudyConfig, args=None, device=None, data_type="train"):
    sparse = torch.tensor(config.sparse_times, dtype=torch.float32)

    obs = torch.stack([batch[i][3] for i in range(len(batch))]).unsqueeze(-1)
    data_pred = torch.stack([batch[i][1] for i in range(len(batch))])
    auc_be = torch.stack([batch[i][5] for i in range(len(batch))])
    y_true_times = torch.stack([batch[i][6] for i in range(len(batch))])
    dataset = torch.stack([batch[i][7] for i in range(len(batch))])
    auc_red = torch.stack([batch[i][-2] for i in range(len(batch))])
    tp_pred = torch.unique(torch.stack([batch[i][0] for i in range(len(batch))]))
    others = torch.stack([batch[i][9] for i in range(len(batch))])
    patient_id = [batch[i][8] for i in range(len(batch))]

    if batch[0][-1].ndim == 0:
        dose = (
            torch.stack([batch[i][-1] for i in range(len(batch))])
            .unsqueeze(1)
            .expand(-1, len(sparse))
            .unsqueeze(-1)
        )
        static = torch.stack([batch[i][4] for i in range(len(batch))])
        static2 = torch.stack([batch[i][2] for i in range(len(batch))]) - sparse
    else:
        dose = torch.stack([batch[i][-1] for i in range(len(batch))]).unsqueeze(1).expand(
            -1, len(sparse), -1
        )
        static = torch.stack([batch[i][4] for i in range(len(batch))])
        static2 = torch.stack([batch[i][2] for i in range(len(batch))]) - sparse

    obs = obs.float()
    data_pred = data_pred.float()
    auc_be = auc_be.float()
    y_true_times = y_true_times.float()
    dataset = dataset.float()
    auc_red = auc_red.float()
    tp_pred = tp_pred.float()
    others = others.float()
    dose = dose.float()
    static = static.float()
    sparse = sparse.float()
    static2 = static2.float()

    if device is not None:
        obs = obs.to(device)
        data_pred = data_pred.to(device)
        auc_be = auc_be.to(device)
        y_true_times = y_true_times.to(device)
        dataset = dataset.to(device)
        auc_red = auc_red.to(device)
        tp_pred = tp_pred.to(device)
        others = others.to(device)
        dose = dose.to(device)
        static = static.to(device)
        sparse = sparse.to(device)
        static2 = static2.to(device)

    split_dict = {
        "observed_data": obs.clone(),
        "observed_tp": sparse.clone(),
        "data_to_predict": data_pred.unsqueeze(-1).clone(),
        "tp_to_predict": tp_pred.clone(),
        "dose": dose.clone(),
        "auc_be": auc_be.clone(),
        "auc_red": auc_red.clone(),
        "dataset_number": dataset.clone(),
        "y_true_times": y_true_times.clone(),
        "patient_id": patient_id,
        "static": static.clone(),
        "others": others.clone(),
        "mask_predicted_data": None,
        "labels": None,
        "mode": "interp",
    }
    return split_dict


def extract_gen_pk_film(
    config: DrugStudyConfig,
    file_path: Optional[List[str]] = None,
    exp: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[float]]:
    import os
    import pandas as pd
    from scipy import stats
    import torch
    import numpy as np

    if exp is not None:
        config = DrugStudyConfig.from_experiment(exp) or config
        config.exp_id = str(exp)

    if file_path is None:
        base = config.results_path
        file_path = [
            os.path.join(base, "virtual_cohort_train.csv"),
            os.path.join(base, "virtual_cohort_test.csv"),
        ]

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


