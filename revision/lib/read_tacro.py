import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np

# Hematocrit as a 4th static channel is OPT-IN and off by default, so every existing
# caller keeps producing the published 3-channel static = [dose, formulation, CYP].
# Runs that want it set --static-dim 4, which calls set_static_hematocrit(True).
_WITH_HT = False


def set_static_hematocrit(flag):
    """Enable/disable the hematocrit channel. Call before building the data dict."""
    global _WITH_HT
    _WITH_HT = bool(flag)


def static_hematocrit_enabled():
    return _WITH_HT
from scipy import stats
from scipy.special import inv_boxcox
import datetime
import glob
import os
import re 
import chardet
import itertools

def auc_linuplogdown(conc, time):
    """
    Calculates the Area Under the Curve (AUC) using the linear-up / log-down method.

    This method is the standard for pharmacokinetic (PK) analysis.
    - It uses the linear trapezoidal rule when the concentration is increasing.
    - It uses the logarithmic trapezoidal rule when the concentration is decreasing.

    Args:
        conc (array-like): Array of concentration values.
        time (array-like): Array of time values, corresponding to concentrations.

    Returns:
        float: The calculated AUC.
    """
    # Ensure inputs are numpy arrays
    conc = np.asarray(conc)
    time = np.asarray(time)

    if len(conc) != len(time):
        raise ValueError("Concentration and time arrays must have the same length.")

    total_auc = 0.0

    # Iterate over each segment of the curve
    for i in range(len(time) - 1):
        t1, t2 = time[i], time[i+1]
        c1, c2 = conc[i], conc[i+1]

        # Skip if time interval is zero
        if t1 == t2:
            continue
        
        # Use linear rule if concentration is rising or for zero concentrations
        if c2 >= c1 or c1 <= 0 or c2 <= 0:
            # Linear trapezoidal rule
            segment_auc = (c1 + c2) * (t2 - t1) / 2.0
        else:
            # Logarithmic trapezoidal rule for falling concentrations
            segment_auc = (c1 - c2) * (t2 - t1) / (np.log(c1) - np.log(c2))
        
        total_auc += segment_auc

    return total_auc

def convert(value):
    if value.startswith('D') or value.startswith('J'):
        return int(value[1:])/180
    elif value.startswith('M'):
        return int(value[1:]) * 30/180  # Assuming 1 month = 30 days
    else:
        raise ValueError(f"Invalid format: {value}")

def extract_gen_tac(file_path=["virtual_cohort_train.csv", "virtual_cohort_test.csv"], plot=True, exp=None):
   
    data_dict = {}
    
    # 1. Load the dataset (handles standard or film datasets)
    if exp:
        try:
            file_path[0] = f"./results/{exp}/virtual_cohort_train.csv"
            file_path[1] = f"./results/{exp}/virtual_cohort_test.csv"
            df1 = pd.read_csv(file_path[0], sep=",")
            df2 = pd.read_csv(file_path[1], sep=",") if os.path.exists(file_path[1]) else pd.DataFrame()
            df = pd.concat([df1, df2])
        except:
            file_path[0] = f"./results/{exp}/virtual_cohort_film_train.csv"
            file_path[1] = f"./results/{exp}/virtual_cohort_film_test.csv"
            df1 = pd.read_csv(file_path[0], sep=",")
            df2 = pd.read_csv(file_path[1], sep=",") if os.path.exists(file_path[1]) else pd.DataFrame()
            df = pd.concat([df1, df2])
    else:
        df1 = pd.read_csv(file_path[0], sep=",")
        df2 = pd.read_csv(file_path[1], sep=",") if len(file_path) > 1 and os.path.exists(file_path[1]) else pd.DataFrame()
        df = pd.concat([df1, df2])

    df.columns = df.columns.str.strip()
    if 'VISIT' in df.columns:
        df['ID_new'] = df['ID'].astype(str) + "_" + df['VISIT'].astype(str)
    else:
        df['ID_new'] = df['ID'].astype(str)
    # df['ID_2'] = df['ID'].astype(str)
    df['OUT'] = pd.to_numeric(df['DV'].replace('.', None), errors='coerce')
    df['DOSE'] = pd.to_numeric(df['AMT'].replace('.', None), errors='coerce')
    df['TIME'] = pd.to_numeric(df['TIME'].replace('.', None), errors='coerce')
    
    ids_to_remove = df.loc[df['OUT'] > 50000, 'ID_new'].unique()
    df = df[~df['ID_new'].isin(ids_to_remove)]
    
    # Scaling
    df['DOSE'] = df['DOSE'] / df['DOSE'].max()
    
    max_out = df['OUT'].max()
    df['OUT'] = df['OUT'] / max_out

    mask = (df['OUT'] > 0) & (df['OUT'].notna())
    boxcox_transformed_data, best_lambda = stats.boxcox(df[mask]['OUT'])
    df.loc[mask, 'OUT'] = boxcox_transformed_data

    unique_patient = df['ID_new'].unique()
    
    # ---------------------------------------------------------
    # 2. SAFELY Load MapBayesian Estimations (Make it Optional)
    # ---------------------------------------------------------
    data_be = None
    if exp:
        try:
            search_pattern = os.path.join(f"./results/{exp}", 'tacro_mapbayest_auc_*.csv')
            matching_files = glob.glob(search_pattern)
            if matching_files:
                filename = matching_files[-1]
                data_be = pd.read_csv(filename)
            else:
                # Try yesterday's date as fallback
                today_date = datetime.date.today()
                yesterday_date = today_date - datetime.timedelta(days=1)
                date_str = yesterday_date.strftime("%Y%m%d")
                fallback_filename = f"./results/{exp}/tacro_mapbayest_auc_{date_str}.csv"
                if os.path.exists(fallback_filename):
                    data_be = pd.read_csv(fallback_filename)
        except Exception as e:
            print(f"Note: MapBayEst file not found or could not be loaded ({e}). Proceeding without it.")
    
    # ---------------------------------------------------------
    # 3. Process Patients
    # ---------------------------------------------------------
    for patient_id in unique_patient:
        patient_df = df[df['ID_new'] == patient_id]
        patient_df_clean = patient_df[patient_df['OUT'].notna()]
        
        if patient_df_clean.empty:
            continue
            
        patient_df_clean.loc[:, 'TIME'] = patient_df_clean['TIME'] - patient_df_clean['TIME'].iloc[0]
        nbr_ss = patient_df_clean['nbr_ss'].tolist()[0]
        y_times = patient_df_clean['TIME'].tolist()
        y_values = patient_df_clean['OUT'].astype(float).tolist()
        
        # Safe AUC extraction
        auc = patient_df_clean['AUC'].values[1] if len(patient_df_clean['AUC'].values) > 1 else patient_df_clean['AUC'].values[0]
        
        target_times = [0., 1., 3.]
        x_times = []
        for target in target_times:
            diffs = np.abs(patient_df_clean['TIME'] - target)
            closest_idx = diffs.idxmin()
            x_times.append(float(patient_df_clean.loc[closest_idx, 'TIME']))
            
        x_values = patient_df_clean[patient_df_clean['TIME'].isin(x_times)]['OUT'].astype(float).tolist()[:3]
        
        if x_values and len(x_times) == len(target_times):
            y_true_times = y_times
            doses = patient_df['DOSE'].astype(float).tolist()[0]
            
            # --- Safe extraction of 'others' and 'auc_be' ---
            others = [-1.0] * 6 # Default to -1.0
            auc_be_val = np.array([0.0])
            
            # If MapBayEst data is available, try to use it
            if data_be is not None:
                patient_be = data_be[data_be['ID'] == int(patient_id)]
                if not patient_be.empty:
                    try:
                        others = [patient_be['Cl'].values[0], patient_be['Vc'].values[0], patient_be['Q'].values[0], 
                                  patient_be['Vp'].values[0], patient_be['Ktr'].values[0], patient_df['HT'].values[0]]
                        
                        auc_be_raw = patient_be['auc_ipred'].values / max_out
                        if len(auc_be_raw) > 0:
                            auc_be_val = auc_be_raw
                    except Exception as e:
                        pass # Fallback to trying patient_df if columns are missing
            
            # If 'others' is still default, fallback to patient_df data if available
            if others[0] == -1.0:
                try:
                    if 'K_ELIM' in patient_df.columns:
                        k_elim = patient_df['K_ELIM'].astype(float).values[0]
                        k_12 = patient_df['K_12'].astype(float).values[0]
                        k_21 = patient_df['K_21'].astype(float).values[0]
                        ht = patient_df['HT'].astype(float).values[0]
                        others = [k_elim, k_12, k_21, -1, -1, ht]
                    elif 'HT' in patient_df.columns:
                        others = [-1.0, -1.0, -1.0, -1.0, -1.0, patient_df['HT'].astype(float).values[0]]
                except:
                    pass
            # ------------------------------------------------
            
            if (patient_df['CYP'] == 1).all():
                classe = 1 # 0: Rein, 1: Card, 2: Poumons classe = organ
            else: 
                classe = 0
                
            if float(patient_df['II'].values[0]) == 24:
                traitement = 0 # 0: ADV
            else:
                traitement = 1 # 1: PRO
                
            static = [doses, traitement, classe]
            if _WITH_HT and 'HT' in patient_df.columns:
                # opt-in (--static-dim 4), as in extract_gen_tac_film; off by default -> published 3 channels
                static.append(float(patient_df['HT'].astype(float).values[0]) / 35.0)
            
            data_dict[patient_id] = {
                'times_val': torch.tensor(y_times),
                'values_val': torch.tensor(y_values),
                'y_true_times': torch.tensor(y_true_times),
                'x_values': torch.tensor(x_values),
                'x_times': torch.tensor(x_times),
                'doses': torch.tensor(doses),
                'static': torch.tensor(static),
                'patient_id': patient_id,
                'dataset_number': torch.tensor(int(0.)),
                'others': torch.tensor(others),
                'auc_be': torch.tensor(auc_be_val),
                'auc_red': torch.tensor(auc / max_out)
            }

    return data_dict, [max_out, best_lambda]

def extract_gen_tac_film(file_path=["virtual_cohort_film_train.csv", "virtual_cohort_film_test.csv"], exp=None, scale=None):
    """scale=[max_out, best_lambda] applies a FIXED normalisation instead of refitting
    it from these files. Needed whenever data is encoded by a model trained elsewhere:
    refitting changes max_out and the Box-Cox lambda, so the inputs no longer match the
    transform the model saw. Default None reproduces the original behaviour exactly."""
    if exp:
        file_path = [f"./results/{exp}/virtual_cohort_film_train.csv", f"./results/{exp}/virtual_cohort_film_test.csv"]
    if len(file_path) >1:
        df = pd.concat([pd.read_csv(f) for f in file_path if os.path.exists(f)])
    else:
        df = pd.read_csv(file_path[0])
    df['OUT'] = pd.to_numeric(df['DV'].replace('.', None), errors='coerce')
    df['DOSE'] = pd.to_numeric(df['AMT'].replace('.', None), errors='coerce')
    df['TIME'] = pd.to_numeric(df['TIME'].replace('.', None), errors='coerce')
    
    # DOSE is normalised by the cohort max. When a fixed scale is supplied and
    # carries a third element, pin the dose max too: a cohort whose dose range
    # differs from the training one would otherwise re-scale every dose and the
    # model would read them wrong (e.g. 5 mg -> 0.625 instead of 1.0).
    _dose_max = float(scale[2]) if (scale is not None and len(scale) > 2) else df['DOSE'].max()
    df['DOSE'] = df['DOSE'] / _dose_max
    max_out = df['OUT'].max() if scale is None else float(scale[0])
    df['OUT'] = df['OUT'] / max_out
    
    mask = (df['OUT'] > 0) & (df['OUT'].notna())
    if scale is None:
        boxcox_transformed_data, best_lambda = stats.boxcox(df[mask]['OUT'])
    else:
        best_lambda = float(scale[1])
        boxcox_transformed_data = stats.boxcox(df[mask]['OUT'], lmbda=best_lambda)
    df.loc[mask, 'OUT'] = boxcox_transformed_data

    data_dict = {}
    unique_patients = df['ID'].unique()
    
    for patient_id in unique_patients:
        patient_df = df[df['ID'] == patient_id]
        data_dict[patient_id] = {}
        auc = [0.0,0.0]
        for visit in [1, 2]:
            visit_df = patient_df[patient_df['VISIT'] == visit]
            if visit_df.empty: continue
            
            visit_df_clean = visit_df[visit_df['OUT'].notna()]
            visit_df_clean.loc[:, 'TIME'] = visit_df_clean['TIME'] - visit_df_clean['TIME'].iloc[0]
            auc[visit-1] = visit_df['AUC'].values[1] if len(visit_df['AUC'].values) > 1 else visit_df['AUC'].values[0]
            
            y_times = visit_df_clean['TIME'].tolist()
            y_values = visit_df_clean['OUT'].astype(float).tolist()
            k_elim = patient_df['K_ELIM'].astype(float).values[0]
            k_12 = patient_df['K_12'].astype(float).values[0]
            k_21 = patient_df['K_21'].astype(float).values[0]
            # HT is per-OCCASION in the covariate-shift cohorts; in every earlier
            # cohort it is constant within a patient, so this returns the same value.
            ht = visit_df['HT'].astype(float).values[0]
            days = float(visit_df['DAYS'].astype(float).values[0]) if 'DAYS' in visit_df.columns else 0.0
            others = [k_elim, k_12, k_21, -1, -1, ht]
            # others = [-1, -1, -1, -1, -1, -1]
            x_times, x_values = [], []
            for target in [0., 1., 3.]:
                idx = np.abs(visit_df_clean['TIME'] - target).idxmin()
                x_times.append(float(visit_df_clean.loc[idx, 'TIME']))
                x_values.append(float(visit_df_clean.loc[idx, 'OUT']))
            
            dose = visit_df[visit_df['DOSE'].notna()]['DOSE'].astype(float).tolist()[0]
            classe = 1 if (visit_df['CYP'] == 1).all() else 0
            traitement = 0 if float(visit_df['II'].values[0]) == 24 else 1
            # 4th channel is hematocrit, which a scenario-4 cohort varies and puts on Km.
            # Models built with static_dim=3 slice it off, so older checkpoints are unaffected.
            static = [dose, traitement, classe]
            if _WITH_HT:
                # float() is load-bearing: ht is numpy float64 and would promote the
                # whole static tensor to Double, which the encoder's Linear rejects.
                static.append(float(ht) / 35.0)

            data_dict[patient_id][f'v{visit}'] = {
                'times_val': torch.tensor(y_times),
                'values_val': torch.tensor(y_values),
                'x_times': torch.tensor(x_times),
                'x_values': torch.tensor(x_values),
                'others': torch.tensor(others),
                'doses': torch.tensor(dose),
                'static': torch.tensor(static),
                'macro_time': torch.tensor([0.0], dtype=torch.float32),
                # occasion covariates, scaled to O(1): [days post-graft / 100, Ht / 30].
                # Zero-days for cohorts without a DAYS column, so old data is unchanged.
                'occ': torch.tensor([days / 100.0, ht / 30.0], dtype=torch.float32),
                'auc_red': torch.tensor(auc[visit - 1] / max_out),
                'delta_t': torch.tensor([0.0])
            }
            
    return data_dict, [max_out, best_lambda, _dose_max]

class TacroFilmDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict):
        self.patient_ids = list(data_dict.keys())
        self.data_dict = data_dict

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        v1 = self.data_dict[pid]['v1']
        v2 = self.data_dict[pid]['v2']
        try:
            delta_t = self.data_dict[pid]['delta_t']
        except:
            delta_t = self.data_dict[pid]['v2']['delta_t']
        return (
            v1['times_val'], v1['values_val'], v1['x_times'], v1['x_values'], v1['doses'], v1['static'], v1['auc_red'],v1['others'],
            v2['times_val'], v2['values_val'], v2['x_times'], v2['x_values'], v2['doses'], v2['static'], v2['auc_red'], v2['others'],
            # occasion covariates go HERE, at fixed positive indices 16/17: the collate
            # reads macro_time/delta_t/pid by NEGATIVE index (-3/-2/-1), so appending
            # at the end would silently shift them.
            v1.get('occ', torch.zeros(2)), v2.get('occ', torch.zeros(2)),
            v1['macro_time'], delta_t, pid
        )
    
def collate_fn_tacro_film(batch, device):
    # V1 (Encoder Input & Target)

    nobv = 8 #number_of_obs_by_visit
    obs_v1 = torch.stack([b[3] for b in batch]).unsqueeze(-1).to(device)
    tp_obs_v1 = torch.tensor([0.0, 1.0, 3.0]).to(device)
    dose_v1 = torch.stack([b[4] for b in batch]).to(device)
    static_v1 = torch.stack([b[5] for b in batch]).to(device)
    data_pred_v1 = torch.stack([b[1] for b in batch]).unsqueeze(-1).to(device)
    #tp_pred_v1 = torch.unique(torch.stack([b[0] for b in batch])).to(device)
    tp_pred_v1 = torch.stack([b[0] for b in batch]).mean(dim=0).to(device)
    auc_red_v1 = torch.stack([b[6] for b in batch]).to(device)
    others_v1 = torch.stack([b[7] for b in batch]).to(device)
    # --- NEW: V2 (Encoder Input & Target) ---
    obs_v2 = torch.stack([b[nobv +3] for b in batch]).unsqueeze(-1).to(device)
    tp_obs_v2 = torch.tensor([0.0, 1.0, 3.0]).to(device)
    dose_v2 = torch.stack([b[nobv +4] for b in batch]).to(device)
    static_v2 = torch.stack([b[nobv +5 ] for b in batch]).to(device)
    data_pred_v2 = torch.stack([b[nobv +1] for b in batch]).unsqueeze(-1).to(device)
    # tp_pred_v2 = torch.unique(torch.stack([b[nobv] for b in batch])).to(device)
    tp_pred_v2 = torch.stack([b[nobv] for b in batch]).mean(dim=0).to(device) # NEW
    auc_red_v2 = torch.stack([b[nobv+6] for b in batch]).to(device)
    others_v2 = torch.stack([b[nobv +7] for b in batch]).to(device)
    occ_v1 = torch.stack([b[16] for b in batch]).to(device)
    occ_v2 = torch.stack([b[17] for b in batch]).to(device)
    delta_t = torch.stack([b[-2] for b in batch]).to(device)
    t_v1 = torch.stack([b[-3] for b in batch]).to(device)
    return {
        "observed_data_v1": obs_v1, "observed_tp_v1": tp_obs_v1, "dose_v1": dose_v1, "static_v1": static_v1,
        "data_to_predict_v1": data_pred_v1, "tp_to_predict_v1": tp_pred_v1, "auc_red_v1" : auc_red_v1, "others_v1":others_v1,
        "others_v2": others_v2,
        
        # Add the new V2 sparse inputs
        "observed_data_v2": obs_v2, "observed_tp_v2": tp_obs_v2, "dose_v2": dose_v2, "static_v2": static_v2,
        "data_to_predict_v2": data_pred_v2, "tp_to_predict_v2": tp_pred_v2, "auc_red_v2" : auc_red_v2,
        "delta_t": delta_t,
        "t_v1" : t_v1,
        "occ_v1": occ_v1, "occ_v2": occ_v2,
        "patient_ids": [b[-1] for b in batch],
        "mode": "interp"
    }

class TacroDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict, is_train = False, mean=None, std=None):
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
        
        for patient_id, patient_dict in self.data_dict.items():
            patient_dict = self.data_dict[patient_id]
            self.data.append(patient_dict['values_val'])
            self.data_t.append(patient_dict['times_val'])
            self.data_t_input.append(patient_dict['x_times'])
            self.data_input.append(patient_dict['x_values'])
            self.dose.append(patient_dict['doses'])
            if 'static' in patient_dict:
                self.static.append(patient_dict['static'])
            self.auc_be.append(patient_dict['auc_be'])
            self.auc_red.append(patient_dict['auc_red'])
            self.y_true_times.append(patient_dict['y_true_times'])
            self.dataset.append(patient_dict['dataset_number'])
            self.patient_id.append(patient_dict['patient_id'])
            self.others.append(patient_dict['others'])
    def __len__(self):
        return len(self.data_dict)  
    

    def __getitem__(self, idx):
        return (self.data_t[idx], self.data[idx], self.data_t_input[idx], self.data_input[idx], self.static[idx], self.auc_be[idx], self.y_true_times[idx], self.dataset[idx], self.patient_id[idx], self.others[idx], self.auc_red[idx], self.dose[idx])


def collate_fn_tacro(batch, args, device, data_type = "train"):

    
    obs = torch.stack([batch[i][3] for i in range(len(batch))]).unsqueeze(-1)
    
    classes = torch.stack([batch[i][4] for i in range(len(batch))])
    
    tp_obs = torch.tensor([0.0, 1.0, 3.0])

    
    data_pred = torch.stack([batch[i][1] for i in range(len(batch))])
    auc_be = torch.stack([batch[i][5] for i in range(len(batch))])
    y_true_times = torch.stack([batch[i][6] for i in range(len(batch))])
    dataset = torch.stack([batch[i][7] for i in range(len(batch))])
    # tp_pred = batch[0][0]
    auc_red = torch.stack([batch[i][-2] for i in range(len(batch))])
    tp_pred = torch.unique(torch.stack([batch[i][0] for i in range(len(batch))]))
    others = torch.stack([batch[i][9] for i in range(len(batch))])
    patient_id = [batch[i][8] for i in range(len(batch))]
    
    if batch[0][-1].ndim == 0:
        dose = torch.stack([batch[i][-1] for i in range(len(batch))]).unsqueeze(1).expand(-1, len(tp_obs)).unsqueeze(-1)
       
        static = torch.stack([batch[i][4] for i in range(len(batch))])
        static2 = (torch.stack([batch[i][2] for i in range(len(batch))]) - torch.tensor([0.0, 1.0, 3.0]))
    else:
        dose = torch.stack([batch[i][-1] for i in range(len(batch))]).unsqueeze(1).expand(-1,3, -1)
    split_dict = {
        "observed_data": obs.clone(),
        "observed_tp": tp_obs.clone(),
        "data_to_predict": data_pred.unsqueeze(-1).clone(),
        "tp_to_predict": tp_pred.clone(),
        "dose": dose.clone(),
        "auc_be": auc_be.clone(),
        "auc_red": auc_red.clone(),
        "dataset_number": dataset.clone(),
        "y_true_times": y_true_times.clone(),
        "patient_id": patient_id,
        "static": static.clone(),
        "others": others.clone()}
    
    split_dict["mask_predicted_data"] = None 
    split_dict["labels"] = None 
    split_dict["mode"] = "interp"
    return split_dict