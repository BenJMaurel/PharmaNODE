import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy import stats
from scipy.special import inv_boxcox
import datetime
import glob
import os
import re 
import chardet

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

def extract_gen_tac(file_path = ["virtual_cohort_train.csv", "virtual_cohort_test.csv"], plot = True, exp = None):
    data_dict = {}
    if exp:
        file_path[0] = f"exp_run_all/{exp}/virtual_cohort_train.csv"
        try:
            file_path[1] = f"exp_run_all/{exp}/virtual_cohort_test.csv"
        except:
            pass
    df1 = pd.read_csv(file_path[0], sep=",")
    df2 = pd.read_csv(file_path[1], sep=",")
    df = pd.concat([df1, df2])

    df.columns = df.columns.str.strip()
    # Combine ID and group to make a unique new ID
    
    df['ID_2'] = df['ID'].astype(str)
    df['ID_new'] = df['ID'].astype(str)
    df['OUT'] = pd.to_numeric(df['DV'].replace('.', None), errors='coerce')
    df['DOSE'] = pd.to_numeric(df['AMT'].replace('.', None), errors='coerce')
    df['TIME'] = pd.to_numeric(df['TIME'].replace('.', None), errors='coerce')
    ids_to_remove = df.loc[df['OUT'] > 50000, 'ID_new'].unique()
    # Step 2: Filter out rows where ID_new is in that list
    df= df[~df['ID_new'].isin(ids_to_remove)]
    
    # Scaling
    df['DOSE'] = df['DOSE'] / df['DOSE'].max()
    
    df['OUT'] = df['OUT']
    max_out = df['OUT'].max()
    df['OUT'] = df['OUT'] / max_out

    boxcox_transformed_data, best_lambda = stats.boxcox(df[df['OUT'] > 0]['OUT']) # Add a tiny constant to ensure all data is > 0
    mask = (df['OUT'] > 0) & (df['OUT'].notna())
    df.loc[mask, 'OUT'] = boxcox_transformed_data

    # Optional: Drop intermediate columns
    unique_patient = df['ID_new'].unique()
    
    # Exclude 
    aucs = []
    today_date = datetime.date.today()
    date_str = today_date.strftime("%Y%m%d")
    try:
        search_pattern = os.path.join(f"exp_run_all/{exp}", 'tacro_mapbayest_auc_*.csv')
        # 2. Find files matching the pattern
        matching_files = glob.glob(search_pattern)
        try:
            filename = matching_files[-1]
        except:
            filename = matching_files[0]
        data_be = pd.read_csv(filename)
    except:
        yesterday_date = today_date - datetime.timedelta(days=1)
        date_str = yesterday_date.strftime("%Y%m%d")
        filename = f"exp_run_all/{exp}/tacro_mapbayest_auc_{date_str}.csv"
        # filename = 'exp_all_runs/tacro_mapbayest_auc_20250926.csv'
        data_be = pd.read_csv(filename)
    for patient_id in unique_patient:
         
        patient_df = df[df['ID_new'] == patient_id]
        patient_df_clean = patient_df[patient_df['OUT'].notna()]
        
        patient_df_clean.loc[:, 'TIME'] = patient_df_clean['TIME'] - patient_df_clean['TIME'].iloc[0]
        nbr_ss = patient_df_clean['nbr_ss'].tolist()[0]
        y_times = patient_df_clean['TIME'].tolist()
        y_values = patient_df_clean['OUT'].astype(float).tolist()
        auc = patient_df_clean['AUC'].values[1]
        target_times = [0., 1., 3.]
        x_times = []
        for target in target_times:
            # Find the absolute difference between each time and the target
            diffs = np.abs(patient_df_clean['TIME'] - target)
            # Get the index of the closest value
            closest_idx = diffs.idxmin()
            # Append the closest time value
            x_times.append(float(patient_df_clean.loc[closest_idx, 'TIME']))
        x_values = patient_df_clean[patient_df_clean['TIME'].isin(x_times)]['OUT'].astype(float).tolist()[:3]
        
        if x_values and len(x_times) == len(target_times):
                y_true_times = y_times
                patient_df_clean = patient_df[patient_df['DOSE'].notna()]
                patient_be = data_be[data_be['ID'] == int(patient_id)]
                doses = patient_df['DOSE'].astype(float).tolist()[0]
                try:
                    # For visualisation inside the latent space, can be relevant.
                    others = [patient_be['Cl'].values[0], patient_be['Vc'].values[0], patient_be['Q'].values[0], patient_be['Vp'].values[0], patient_be['Ktr'].values[0], patient_df['HT'].values[0]]
                except:
                    others = [-1 for i in range(6)]
                
                if (patient_df['CYP'] == 1).all():
                    classe = 1 # 0: Rein, 1: Card, 2: Poumons classe = organ
                else: 
                    classe = 0
                if float(patient_df['II'].values[0]) == 24:
                    traitement = 0 # 0: ADV
                else:
                    traitement = 1 # 1: PRO
                auc_be = patient_be['auc_ipred'].values/max_out
                if len(auc_be) ==0:
                    auc_be = np.array([0.0])
                static = [doses, traitement, classe]
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
                    'others' : torch.tensor(others),
                    'auc_be': torch.tensor(auc_be),
                    'auc_red' : torch.tensor(auc/max_out)}
                
    # if plot:
    #     # Plot OUT vs TIME for each subject
    #     # plt.figure(figsize=(10, 6))
    #     # plt.hist(aucs, bins = 20)
    #     for subject_id, group in data_dict.items():
    #         print(subject_id)
    #         plt.plot(group['times_val'], group['values_val'], marker='o', label=f'ID {(subject_id)}')
        
    #     plt.title('PK Data: OUT vs TIME by Subject')
    #     plt.xlabel('Time')
    #     plt.ylabel('Concentration (OUT)')
    #     plt.legend(title='Subject ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    #     plt.tight_layout()
    #     plt.grid(True)
    #     plt.show()

    return data_dict, [max_out, best_lambda]


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