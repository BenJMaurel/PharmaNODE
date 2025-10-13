import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy import stats
from scipy.special import inv_boxcox
import datetime
import glob
import os


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

def extract_pccp(file_path = "/Users/benjaminmaurel/Documents/Data_LG/MMF_data/PCCP-MPA.csv", plot = True):
    data_dict = {}
    df = pd.read_csv(file_path, sep=";", skipinitialspace=True,  skiprows=1)
    df = df.applymap(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x)
    df = df.apply(pd.to_numeric, errors='coerce')
    df.columns = df.columns.str.strip()
    # Combine ID and group to make a unique new ID
    df['ID_new'] = df['ID'].astype(int).astype(str).str.zfill(2) + df['PERI'].astype(str)
    ids_to_remove = df.loc[df['DV'] > 50000, 'ID_new'].unique()
    # Step 2: Filter out rows where ID_new is in that list
    df= df[~df['ID_new'].isin(ids_to_remove)]
    
    # plt.hist(df_filtered['DV'].dropna(), bins=30, edgecolor='black')
    # plt.xlabel('DV')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of DV')
    # plt.show()
    # import pdb; pdb.set_trace()
    df['DOSE'] = df['DOSE'] / df['DOSE'].max()
    # df['DOSE'] = df['DOSE'] / 1000
    df['DV'] = df['DV'] / 1000
    max_out = df['DV'].max()
    df['DV'] = df['DV'] / max_out
    
    # Optional: Drop intermediate columns
    unique_patient = df['ID_new'].unique()
    # Exclude 
    
    aucs = []
    data_be = pd.DataFrame(extract_pharmac_data_from_file('/Users/benjaminmaurel/Documents/Data_LG/MMF_data/pccp_pharmac.txt'))
    for patient_id in unique_patient:
        patient_df = df[df['ID_new'] == patient_id]
        patient_df_clean = patient_df[patient_df['DV'].notna()]
        # patient_df_clean['TIME'] = patient_df_clean['TIME'] - patient_df_clean['TIME'].tolist()[0]
        patient_df_clean.loc[:, 'TIME'] = patient_df_clean['TIME'] - patient_df_clean['TIME'].iloc[0]
        y_times = patient_df_clean['TIME'].tolist()
        y_values = patient_df_clean['DV'].astype(float).tolist()
        target_times = [0.33, 1., 3.]
        x_times = []
        for target in target_times:
            # Find the absolute difference between each time and the target
            diffs = np.abs(patient_df_clean['TIME'] - target)[:11]
            # Get the index of the closest value
            closest_idx = diffs.idxmin()
            # Append the closest time value
            x_times.append(float(patient_df_clean.loc[closest_idx, 'TIME']))
        x_values = patient_df_clean[:10][patient_df_clean['TIME'][:10].isin(x_times)]['DV'].astype(float).tolist()
        
        if x_values and len(x_times) == len(target_times) and len(y_times) >= 10 and y_times[-1] < 30 and x_times[-1] < 5:
                y_true_times = y_times
                y_times = [0, 0.33, 0.67, 1., 1.5, 2., 3., 4., 6., 8., 12.]
                patient_df_clean = patient_df[patient_df['DOSE'].notna()]
                if len(y_values) == 10:
                    y_true_times.append(12.)
                    y_values.append(min(y_values[0], y_values[-1]))
                auc = np.trapz(y_values, y_times)
                aucs.append(auc)
                patient_be = data_be[data_be['Nom'] == patient_id]
                if patient_be.empty:
                    print(patient_id, "NO BE for this patient")
                    continue
                doses = patient_df['DOSE'].astype(float).tolist()[0]
                # doses = torch.tensor([doses, cort, crea])
                traitement = 1 # 0: Ciclo, 1: Tacro, 2: Siro
                classe = 0 # 0: Rein, 1: Card, 2: Poumons classe = organ
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
                    'auc_be': torch.tensor(patient_be['AUC(0-12)'].values)}
    if plot:
        # Plot OUT vs TIME for each subject
        plt.figure(figsize=(10, 6))
        plt.hist(aucs, bins = 20)
        # for subject_id, group in data_dict.items():
        #     if group['values_val'].max() > 0.1:
        #         print(subject_id)
        #         plt.plot(group['times_val'], group['values_val'], marker='o', label=f'ID {(subject_id)}')
        
        # plt.title('PK Data: OUT vs TIME by Subject')
        # plt.xlabel('Time')
        # plt.ylabel('Concentration (OUT)')
        # plt.legend(title='Subject ID', bbox_to_anchor=(1.05, 1), loc='upper left')
        # plt.tight_layout()
        # plt.grid(True)
        plt.show()

    return data_dict, max_out

def extract_aadapt_tac(file_path = "/Users/benjaminmaurel/Documents/Data_LG/FileSenderDownload_7-2-2025__9-5-56/validation_modele_vs_trapeze/aadapt/AADAPT_proadv_281016.csv", plot = True):
    data_dict = {}
    df = pd.read_csv(file_path, sep=";", skipinitialspace=True,  skiprows=1)
    df = df.applymap(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x)
    
    # df = df.apply(pd.to_numeric, errors='coerce')
    df.columns = df.columns.str.strip()
    # Combine ID and group to make a unique new ID
    # df = df[(df["DRUG"] == "ADV") & (df["nycth"] == "J")]
    # df = df[(df['DRUG'] == 'PRO') & (df["nycth"] == "J")]
    df = df[(df["nycth"] == "J") | (df["nycth"].isna())]
    df['ID_new'] = df['#ID'] + df['periode'].astype(str)
    df['OUT'] = pd.to_numeric(df['OUT'].replace('.', None), errors='coerce')
    df['DOSE'] = pd.to_numeric(df['DOSE'].replace('.', None), errors='coerce')
    df['TIME'] = pd.to_numeric(df['TIME'].replace('.', None), errors='coerce')
    ids_to_remove = df.loc[df['OUT'] > 50000, 'ID_new'].unique()
    # Step 2: Filter out rows where ID_new is in that list
    df= df[~df['ID_new'].isin(ids_to_remove)]
    
    df['DOSE'] = df['DOSE'] / df['DOSE'].max()
    # df['DOSE'] = df['DOSE'] / 1000
    df['OUT2'] = df['OUT']
    max_out = df['OUT'].max()
    df['OUT2'] = df['OUT2']/max_out
    df['OUT'] = df['OUT']/ max_out
    boxcox_transformed_data, best_lambda = stats.boxcox(df['OUT'].dropna())
    mask = (df['OUT'] > 0) & (df['OUT'].notna())
    df.loc[mask, 'OUT'] = boxcox_transformed_data
    # Optional: Drop intermediate columns
    unique_patient = df['ID_new'].unique()
    # Exclude 
    aucs = []
    data_be = pd.DataFrame(extract_pharmac_data_from_file('/Users/benjaminmaurel/Documents/Data_LG/FileSenderDownload_7-2-2025__9-5-56/validation_modele_vs_trapeze/aadapt/pharmac_aadapt_pro_jd.txt'))
    data_be_2 = pd.DataFrame(extract_pharmac_data_from_file('/Users/benjaminmaurel/Documents/Data_LG/FileSenderDownload_7-2-2025__9-5-56/validation_modele_vs_trapeze/aadapt/pharmac_aadapt_adv_jd.txt'))
    data_be = pd.concat([data_be, data_be_2])
    df_auc = pd.read_csv('/Users/benjaminmaurel/tacro_mapbayest_auc_20250826.csv')
    for patient_id in unique_patient:
        #'02PF-8J7', '11RM-86M3', '10PB-8J7', '06JJ-87M3', '11HP-7J7'
        # if patient_id in ['10PB-8J7', '02PF-8J7', '11HP-7J7', '16EA-84M3', '06JJ-87M3']:
        #     continue
        if patient_id in ['02PF-8J7', '0400-8J7', '05MT-8J7']:
            continue
        patient_df = df[df['ID_new'] == patient_id]
        patient_df_clean = patient_df[patient_df['OUT'].notna()]
        # patient_df_clean['TIME'] = patient_df_clean['TIME'] - patient_df_clean['TIME'].tolist()[0]
        try:
            if patient_df_clean['TIME'].iloc[0] > 100:
                patient_df_clean.loc[:, 'TIME'] = patient_df_clean['TIME'] - patient_df_clean['TIME'].iloc[0]
        except:
            import pdb; pdb.set_trace()
        times_for_auc = patient_df_clean['TIME'].tolist()
        values_for_auc = patient_df_clean['OUT2'].astype(float).tolist()
        y_times = patient_df_clean['TIME'].tolist()[:10]
        y_values = patient_df_clean['OUT'].astype(float).tolist()[:10]

        # auc = np.trapz(np.array(values_for_auc), times_for_auc)
        auc = auc_linuplogdown(np.array(values_for_auc), times_for_auc)
        target_times = [0.0, 1., 3.]
        # target_times = [1., 3.]
        x_times = []
        for target in target_times:
            # Find the absolute difference between each time and the target
            diffs = np.abs(patient_df_clean['TIME'] - target)[:10]
            # Get the index of the closest value
            closest_idx = diffs.idxmin()
            # Append the closest time value
            x_times.append(float(patient_df_clean.loc[closest_idx, 'TIME']))
        x_values = patient_df_clean[:10][patient_df_clean['TIME'][:10].isin(x_times)]['OUT'].astype(float).tolist()
        # if len(y_values) < 10 or len(y_values) > 10:
        #     print(patient_id)
        if x_values and len(x_times) == len(target_times) and len(y_values) ==10 and len(y_times) == 10 and y_times[-1] < 30 and x_times[-1] < 5:
                y_true_times = y_times
                # y_times = [0., 0.33, 0.66, 1., 2., 3., 4.0, 6.00, 8.0, 12.0]
                # y_times = [0, 0.25, 0.67, 1., 1.5, 2., 3., 4., 6., 9., 12.]
                patient_be = data_be[data_be['Nom'] == patient_id]
                patient_auc = df_auc[df_auc['ID'] == patient_id]
                if patient_be.empty:
                    print(patient_id, "NO BE for this patient")
                    continue
                doses = patient_df['DOSE'].astype(float).tolist()[0]
                
                # doses = torch.tensor([doses, cort, crea])
                if (patient_df['DRUG']== 'PRO').all():
                    traitement = 1 # 0: Ciclo, 1: Tacro, 2: Siro
                else: 
                    traitement = 0
                # traitement = 1
                # classe = 0 # 0: Rein, 1: Card, 2: Poumons classe = organ
                if (patient_df['CYP3A5'] == 'EXP').all():
                    classe = 1 # 0: Ciclo, 1: Tacro, 2: Siro
                else: 
                    classe = 0
                time = convert(patient_df['periode'].values[0])
                # static = [doses, traitement, classe, time]
                static = [doses, traitement, classe]
                if patient_be['AUC(0-12)'].values > 0:
                    auc_be = [patient_be['AUC(0-12)'].values/max_out,patient_auc['auc_ipred'].values/(1000*max_out)]
                else:
                    auc_be = [patient_be['AUC(0-24)'].values/max_out,patient_auc['auc_ipred'].values/(1000*max_out)]
                # print(patient_id, patient_auc['AUC_ii'].values, auc*max_out*1000) 
                # auc = patient_auc['AUC_ii'].values/(1000*max_out)
                PK_params = patient_be['CL'].values
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
                    'auc_be': torch.tensor(auc_be),
                    'auc_red' : torch.tensor(auc)}
                # aucs.append(((auc - torch.tensor(patient_be['AUC(0-12)'].values))/auc)**2)
        # else:
        #     import pdb; pdb.set_trace()
    return data_dict, [max_out, best_lambda]

def extract_pccp_tac_2(file_path = "/Users/benjaminmaurel/Documents/Data_LG/FileSenderDownload_7-2-2025__9-5-56/validation_modele_vs_trapeze/pccp/pccp_tac.csv", plot = True):
    data_dict = {}
    df = pd.read_csv(file_path, sep=";", skipinitialspace=True,  skiprows=1)
    df = df.applymap(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x)
    # df = df.apply(pd.to_numeric, errors='coerce')
    df.columns = df.columns.str.strip()
    # Combine ID and group to make a unique new ID
    df['ID_new'] = df['#ID'] + df['visit'].astype(str)
    df['OUT'] = pd.to_numeric(df['OUT'].replace('.', None), errors='coerce')
    df['DOSE'] = pd.to_numeric(df['DOSE'].replace('.', None), errors='coerce')
    df['TIME'] = pd.to_numeric(df['TIME'].replace('.', None), errors='coerce')/60
    ids_to_remove = df.loc[df['OUT'] > 50000, 'ID_new'].unique()
    # Step 2: Filter out rows where ID_new is in that list
    df= df[~df['ID_new'].isin(ids_to_remove)]
    
    # plt.hist(df_filtered['DV'].dropna(), bins=30, edgecolor='black')
    # plt.xlabel('DV')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of DV')
    # plt.show()
    
    df['DOSE'] = df['DOSE'] / df['DOSE'].max()
    # df['DOSE'] = df['DOSE'] / 1000
    df['OUT'] = df['OUT']/1000
    max_out = df['OUT'].max()
    df['OUT'] = df['OUT'] / max_out
    
    # Optional: Drop intermediate columns
    unique_patient = df['ID_new'].unique()
    # Exclude 
    aucs = []
    data_be = pd.DataFrame(extract_pharmac_data_from_file('/Users/benjaminmaurel/Documents/Data_LG/FileSenderDownload_7-2-2025__9-5-56/validation_modele_vs_trapeze/pccp/pharmac_pccp_jd.txt'))
    for patient_id in unique_patient:
        if patient_id in ['LM15M6', 'RE02M6']:
            continue
        patient_df = df[df['ID_new'] == patient_id]
        patient_df_clean = patient_df[patient_df['OUT'].notna()]
        # patient_df_clean['TIME'] = patient_df_clean['TIME'] - patient_df_clean['TIME'].tolist()[0]
        try:
            patient_df_clean.loc[:, 'TIME'] = patient_df_clean['TIME'] - patient_df_clean['TIME'].iloc[0]
        except:
            import pdb; pdb.set_trace()
        y_times = patient_df_clean['TIME'].tolist()
        y_values = patient_df_clean['OUT'].astype(float).tolist()
        target_times = [0., 1., 3.]
        x_times = []
        for target in target_times:
            # Find the absolute difference between each time and the target
            diffs = np.abs(patient_df_clean['TIME'] - target)[:11]
            # Get the index of the closest value
            closest_idx = diffs.idxmin()
            # Append the closest time value
            x_times.append(float(patient_df_clean.loc[closest_idx, 'TIME']))
        x_values = patient_df_clean[:10][patient_df_clean['TIME'][:10].isin(x_times)]['OUT'].astype(float).tolist()
        
        if x_values and len(x_times) == len(target_times) and len(y_times) >= 10 and y_times[-1] < 30 and x_times[-1] < 5:
                y_true_times = y_times
                # y_times = [ 0., 0.33430636, 0.668, 1.004, 1.498, 2.006, 2.995, 4.012, 5.997, 9.007, 11.858]
                # y_times = [0, 0.25, 0.67, 1., 1.5, 2., 3., 4., 6., 9., 12.]
                patient_df_clean = patient_df[patient_df['DOSE'].notna()]
                if len(y_values) == 10:
                    y_true_times.append(12.)
                    y_values.append(min(y_values[0], y_values[-1]))
                auc = np.trapz(y_values, y_times)
                # if auc > 2.5:
                #     print(patient_id)
                #     continue
                aucs.append(auc)
                patient_be = data_be[data_be['Nom'] == patient_id]
                if patient_be.empty:
                    print(patient_id, "NO BE for this patient")
                    continue
                doses = patient_df['DOSE'].astype(float).tolist()[0]
                # doses = torch.tensor([doses, cort, crea])
                traitement = 1 # 0: Ciclo, 1: Tacro, 2: Siro
                classe = 0 # 0: Rein, 1: Card, 2: Poumons classe = organ
                try:
                    time = convert(patient_df['visit'].values[0])
                except:
                    import pdb; pdb.set_trace()
                # static = [doses, traitement, classe, time]
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
                    'dataset_number': torch.tensor(int(1.)),
                    'auc_be': torch.tensor(patient_be['AUC(0-12)'].values/max_out),
                    'auc_red' : torch.tensor([0.0])}
                
    if plot:
        # Plot OUT vs TIME for each subject
        plt.figure(figsize=(10, 6))
        plt.hist(aucs, bins = 20)
        # for subject_id, group in data_dict.items():
        #     if group['values_val'].max() > 0.1:
        #         print(subject_id)
        #         plt.plot(group['times_val'], group['values_val'], marker='o', label=f'ID {(subject_id)}')
        
        # plt.title('PK Data: OUT vs TIME by Subject')
        # plt.xlabel('Time')
        # plt.ylabel('Concentration (OUT)')
        # plt.legend(title='Subject ID', bbox_to_anchor=(1.05, 1), loc='upper left')
        # plt.tight_layout()
        # plt.grid(True)
        plt.show()

    return data_dict, max_out

def extract_pccp_tac(file_path = "/Users/benjaminmaurel/Downloads/mr4_tls_pccp.csv", plot = True):
    data_dict = {}
    df = pd.read_csv(file_path, sep=";", skipinitialspace=True)
    df = df.applymap(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x)
    # df = df.apply(pd.to_numeric, errors='coerce')
    df.columns = df.columns.str.strip()
    # Combine ID and group to make a unique new ID
    df['ID_2'] = df['ID'].astype(str) + df['PERI'].astype(str)
    df['ID_new'] = df['ID'].astype(str) + df['PERI'].astype(str) + df['ET'].astype(str)
    df['OUT'] = pd.to_numeric(df['DV'].replace('.', None), errors='coerce')
    df['OUT2'] = pd.to_numeric(df['DV'].replace('.', None), errors='coerce')
    df['DOSE'] = pd.to_numeric(df['AMT'].replace('.', None), errors='coerce')
    df['TIME'] = pd.to_numeric(df['TIME'].replace('.', None), errors='coerce')
    ids_to_remove = df.loc[df['OUT'] > 50000, 'ID_new'].unique()
    # Step 2: Filter out rows where ID_new is in that list
    df= df[~df['ID_new'].isin(ids_to_remove)]
    
    
    df['DOSE'] = df['DOSE'] / df['DOSE'].max()
    # df['DOSE'] = df['DOSE'] / 1000
    max_out = df['OUT'].max()
    df['OUT'] = df['OUT'] / max_out
    df['OUT2'] = df['OUT2'] / max_out
    boxcox_transformed_data, best_lambda = stats.boxcox(df[df['OUT'] != 0]['OUT']) # Add a tiny constant to ensure all data is > 0
    mask = (df['OUT'] > 0) & (df['OUT'].notna())
    df.loc[mask, 'OUT'] = boxcox_transformed_data
    # plt.hist(boxcox_transformed_data, bins=30, edgecolor='black')
    # plt.xlabel('DV')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of DV')
    # plt.savefig("pccp.png")
    
    # Optional: Drop intermediate columns
    unique_patient = df['ID_new'].unique()
    list_wrong_id = ['1151', '1441', '1551', '1631']
    # Exclude 
    aucs = []
    data_be = pd.DataFrame(extract_pharmac_data_from_file('/Users/benjaminmaurel/Documents/Data_LG/FileSenderDownload_7-2-2025__9-5-56/validation_modele_vs_trapeze/pccp/pharmac_pccp_jd.txt'))
    for patient_id in unique_patient:
        patient_df = df[df['ID_new'] == patient_id]
        patient_df_clean = patient_df[patient_df['OUT'].notna()]
        
        # patient_df_clean['TIME'] = patient_df_clean['TIME'] - patient_df_clean['TIME'].tolist()[0]
        try:
            patient_df_clean.loc[:, 'TIME'] = patient_df_clean['TIME'] - patient_df_clean['TIME'].iloc[1]
        except:
            import pdb; pdb.set_trace()
        
        y_times = patient_df_clean['TIME'].tolist()[1:]
        y_values2 = patient_df_clean['OUT2'].astype(float).tolist()[1:]
        y_values = patient_df_clean['OUT'].astype(float).tolist()[1:]
        crea = np.log(float(patient_df['CREA'].values[0]))
        hb = float(patient_df['HB'].values[0])
        ht = float(patient_df['HT'].values[0])
        target_times = [0., 1., 3.]
        x_times = []
        for target in target_times:
            # Find the absolute difference between each time and the target
            diffs = np.abs(patient_df_clean['TIME'] - target)[:11]
            # Get the index of the closest value
            closest_idx = diffs.idxmin()
            # Append the closest time value
            x_times.append(float(patient_df_clean.loc[closest_idx, 'TIME']))
        x_values = patient_df_clean[1:11][patient_df_clean['TIME'][1:11].isin(x_times)]['OUT'].astype(float).tolist()[:3]
        # import pdb; pdb.set_trace()
        if y_times[-2]<0:
            import pdb; pdb.set_trace()
        if y_times[-1]<=0:
            y_times = y_times[:-1]
            y_values = y_values[:-1]
        if x_values and len(x_times) == len(target_times) and len(y_times) >= 10 and y_times[-1] < 30 and x_times[-1] < 5:
                if not all(a < b for a, b in zip(y_times, y_times[1:])):
                    import pdb; pdb.set_trace()
                y_true_times = y_times.copy()
                # y_times = [ 0., 0.33430636, 0.668, 1.004, 1.498, 2.006, 2.995, 4.012, 5.997, 9.007, 11.858]
                # y_times = [0, 0.25, 0.67, 1., 1.5, 2., 3., 4., 6., 9., 12.]
                patient_df_clean = patient_df[patient_df['DOSE'].notna()]
                if len(y_values) == 10:
                    if y_true_times[-1]>=11:
                        y_true_times.insert(-1, 9.)
                        y_values.insert(-1,min(y_values[0], y_values[-1]))
                        y_values2.insert(-1,min(y_values2[0], y_values2[-1]))  
                        y_times.insert(-1, 9.) 
                    else:
                        y_true_times.append(12.)
                        y_values.append(min(y_values[0], y_values[-1]))
                        y_values2.append(min(y_values2[0], y_values2[-1]))
                        y_times.append(12.)  
                if len(y_values) == 11:
                    if y_true_times[-1]>=23:
                        y_true_times.insert(-1, 12.)
                        y_values.insert(-1,min(y_values[0], y_values[-1])) 
                        y_values2.insert(-1,min(y_values2[0], y_values2[-1]))  
                        y_times.insert(-1, 12.) 
                    else:
                        y_true_times.append(24.)
                        y_values.append(min(y_values[0], y_values[-1]))
                        y_values2.append(min(y_values2[0], y_values2[-1]))
                        y_times.append(24.)
                # if auc > 2.5:
                #     print(patient_id)
                #     continue
                # aucs.append(auc)
                patient_be = data_be[data_be['Nom'] == patient_id]
                doses = patient_df['DOSE'].astype(float).tolist()[0]
                # ht = patient_df['HT'].astype(float).tolist()[0]
                # doses = torch.tensor([doses, cort, crea])
                if (patient_df['CYP'] == '1').all():
                    classe = 1 # 0: Rein, 1: Card, 2: Poumons classe = organ
                else: 
                    classe = 0
                if patient_df['II'].values[0] == 24:
                    traitement = 0 # 0: ADV
                    try:
                        auc = auc_linuplogdown(np.array(y_values2), y_times)
                    except:
                        import pdb; pdb.set_trace()
                else:
                    traitement = 1 # 1: PRO
                    auc = auc_linuplogdown(np.array(y_values2[:11]), y_times[:11])
                # try:
                #     time = convert(patient_df['visit'].values[0])
                # except:
                #     import pdb; pdb.set_trace(  )
                # static = [doses, traitement, classe, time]
                auc_be = patient_be['AUC(0-12)'].values/max_out
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
                    'auc_be': torch.tensor([auc_be]),
                    'auc_red' : torch.tensor([auc])}
                
    if plot:
        # Plot OUT vs TIME for each subject
        plt.figure(figsize=(10, 6))
        plt.hist(aucs, bins = 20)
        # for subject_id, group in data_dict.items():
        #     if group['values_val'].max() > 0.1:
        #         print(subject_id)
        #         plt.plot(group['times_val'], group['values_val'], marker='o', label=f'ID {(subject_id)}')
        
        # plt.title('PK Data: OUT vs TIME by Subject')
        # plt.xlabel('Time')
        # plt.ylabel('Concentration (OUT)')
        # plt.legend(title='Subject ID', bbox_to_anchor=(1.05, 1), loc='upper left')
        # plt.tight_layout()
        # plt.grid(True)
        plt.show()

    return data_dict, [max_out, best_lambda]

def extract_gen_tac(train_data_path, test_data_path=None, plot=False):
    """
    Extracts and processes Tacrolimus data from given CSV files.

    Args:
        train_data_path (str): The full path to the training data CSV file.
        test_data_path (str, optional): The full path to the test data CSV file.
        plot (bool): If True, generates and displays a plot of the data.

    Returns:
        tuple: A tuple containing the processed data dictionary and a list [max_out, best_lambda].
    """
    data_dict = {}

    # Load training data
    df1 = pd.read_csv(train_data_path, sep=",")

    # Load test data if provided
    if test_data_path and os.path.exists(test_data_path):
        df2 = pd.read_csv(test_data_path, sep=",")
        df = pd.concat([df1, df2])
    else:
        df = df1

    # df = df.applymap(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x)
    # df = df.apply(pd.to_numeric, errors='coerce')
    df.columns = df.columns.str.strip()
    # Combine ID and group to make a unique new ID
    # df['ID_2'] = df['ID'].astype(str) + df['PERI'].astype(str)
    # df['ID_new'] = df['ID'].astype(str) + df['PERI'].astype(str)
    df['ID_2'] = df['ID'].astype(str)
    df['ID_new'] = df['ID'].astype(str)
    df['OUT'] = pd.to_numeric(df['DV'].replace('.', None), errors='coerce')
    df['DOSE'] = pd.to_numeric(df['AMT'].replace('.', None), errors='coerce')
    df['TIME'] = pd.to_numeric(df['TIME'].replace('.', None), errors='coerce')
    ids_to_remove = df.loc[df['OUT'] > 50000, 'ID_new'].unique()
    # Step 2: Filter out rows where ID_new is in that list
    df= df[~df['ID_new'].isin(ids_to_remove)]
    
    # plt.hist(df_filtered['DV'].dropna(), bins=30, edgecolor='black')
    # plt.xlabel('DV')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of DV')
    # plt.show()
    
    df['DOSE'] = df['DOSE'] / df['DOSE'].max()
    # df['DOSE'] = df['DOSE'] / 1000
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
        # filename = f"exp_run_all/{exp}/tacro_mapbayest_auc_{date_str}.csv"
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
        # patient_df_clean['TIME'] = patient_df_clean['TIME'] - patient_df_clean['TIME'].tolist()[0]
        try:
            patient_df_clean.loc[:, 'TIME'] = patient_df_clean['TIME'] - patient_df_clean['TIME'].iloc[0]
        except:
            import pdb; pdb.set_trace()
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
            try:
                x_times.append(float(patient_df_clean.loc[closest_idx, 'TIME']))
            except:
                import pdb; pdb.set_trace()
        x_values = patient_df_clean[patient_df_clean['TIME'].isin(x_times)]['OUT'].astype(float).tolist()[:3]
        # import pdb; pdb.set_trace()
        if y_times[-2]<0:
            import pdb; pdb.set_trace()
        if x_values and len(x_times) == len(target_times):
                if not all(a < b for a, b in zip(y_times, y_times[1:])):
                    import pdb; pdb.set_trace()
                y_true_times = y_times
                patient_df_clean = patient_df[patient_df['DOSE'].notna()]
                patient_be = data_be[data_be['ID'] == int(patient_id)]
                doses = patient_df['DOSE'].astype(float).tolist()[0]
                # doses = torch.tensor([doses, cort, crea])
                try:
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
                
    if plot:
        # Plot OUT vs TIME for each subject
        plt.figure(figsize=(10, 6))
        plt.hist(aucs, bins = 20)
        # for subject_id, group in data_dict.items():
        #     if group['values_val'].max() > 0.1:
        #         print(subject_id)
        #         plt.plot(group['times_val'], group['values_val'], marker='o', label=f'ID {(subject_id)}')
        
        # plt.title('PK Data: OUT vs TIME by Subject')
        # plt.xlabel('Time')
        # plt.ylabel('Concentration (OUT)')
        # plt.legend(title='Subject ID', bbox_to_anchor=(1.05, 1), loc='upper left')
        # plt.tight_layout()
        # plt.grid(True)
        plt.show()

    return data_dict, [max_out, best_lambda]

import re 
import chardet

def extract_stimmugrep(file_path = "/Users/benjaminmaurel/Documents/Data_LG/MMF_data/stimmugrep_cinetiques MMF_080910.csv", plot = False):
    data_dict = {}
    # with open(file_path, 'rb') as f:
    #     rawdata = f.read(10000)  # Sample first 10 KB
    #     result = chardet.detect(rawdata)
    #     print(result['encoding'])
    df = pd.read_csv(file_path, encoding='ISO-8859-1', on_bad_lines='skip', sep=';')
    # df = df.applymap(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x)
    # df = df.apply(pd.to_numeric, errors='coerce')
    df['ID_new'] = df['CODE_DOSAGE']
    
    df.columns = df.columns.str.strip()
    # df = df.apply(pd.to_numeric, errors='coerce')
    df['DOSE'] = df['POSO_AMP_MATIN'] / df['POSO_AMP_MATIN'].max()

    # Combine ID and group to make a unique new ID
    aucs = []
    data_be = pd.DataFrame(extract_pharmac_data_from_file('/Users/benjaminmaurel/Documents/Data_LG/MMF_data/pharmac_3_pts_stimugrep.txt'))
    
    all_unique_patient = df['ID_new'].unique()

    # Exclude 'C03-P1'
    unique_patient = [patient_id for patient_id in all_unique_patient if patient_id != 'C03-P1']

    pattern_value = r'^T\d+_AMP$' 
    pattern_time = r'^T\d+$' 
    all_values = []
    for patient_id in unique_patient:
        patient_df = df[df['CODE_DOSAGE'] == patient_id]
        y_times = []
        y_values = []
        time_cols = [col for col in df.columns if re.match(pattern_time, col)]
            
        # Iterate through the identified time columns
        for time_col in time_cols:
            # 1. Construct the corresponding AMP column name
            amp_col = time_col + '_AMP'
            
            # 2. Check if this AMP column exists in the DataFrame
            if amp_col in patient_df.columns:
                # 3. Get the value from the AMP column
                amp_value = patient_df[amp_col].iloc[0]
                
                # 4. CRITICAL CHECK: Is the AMP value NOT missing?
                if pd.notna(amp_value):
                    # If the AMP value is valid, now get the corresponding time value
                    time_value = patient_df[time_col].iloc[0]
                    
                    # Also make sure the time value itself isn't missing
                    if pd.notna(time_value):
                        # print(f"  VALID pair found: {time_col} ({time_value}) and {amp_col} ({amp_value})")
                        # Append the processed values to the lists
                        y_times.append(float(time_value) / 60) # Convert minutes to hours
                        y_values.append(float(amp_value) / 1000)
                        all_values.append(float(amp_value) / 1000)
        
        if pd.notna(patient_df['POSO_TAC_VEILLE']).item():
            traitement = 1 # 0: Ciclo, 1: Tacro, 2: Siro
        elif pd.notna(patient_df['POSO_CsA_VEILLE']).item():
            traitement = 0
        target_times = [0.33, 1., 3.]
        x_times = []
        x_values = []
        patient_be = data_be[data_be['Nom'] == patient_id]
        if patient_be.empty:
                print(patient_id, "NO BE for this patient")
                continue
        for target in target_times:
            # Find the absolute difference between each time and the target
            diffs = np.abs(np.array(y_times) - target)[:10]
            
            # Get the index of the closest value
            closest_idx = np.argmin(diffs)
            # Append the closest time value
            x_times.append(y_times[closest_idx])
            x_values.append(y_values[closest_idx])
        assert len(y_values) == len(y_times)
        
        if y_times and x_values and len(x_times) == 3 and len(y_times) >= 10 and y_times[-1] < 30 and x_times[-1] < 5:
            
            y_true_times = y_times
            y_times = [0, 0.33, 0.67, 1., 1.5, 2., 3., 4., 6., 9., 12.]
            patient_df = patient_df[patient_df['DOSE'].notna()]
            doses = patient_df['DOSE'].astype(float).tolist()[0]
            # doses = torch.tensor([doses, cort, crea])
            # concept_ciclo pharmac.txt
            if len(y_values) == 10:
                y_true_times.append(12.)
                y_values.append(min(y_values[0], y_values[-1]))
            else:    
                y_values = y_values[:11]
                y_true_times = y_true_times[:11]
            # auc = np.trapz(y_values, y_times)
            # if auc > 2.5:
            #     print(patient_id)
            #     continue
            # aucs.append(auc)
            classe = 2 # 0: Rein, 1: Card, 2: Poumons. classe = organ
            
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
                'dataset_number': torch.tensor(int(5.)),
                'auc_be': torch.tensor(patient_be['AUC(0-12)'].values)}
            
    max_out = max(all_values)
    for patient_id in data_dict.keys():
        data_dict[patient_id]['x_values']/=max_out
        data_dict[patient_id]['values_val']/=max_out

    if plot:
        # Plot OUT vs TIME for each subject
        plt.figure(figsize=(10, 6))
        for subject_id, group in data_dict.items():
            if group['static'][1] == 0:
                # Assign 'red' color when group['static'][2] is 0
                plt.plot(group['times_val'], group['values_val'], marker='o', color='red', label=f'ID {subject_id} (Static 2 = 0)')
            else:
                # Assign 'blue' color when group['static'][2] is not 0
                plt.plot(group['times_val'], group['values_val'], marker='o', color='blue', label=f'ID {subject_id} (Static 2 != 0)')

        plt.title('PK Data: OUT vs TIME by Subject')
        plt.xlabel('Time')
        plt.ylabel('Concentration (OUT)')
        plt.legend(title='Subject ID', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.grid(True)
        plt.show()
    
    return data_dict, max_out
def extract_concept(file_path = "/Users/benjaminmaurel/Documents/Data_LG/MMF_data/concept_mmf.csv", plot = True):
    data_dict = {}
    df = pd.read_csv(file_path, sep=";", skipinitialspace=True, skiprows=1)
    # df = df.applymap(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x)
    # df = df.apply(pd.to_numeric, errors='coerce')
    df['TIME'] = pd.to_numeric(df['TIME'], errors='coerce')
    df['TIME'] /= 60
    df['ID_new'] = df['#ID'].astype(str) + "_" + df['visit'].astype(str)
    df['OUT'] = pd.to_numeric(df['OUT'].replace('.', None), errors='coerce')
    df.columns = df.columns.str.strip()
    df['DOSE'] = pd.to_numeric(df['DOSE'], errors='coerce')
    df['DOSE'] = df['DOSE'] / df['DOSE'].max()

    # Combine ID and group to make a unique new ID
    df['OUT'] = df['OUT'] / 1000
    max_out = df['OUT'].max()
    df['OUT'] = df['OUT'] / max_out
    test = 0
    aucs = []
    data_be_ciclo = pd.DataFrame(extract_pharmac_data_from_file('/Users/benjaminmaurel/Documents/Data_LG/MMF_data/concept_ciclo pharmac.txt'))
    data_be_siro = pd.DataFrame(extract_pharmac_data_from_file('/Users/benjaminmaurel/Documents/Data_LG/MMF_data/concept_siro_pharmac.txt'))
    all_treatments = {'csa': 0, 'Tacrolimus' : 1, 'srl': 2}
    unique_patient = df['ID_new'].unique()
    for patient_id in unique_patient:
        patient_df = df[df['ID_new'] == patient_id]
        patient_df_clean = patient_df[patient_df['OUT'].notna()]
        y_times = patient_df_clean['TIME'].tolist()
        y_values = patient_df_clean['OUT'].astype(float).tolist()
        target_times = [0.33, 1., 3.]
        x_times = []
        if not data_be_ciclo[data_be_ciclo['Nom'] == patient_id].empty:
            patient_be = data_be_ciclo[data_be_ciclo['Nom'] == patient_id]
        else:
            patient_be = data_be_siro[data_be_siro['Nom'] == patient_id]
        if patient_be.empty:
                    print(patient_id, "NO BE for this patient")
                    continue
        for target in target_times:
            # Find the absolute difference between each time and the target
            diffs = np.abs(patient_df_clean['TIME'] - target)[:10]
            # Get the index of the closest value
            closest_idx = diffs.idxmin()
            # Append the closest time value
            x_times.append(float(patient_df_clean.loc[closest_idx, 'TIME']))
        x_values = patient_df_clean[:11][patient_df_clean['TIME'][:11].isin(x_times)]['OUT'].astype(float).tolist()
        if y_times and x_values and len(x_times) == 3 and len(y_times) >= 10 and y_times[-1] < 30 and x_times[-1] < 5:
            y_true_times = y_times
            y_times = [0, 0.33, 0.67, 1., 1.5, 2., 3., 4., 6., 9., 12.]
            patient_df_clean = patient_df[patient_df['DOSE'].notna()]
            doses = patient_df['DOSE'].astype(float).tolist()[0]
            # doses = torch.tensor([doses, cort, crea])
            # concept_ciclo pharmac.txt
            if len(y_values) == 10:
                y_true_times.append(12.)
                y_values.append(min(y_values[0], y_values[-1]))
            auc = np.trapz(y_values, y_times)
            if auc > 2.5:
                print(patient_id)
                continue
            aucs.append(auc)
            classe = 0 # 0: Rein, 1: Card, 2: Poumons classe = organ
            traitement = all_treatments[patient_df['IS'].tolist()[0]] # 0: Ciclo, 1: Tacro, 2: Siro
            # traitement = 2
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
                'dataset_number': torch.tensor(int(2.)),
                'auc_be': torch.tensor(patient_be['AUC(0-12)'].values)}
    if plot:
        # Plot OUT vs TIME for each subject
        plt.figure(figsize=(10, 6))
        for subject_id, group in data_dict.items():
            if group['static'][1] == 0:
                # Assign 'red' color when group['static'][2] is 0
                plt.plot(group['times_val'], group['values_val'], marker='o', color='red', label=f'ID {subject_id} (Static 2 = 0)')
            else:
                # Assign 'blue' color when group['static'][2] is not 0
                plt.plot(group['times_val'], group['values_val'], marker='o', color='blue', label=f'ID {subject_id} (Static 2 != 0)')

        plt.title('PK Data: OUT vs TIME by Subject')
        plt.xlabel('Time')
        plt.ylabel('Concentration (OUT)')
        plt.legend(title='Subject ID', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.grid(True)
        plt.show()

    return data_dict, max_out

def extract_pharmac_data_from_file(file_path):
    """
    Extracts pharmacokinetic data from a text file.

    Args:
        file_path (str): The path to the text file containing the data.

    Returns:
        list of dict: A list of dictionaries, where each dictionary
                      represents a row of data with keys from the header.
                      Returns an empty list if the file is not found or empty.
    """
    try:
        with open(file_path, 'r') as f:
            data_string = f.read()
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return []
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return []

    lines = data_string.strip().split('\n')
    if not lines:
        return []

    # Combine the first two lines to form the complete header
    # and clean up the header names by stripping whitespace and removing empty strings
    full_header_line = lines[0].strip() + " " + lines[1].strip()
    header_elements = full_header_line.split()
    header = [h.strip() for h in header_elements if h.strip()]

    data = []
    # Process data lines starting from the third line (index 2)
    for line in lines[1:]:
        cleaned_line = line.strip()
        if not cleaned_line:
            continue

        parts = cleaned_line.split()
        if not parts:
            continue

        row_name = parts[0]
        values = parts[1:]

        row_dict = {"Nom": row_name}

        # Populate the dictionary with values, aligning with the header.
        # Ensure we don't go out of bounds if a line has fewer values than expected.
        for i in range(len(values)):
            if i + 1 < len(header): # +1 because 'Nom' is already handled
                try:
                    # Convert to float for numerical values, keep as string for 'Nom'
                    row_dict[header[i+1]] = float(values[i])
                except ValueError:
                    row_dict[header[i+1]] = values[i]  # Fallback if not a float
            else:
                break  # Stop if we run out of header keys for the current line

        data.append(row_dict)

    return data

def extract_pigrec(file_path = "/Users/benjaminmaurel/Documents/Data_LG/MMF_data/MMF_PIGREC.csv", plot = True):
    data_dict = {}
    df = pd.read_csv(file_path, sep=";", skipinitialspace=True, skiprows=1)
    # df = df.applymap(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x)
    # df = df.apply(pd.to_numeric, errors='coerce')
    df['TIME'] = pd.to_numeric(df['TIME'], errors='coerce')
    df['ID_new'] = df['ID'].astype(str)
    df['OUT'] = pd.to_numeric(df['OUT'].replace('.', None), errors='coerce')
    df.columns = df.columns.str.strip()
    df['DOSE'] = pd.to_numeric(df['DOSE'], errors='coerce')
    df['EVID'] = pd.to_numeric(df['EVID'], errors='coerce')
    df['DOSE'] = df['DOSE'] / df['DOSE'].max() 
    # df['DOSE'] = df['DOSE'] / 1000
    # Combine ID and group to make a unique new ID
    
    max_out = df['OUT'].max()
    df['OUT'] = df['OUT'] / max_out
    df['OUT'] = df['OUT']
    unique_patient = df['ID_new'].unique()
    aucs = []
    all_treatments = {'Ciclosporine': 0, 'Tacrolimus' : 1, 'Sirolimus': 2}
    data_be_ciclo = pd.DataFrame(extract_pharmac_data_from_file('/Users/benjaminmaurel/Documents/Data_LG/MMF_data/pigrec_ciclo_pharmac.txt'))
    data_be_tacro = pd.DataFrame(extract_pharmac_data_from_file('/Users/benjaminmaurel/Documents/Data_LG/MMF_data/pigrec_tacro_pharmac.txt'))
    for patient_id in unique_patient:
        patient_df = df[df['ID_new'] == patient_id]
        if not data_be_ciclo[data_be_ciclo['Nom'] == patient_id].empty:
            patient_be = data_be_ciclo[data_be_ciclo['Nom'] == patient_id]
        else:
            patient_be = data_be_tacro[data_be_tacro['Nom'] == patient_id]
        if patient_be.empty:
                    print(patient_id, "NO BE for this patient")
                    continue
        if pd.isna(patient_df['Tt_associe'].tolist()[0]):
            try:
                patient_id_treat = patient_id[:-2] + 'P1'
                patient_df_treat = df[df['ID_new'] == patient_id_treat]
                traitement = all_treatments[patient_df_treat['Tt_associe'].tolist()[0]]
            except:
                try:
                    patient_id_treat = patient_id[:-2] + 'P3'
                    patient_df_treat = df[df['ID_new'] == patient_id_treat]
                    traitement = all_treatments[patient_df_treat['Tt_associe'].tolist()[0]]
                except:
                    patient_id_treat = patient_id[:-2] + 'P2'
                    patient_df_treat = df[df['ID_new'] == patient_id_treat]
                    traitement = all_treatments[patient_df_treat['Tt_associe'].tolist()[0]]
        else:
            traitement = all_treatments[patient_df['Tt_associe'].tolist()[0]]  # 0: Ciclo, 1: Tacro, 2: Siro
        patient_df_clean = patient_df[patient_df['OUT'].notna()]
        y_times = patient_df_clean['TIME'].tolist()
        y_values = patient_df_clean['OUT'].astype(float).tolist()
        target_times = [0.33, 1., 3.]
        x_times = []
        for target in target_times:
            # Find the absolute difference between each time and the target
            diffs = np.abs(patient_df_clean['TIME'] - target)[:10]
            # Get the index of the closest value
            closest_idx = diffs.idxmin()
            # Append the closest time value
            x_times.append(float(patient_df_clean.loc[closest_idx, 'TIME']))
        x_values = patient_df_clean[:10][patient_df_clean['TIME'][:10].isin(x_times)]['OUT'].astype(float).tolist()
        if y_times and x_values and len(x_times) == 3 and len(y_times) >= 10 and y_times[-1] < 30 and x_times[-1] < 5:
                if len(x_values) == 3:
                    y_true_times = y_times
                    y_times = [0, 0.33, 0.67, 1., 1.5, 2., 3., 4., 6., 9., 12.]
                
                    patient_df_clean = patient_df[patient_df['DOSE'].notna()]   
                    if len(y_values) == 10:
                        y_true_times.append(12.)
                        y_values.append(min(y_values[0], y_values[-1]))
                    auc = np.trapz(y_values, y_times)
                    if auc > 2.5:
                        print(patient_id)
                        continue
                    aucs.append(auc)
                    doses = patient_df['DOSE'].astype(float).tolist()[0]
                    # doses = torch.tensor([doses, cort, crea])
                    classe = 1 # 0: Rein, 1: Card, 2: Poumons classe = organ
                    static = [doses, traitement, classe]
                    
                    data_dict[patient_id] = {
                        'times_val': torch.tensor(y_times),
                        'values_val': torch.tensor(y_values),
                        'y_true_times': torch.tensor(y_true_times),
                        'x_values': torch.tensor(x_values),
                        'x_times': torch.tensor(x_times),
                        'doses': torch.tensor(doses),
                        'patient_id': patient_id,
                        'static': torch.tensor(static),
                        'dataset_number': torch.tensor(int(1.)),
                        'auc_be': torch.tensor(patient_be['AUC(0-12)'].values)}
                else:
                    import pdb; pdb.set_trace()
    if plot:
        # Plot OUT vs TIME for each subject
        plt.figure(figsize=(10, 6))
        plt.hist(aucs, bins = 20)
        # for subject_id, group in data_dict.items():
        #     plt.plot(group['times_val'], group['values_val'], marker='o', label=f'ID {(subject_id)}')

        # plt.title('PK Data: OUT vs TIME by Subject')
        # plt.xlabel('Time')
        # plt.ylabel('Concentration (OUT)')
        # plt.legend(title='Subject ID', bbox_to_anchor=(1.05, 1), loc='upper left')
        # plt.tight_layout()
        # plt.grid(True)
        plt.show()

    return data_dict, max_out

def extract_ped(file_path = "/Users/benjaminmaurel/Documents/Data_LG/MMF_data/cinétiques_spiesser_210820.csv", plot = True):
    data_dict = {}
    df = pd.read_csv(file_path, sep=",", skipinitialspace=True)
    # df = df.applymap(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x)
    # df = df.apply(pd.to_numeric, errors='coerce')
    df.columns = df.columns.str.strip()
    
    print(df.columns)
    df['TIME'] = pd.to_numeric(df['TIME'], errors='coerce')/60
    df['ID_new'] = df['id'].astype(str) + "_" + df['peri'].astype(str)
    df['MPA'] = pd.to_numeric(df['MPA'].replace('.', None), errors='coerce')
    
    df['DOSE'] = pd.to_numeric(df['Dose'], errors='coerce')
    df['DOSE'] = df['DOSE'] / df['DOSE'].max()

    # Combine ID and group to make a unique new ID
    
    max_out = df['MPA'].max()
    df['MPA'] = df['MPA'] / max_out
    test = 0
    unique_patient = df['ID_new'].unique()
    all_treatments = {'Ciclosporine': 0, 'Tacrolimus' : 1, 'Sirolimus': 2, 'other': 3}
    data_be_ciclo = pd.DataFrame(extract_pharmac_data_from_file('/Users/benjaminmaurel/Documents/Data_LG/MMF_data/spiesser_ciclo_pharmac.txt'))
    data_be_siro = pd.DataFrame(extract_pharmac_data_from_file('/Users/benjaminmaurel/Documents/Data_LG/MMF_data/spiesser_siro_pharmac.txt'))
    for patient_id in unique_patient:
        patient_df = df[df['ID_new'] == patient_id]
        
        if not data_be_ciclo[data_be_ciclo['Nom'] == patient_id].empty:
            patient_be = data_be_ciclo[data_be_ciclo['Nom'] == patient_id]
        else:
            patient_be = data_be_siro[data_be_siro['Nom'] == patient_id]
        if patient_be.empty:
                    print(patient_id, "NO BE for this patient")
                    continue
        patient_df_clean = patient_df[patient_df['MPA'].notna()]
        y_times = patient_df_clean['TIME'].tolist()
        y_values = patient_df_clean['MPA'].astype(float).tolist()
        target_times = [0.33, 1., 3.]
        x_times = []
        aucs = []
        for target in target_times:
            # Find the absolute difference between each time and the target
            diffs = np.abs(patient_df_clean['TIME'] - target)[:11]
            # Get the index of the closest value
            closest_idx = diffs.idxmin()
            # Append the closest time value
            x_times.append(float(patient_df_clean.loc[closest_idx, 'TIME']))
        x_values = patient_df_clean[:11][patient_df_clean['TIME'][:11].isin(x_times)]['MPA'].astype(float).tolist()
        if y_times and x_values and len(x_times) == 3 and len(y_times) >= 10 and y_times[-1] < 30 and x_times[-1] < 5:
                if len(x_values) == 3:
                    
                    y_true_times = y_times
                    y_times = [0, 0.33, 0.67, 1., 1.5, 2., 3., 4., 6., 9., 12.]
                
                    patient_df_clean = patient_df[patient_df['DOSE'].notna()]   
                    if len(y_values) == 10:
                        y_true_times.append(12.)
                        y_values.append(min(y_values[0], y_values[-1]))
                    auc = np.trapz(y_values, y_times)
                    if auc > 2.5:
                        print(patient_id)
                        continue
                    aucs.append(auc)
                    doses = patient_df['DOSE'].astype(float).tolist()[0]
                    traitement = all_treatments[patient_df['drug'].tolist()[0]]
                    # doses = torch.tensor([doses, cort, crea])
                    classe = 0 # 0: Rein, 1: Card, 2: Poumons classe = organ
                    static = [doses, traitement, classe]
                    
                    data_dict[patient_id] = {
                        'times_val': torch.tensor(y_times),
                        'values_val': torch.tensor(y_values),
                        'y_true_times': torch.tensor(y_true_times),
                        'x_values': torch.tensor(x_values),
                        'x_times': torch.tensor(x_times),
                        'doses': torch.tensor(doses),
                        'patient_id': patient_id,
                        'static': torch.tensor(static),
                        'dataset_number': torch.tensor(int(4.)),
                        'auc_be': torch.tensor(patient_be['AUC(0-12)'].values)}
                else:
                    import pdb; pdb.set_trace()
    if plot:
        # Plot OUT vs TIME for each subject
        plt.figure(figsize=(10, 6))
        for subject_id, group in data_dict.items():
            plt.plot(group['times_val'], group['values_val'], marker='o', label=f'ID {(subject_id)}')

        plt.title('PK Data: OUT vs TIME by Subject')
        plt.xlabel('Time')
        plt.ylabel('Concentration (OUT)')
        plt.legend(title='Subject ID', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.grid(True)
        plt.show()

    return data_dict, max_out

def extract_stablocine(file_path = "/Users/benjaminmaurel/Documents/Data_LG/MMF_data/MMF_LC_MS_stablocine.csv", plot = True):
    data_dict = {}
    df = pd.read_csv(file_path, sep=";", skipinitialspace=True, skiprows=1)
    # df = df.applymap(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x)
    # df = df.apply(pd.to_numeric, errors='coerce')
    df['TIME'] = df['TIME_H'].astype(str)
    df['TIME'] = pd.to_numeric(df['TIME'], errors='coerce')
    df['ID_new'] = df['#ID'].astype(str)
    df['OUT'] = pd.to_numeric(df['OUT'].replace('.', None), errors='coerce')
    df.columns = df.columns.str.strip()
    
    df['DOSE'] = df['DOSE'] / df['DOSE'].max()

    # Combine ID and group to make a unique new ID
    aucs = []
    max_out = df['OUT'].max()
    df['OUT'] = df['OUT'] / max_out
    test = 0
    unique_patient = df['ID_new'].unique()
    data_be = pd.DataFrame(extract_pharmac_data_from_file('/Users/benjaminmaurel/Documents/Data_LG/MMF_data/stablo_lcms_pharmac.txt'))
    for patient_id in unique_patient:
        patient_df = df[df['ID_new'] == patient_id]
        patient_df_clean = patient_df[patient_df['OUT'].notna()]
        y_times = patient_df_clean['TIME'].tolist()
        y_values = patient_df_clean['OUT'].astype(float).tolist()
        target_times = [0.33, 1., 3.]
        x_times = []
        for target in target_times:
            # Find the absolute difference between each time and the target
            diffs = np.abs(patient_df_clean['TIME'] - target)[:10]
            # Get the index of the closest value
            closest_idx = diffs.idxmin()
            # Append the closest time value
            x_times.append(float(patient_df_clean.loc[closest_idx, 'TIME']))
        x_values = patient_df_clean[:10][patient_df_clean['TIME'][:10].isin(x_times)]['OUT'].astype(float).tolist()
        if y_times and x_values and len(x_times) == 3 and len(y_times) >= 10 and y_times[-1] < 30 and x_times[-1] < 5:
                y_true_times = y_times
                y_times = [0, 0.33, 0.67, 1., 1.5, 2., 3., 4., 6., 9., 12.]
                patient_df_clean = patient_df[patient_df['DOSE'].notna()]
                if len(y_values) == 10:
                    y_true_times.append(12.)
                    y_values.append(min(y_values[0], y_values[-1]))
                auc = np.trapz(y_values, y_times)
                if auc > 2.5:
                    print(patient_id)
                    continue
                aucs.append(auc)
                patient_be = data_be[data_be['Nom'] == patient_id]
                if patient_be.empty:
                    print(patient_id, "NO BE for this patient")
                    continue
                doses = patient_df['DOSE'].astype(float).tolist()[0]

                # doses = torch.tensor([doses, cort, crea])
                traitement = 1 # 0: Ciclo, 1: Tacro, 2: Siro
                classe = 0  # 0: Rein, 1: Card, 2: Poumons classe = organ
                static = [doses, traitement, classe]
                data_dict[patient_id] = {
                    'times_val': torch.tensor(y_times),
                    'values_val': torch.tensor(y_values),
                    'y_true_times': torch.tensor(y_true_times),
                    'x_values': torch.tensor(x_values),
                    'x_times': torch.tensor(x_times),
                    'doses': torch.tensor(doses),
                    'static': torch.tensor(static),
                    'dataset_number': torch.tensor(int(3.)),
                    'patient_id': patient_id,
                    'auc_be': torch.tensor(patient_be['AUC(0-12)'].values)}
    if plot:
        # Plot OUT vs TIME for each subject
        plt.figure(figsize=(10, 6))
        for subject_id, group in data_dict.items():
            plt.plot(group['times_val'], group['values_val'], marker='o', label=f'ID {(subject_id)}')

        plt.title('PK Data: OUT vs TIME by Subject')
        plt.xlabel('Time')
        plt.ylabel('Concentration (OUT)')
        plt.legend(title='Subject ID', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.grid(True)
        plt.show()

    return data_dict, max_out

def extract_theo(file_path = "/Users/benjaminmaurel/Downloads/Theoph.csv", plot = False):
    data_dict = {}
    df = pd.read_csv(file_path, sep=",", skipinitialspace=True)
    # df = df.apply(pd.to_numeric, errors='coerce')
    df['TIME'] = df['Time'].astype(str)
    df['TIME'] = pd.to_numeric(df['TIME'], errors='coerce')
    df['ID_new'] = df['Subject'].astype(str)
    df['OUT'] = pd.to_numeric(df['conc'].replace('.', None), errors='coerce')
    df['DOSE'] = pd.to_numeric(df['Dose'].replace('.', None), errors='coerce')
    df.columns = df.columns.str.strip()
    
    df['DOSE'] = df['DOSE'] / df['DOSE'].max()

    # Combine ID and group to make a unique new ID
    aucs = []
    max_out = df['OUT'].max()
    df['OUT'] = df['OUT'] / max_out
    test = 0
    unique_patient = df['ID_new'].unique()
    
    # data_be = pd.DataFrame(extract_pharmac_data_from_file('/Users/benjaminmaurel/Documents/Data_LG/MMF_data/stablo_lcms_pharmac.txt'))
    for patient_id in unique_patient:
        patient_df = df[df['ID_new'] == patient_id]
        patient_df_clean = patient_df[patient_df['OUT'].notna()]
        y_times = patient_df_clean['TIME'].tolist()
        y_values = patient_df_clean['OUT'].astype(float).tolist()
        target_times = [0.0, 0.25, 0.55]
        x_times = []
        used_indices = set()
        for target in target_times:
            # Consider only the first 10 rows
            candidate_times = patient_df_clean['TIME'][:10]
            # Compute absolute differences
            diffs = np.abs(candidate_times - target)
            
            # Sort indices by closeness to the target
            sorted_indices = diffs.sort_values().index
            
            # Find the first index not yet used
            for idx in sorted_indices:
                time_val = float(patient_df_clean.loc[idx, 'TIME'])
                if time_val not in x_times:
                    x_times.append(time_val)
                    used_indices.add(idx)
                    break
        
        x_values = patient_df_clean[:10][patient_df_clean['TIME'][:10].isin(x_times)]['OUT'].astype(float).tolist()
        if y_times and len(x_values)==3 and len(x_times) == 3 and len(y_times) >= 10 and y_times[-1] < 30 and x_times[-1] < 5:
                y_true_times = y_times
                # y_times = [0, 0., 0.67, 1., 1.5, 2., 3., 4., 6., 9., 12.]
                patient_df_clean = patient_df[patient_df['DOSE'].notna()]
                if len(y_values) == 10:
                    y_true_times.append(12.)
                    y_values.append(min(y_values[0], y_values[-1]))
                doses = patient_df['DOSE'].astype(float).tolist()[0]

                # doses = torch.tensor([doses, cort, crea])
                traitement = 1 # 0: Ciclo, 1: Tacro, 2: Siro
                classe = 0  # 0: Rein, 1: Card, 2: Poumons classe = organ
                static = [doses, traitement, classe]
                data_dict[patient_id] = {
                        'times_val': torch.tensor(y_times),
                        'values_val': torch.tensor(y_values),
                        'y_true_times': torch.tensor(y_true_times),
                        'x_values': torch.tensor(x_values),
                        'x_times': torch.tensor(x_times),
                        'doses': torch.tensor(doses),
                        'patient_id': patient_id,
                        'static': torch.tensor(static),
                        'dataset_number': torch.tensor(int(6.)),
                        'auc_be': torch.tensor([])}
    if plot:
        # Plot OUT vs TIME for each subject
        plt.figure(figsize=(10, 6))
        for subject_id, group in data_dict.items():
            plt.plot(group['times_val'], group['values_val'], marker='o', label=f'ID {(subject_id)}')

        plt.title('PK Data: OUT vs TIME by Subject')
        plt.xlabel('Time')
        plt.ylabel('Concentration (OUT)')
        plt.legend(title='Subject ID', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.grid(True)
        plt.show()
    return data_dict, max_out

def extract_tls(file_path = '/Users/benjaminmaurel/Documents/Data_LG/FileSenderDownload_7-2-2025__9-5-56/validation_modele_vs_trapeze/mr4_tls/mr4_tls_pccp.csv', plot = False):
    data_dict = {}
    df = pd.read_csv(file_path, sep=";", skipinitialspace=True)
    # df = df.applymap(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x)
    df = df.apply(pd.to_numeric, errors='coerce')
    df.columns = df.columns.str.strip()
    # Create a new ID that increments when 'CORT' changes within each 'ID'
    df['CORT_shifted'] = df.groupby('ID')['CORT'].shift()
    df['CORT_changed'] = (df['CORT'] != df['CORT_shifted']).fillna(True)
    df['CORT_group'] = df.groupby('ID')['CORT_changed'].cumsum().astype(int)
    df['DOSE'] = df['DOSE'] / df['DOSE'].max()
    df['CREA'] = df['CREA'] / df['CREA'].max()
    df['CORT'] = df['CORT'] / df['CORT'].max()
    # Combine ID and group to make a unique new ID
    df['ID_new'] = df['ID'].astype(str) + "_" + df['CORT'].astype(str)
    # Optional: Drop intermediate columns
    df.drop(columns=['CORT_shifted', 'CORT_changed', 'CORT_group'], inplace=True)
    max_out = df['OUT'].max()
    df['OUT'] = df['OUT'] / max_out
    test = 0
    unique_patient = df['ID_new'].unique()
    for patient_id in unique_patient:
        patient_df = df[df['ID_new'] == patient_id]
        patient_df_clean = patient_df[patient_df['OUT'].notna() & patient_df['EVID'] == 0]
        y_times = patient_df_clean['TIME'].tolist()
        y_values = patient_df_clean['OUT'].astype(float).tolist()
        target_times = [0., 1., 3.]
        x_times = []
        for target in target_times:
            # Find the absolute difference between each time and the target
            diffs = np.abs(patient_df_clean['TIME'] - target)[:10]
            # Get the index of the closest value
            closest_idx = diffs.idxmin()
            # Append the closest time value
            x_times.append(float(patient_df_clean.loc[closest_idx, 'TIME']))
        x_values = patient_df_clean[:10][patient_df_clean['TIME'][:10].isin(x_times)]['OUT'].astype(float).tolist()
        if y_times and x_values and len(x_times) == 3 and len(y_times) >= 10 and y_times[-1] < 30 and x_times[-1] < 5:
                y_times = [0, 0.33, 0.67, 1., 1.5, 2., 3., 4., 6., 9.]
                patient_df_clean = patient_df[patient_df['DOSE'].notna()]
                doses = patient_df['DOSE'].astype(float).tolist()[0]
                age = patient_df['AGE'].astype(float).tolist()[0]
                cort = patient_df['CREA'].astype(float).tolist()[0]
                crea = patient_df['CORT'].astype(float).tolist()[0]
                
                # doses = torch.tensor([doses, cort, crea])
                data_dict[patient_id] = {
                    'times_val': torch.tensor(y_times[:10]),
                    'values_val': torch.tensor(y_values[:10]),
                    'x_values': torch.tensor(x_values),
                    'x_times': torch.tensor(x_times),
                    'doses': torch.tensor(doses)}
    if plot:
        # Plot OUT vs TIME for each subject
        plt.figure(figsize=(10, 6))
        for subject_id, group in data_dict.items():
            plt.plot(group['times_val'], group['values_val'], marker='o', label=f'ID {(subject_id)}')

        plt.title('PK Data: OUT vs TIME by Subject')
        plt.xlabel('Time')
        plt.ylabel('Concentration (OUT)')
        plt.legend(title='Subject ID', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.grid(True)
        plt.show()

    return data_dict, max_out

def extract_val_tacro(file_path = "/Users/benjaminmaurel/Documents/Data_LG/FileSenderDownload_7-2-2025__9-5-56/validation_modele_vs_trapeze/tacro_foie/Adv_dev_2.csv", plot = False):
    # Load and clean the data
    df = pd.read_csv(file_path, skiprows=1)
    df['TIME'] = pd.to_numeric(df['TIME'], errors='coerce')
    df['OUT'] = pd.to_numeric(df['OUT'].replace('.', None), errors='coerce')*10
    df['ID'] = pd.to_numeric(df['#ID'], errors='coerce')
    df = df.dropna(subset=['TIME', 'OUT', 'ID'])
    
    if plot:
        # Plot OUT vs TIME for each subject
        plt.figure(figsize=(10, 6))
        for subject_id, group in df.groupby('ID'):
            plt.plot(group['TIME'], group['OUT'], marker='o', label=f'ID {int(subject_id)}')

        plt.title('PK Data: OUT vs TIME by Subject')
        plt.xlabel('Time')
        plt.ylabel('Concentration (OUT)')
        plt.legend(title='Subject ID', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.grid(True)
        plt.show()

    return df

def extract_in_tacro(file_path, plot = True):
    df = pd.read_csv(file_path)
    df['TIME'] = pd.to_numeric(df['TIME'], errors='coerce')
    df['OUT'] = pd.to_numeric(df['OUT'].replace('.', None), errors='coerce')/100
    df['ID'] = pd.to_numeric(df['id'], errors='coerce')
    df['DOSE'] = pd.to_numeric(df['DOSE'], errors='coerce')
    df = df.dropna(subset=['TIME', 'OUT', 'ID'])

    if plot:
        # Plot OUT vs TIME for each subject
        plt.figure(figsize=(10, 6))
        for subject_id, group in df.groupby('ID'):
            plt.plot(group['TIME'], group['OUT'], marker='o', label=f'ID {int(subject_id)}')

        plt.title('PK Data: OUT vs TIME by Subject')
        plt.xlabel('Time')
        plt.ylabel('Concentration (OUT)')
        plt.legend(title='Subject ID', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.grid(True)
        plt.show()
    return df

def extract_in_val_mmf(file_path, plot = True):
    data_dict = {}
    df = pd.read_csv(file_path, sep = ";")
    df['TIME'] = pd.to_numeric(df['TIME'], errors='coerce')
    df['OUT'] = pd.to_numeric(df['OUT'].replace('.', None), errors='coerce')
    max_out = df['OUT'].max()
    df['OUT'] = df['OUT'] / max_out
    df['ID'] = pd.to_numeric(df['ID'], errors='coerce')
    df['DOSE'] = pd.to_numeric(df['DOSE'], errors='coerce')
    df['DOSE'] = df['DOSE'] / df['DOSE'].max()
    # df = df.dropna(subset=['TIME', 'OUT', 'ID'])
    unique_patient = df['ID'].unique()
    for patient_id in unique_patient:
        patient_df = df[df['ID'] == patient_id]
        patient_df_clean = patient_df[patient_df['OUT'].notna()]
        y_times = patient_df_clean['TIME'].tolist()
        y_values = patient_df_clean['OUT'].astype(float).tolist()
        x_values = patient_df_clean[patient_df['TIME'].isin([0., 1., 3.])]['OUT'].astype(float).tolist()
        x_times = patient_df_clean[patient_df['TIME'].isin([0., 1., 3.])]['TIME'].astype(float).tolist()
        if y_times and x_values and len(x_times) == 3 and len(y_times) >= 10:
                patient_df_clean = patient_df[patient_df['DOSE'].notna()]
                doses = patient_df['DOSE'].astype(float).tolist()[0]
                data_dict[patient_id] = {
                    'times_val': torch.tensor(y_times[:10]),
                    'values_val': torch.tensor(y_values[:10]),
                    'x_values': torch.tensor(x_values),
                    'x_times': torch.tensor(x_times),
                    'doses': torch.tensor(doses)}
    return data_dict, max_out

def create_dataset(file_val, file_in):
    df_val = extract_val_tacro(file_val)
    df_in = extract_in_tacro(file_in)
    data_dict = {}
    unique_patient = df_in['ID'].unique()
    len_x = 3
    len_y = 17
    for patient_id in unique_patient:
        patient_df_in = df_in[df_in['ID'] == patient_id]
        patient_df_val = df_val[df_val['ID'] == patient_id]
        y_times = patient_df_val['TIME'].tolist()
        y_values = patient_df_val['OUT'].astype(float).tolist()
        x_values = patient_df_in['OUT'].astype(float).tolist()
        x_times = patient_df_in['TIME'].astype(float).tolist()
        
        if y_times and x_values and len(x_times) == 3 and len(y_times) == 17 :
            doses = patient_df_in['DOSE'].astype(float).tolist()[0]
            data_dict[patient_id] = {
                'times_val': torch.tensor(y_times),
                'values_val': torch.tensor(y_values),
                'x_values': torch.tensor(x_values),
                'x_times': torch.tensor(x_times),
                'doses': torch.tensor(doses)
                }
    return data_dict


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
        df_auc = pd.read_csv('/Users/benjaminmaurel/tacro_mapbayest_auc_20250826.csv')
        list_patients_id = df_auc['ID'].tolist()
        
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

    # batch = torch.stack(batch)
    # try:
    obs = torch.stack([batch[i][3] for i in range(len(batch))]).unsqueeze(-1)
    # obs2 = (torch.stack([batch[i][2] for i in range(len(batch))]) - torch.tensor([0.0, 1.0, 3.0])).unsqueeze(-1)
    # obs = torch.cat((obs, obs2), -1)
    classes = torch.stack([batch[i][4] for i in range(len(batch))])
    # # tp_obs = batch[0][2]
    # try:
    #     assert len(obs[0]) == 3
    # except:
    #     import pdb; pdb.set_trace()
    tp_obs = torch.tensor([0.0, 1.0, 3.0])

    # tp_obs = torch.stack([batch[i][2] for i in range(len(batch))])
    # all_obs = torch.unique(torch.stack([batch[i][2] for i in range(len(batch))]))
    # obs_mask =(tp_obs.unsqueeze(-1) == all_obs).any(dim=1).int()
    # indices = torch.searchsorted(all_obs, tp_obs)
    # new_observations = torch.zeros(obs_mask.shape[0], obs_mask.shape[1], dtype=obs.dtype)
    # new_observations = new_observations.scatter_(dim=1, index=indices, src=obs.squeeze(-1)).unsqueeze(-1)


    # values_adapted = torch.zeros(tp_obs.shape[0], len(all_obs), dtype=obs.dtype)
    # values_adapted.scatter_(dim=1, index=indices, src=obs)
    
    
    # tp_obs = torch.tensor([1.0, 3.0])
    
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
        # dose = torch.stack([batch[i][-1] for i in range(len(batch))]).unsqueeze(1).expand(-1, len(all_obs)).unsqueeze(-1)
        # classes = torch.stack([batch[i][4] for i in range(len(batch))]).unsqueeze(1).expand(-1, 3).unsqueeze(-1)
        # dose = torch.stack([batch[i][-1] for i in range(len(batch))]).unsqueeze(-1)
        # static = torch.stack([batch[i][4] for i in range(len(batch))]).unsqueeze(-1) ##Unsqueeze only if static dim = 1
        static = torch.stack([batch[i][4] for i in range(len(batch))])
        static2 = (torch.stack([batch[i][2] for i in range(len(batch))]) - torch.tensor([0.0, 1.0, 3.0]))
        # static = torch.cat((static, static2), -1)
        # static_for_obs = static.unsqueeze(1).expand(-1,len(all_obs), static.shape[1])
        # new_observations = torch.cat((new_observations, static_for_obs), -1)
        # obs_mask = obs_mask.unsqueeze(-1).expand(-1, -1, static.shape[1] +1) # We add one for obs data
    else:
        dose = torch.stack([batch[i][-1] for i in range(len(batch))]).unsqueeze(1).expand(-1,3, -1)
    split_dict = {#"observed_data": values_adapted.unsqueeze(-1).clone(),
        "observed_data": obs.clone(),
        "observed_tp": tp_obs.clone(),
        # "observed_data": new_observations.clone(),
        # "observed_tp": all_obs.clone(),
        # "observed_mask": obs_mask.clone(),
        "data_to_predict": data_pred.unsqueeze(-1).clone(),
        "tp_to_predict": tp_pred.clone(),
        "dose": dose.clone(),
        # "dose" : None,
        "auc_be": auc_be.clone(),
        "auc_red": auc_red.clone(),
        "dataset_number": dataset.clone(),
        "y_true_times": y_true_times.clone(),
        "patient_id": patient_id,
        "static": static.clone(),
        "others": others.clone()}
    
    # split_dict["observed_mask"] = None 
    split_dict["mask_predicted_data"] = None 
    split_dict["labels"] = None 
    split_dict["mode"] = "interp"
    return split_dict
    # except:
    #     import pdb; pdb.set_trace()