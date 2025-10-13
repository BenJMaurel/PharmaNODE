import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy import stats
from scipy.special import inv_boxcox
import datetime
import glob
import os


def load_and_clean_dataframe(path, sep=';', skipinitialspace=True, skiprows=1):
    """
    Loads a CSV file into a pandas DataFrame and performs initial cleaning.

    Args:
        path (str): The path to the CSV file.
        sep (str): The separator for the CSV file.
        skipinitialspace (bool): Whether to skip spaces after the separator.
        skiprows (int): The number of rows to skip at the beginning of the file.

    Returns:
        pd.DataFrame: The cleaned pandas DataFrame.
    """
    df = pd.read_csv(path, sep=sep, skipinitialspace=skipinitialspace, skiprows=skiprows)
    df = df.applymap(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x)
    df = df.apply(pd.to_numeric, errors='coerce')
    df.columns = df.columns.str.strip()
    return df


def build_patient_dictionary(patient_df, patient_id, max_out, best_lambda):
    """
    Builds a dictionary containing patient data.

    Args:
        patient_df (pd.DataFrame): The DataFrame containing the patient's data.
        patient_id (str): The ID of the patient.
        max_out (float): The maximum output value for normalization.
        best_lambda (float): The best lambda value for the Box-Cox transformation.

    Returns:
        dict: A dictionary containing the patient's data.
    """
    patient_df_clean = patient_df[patient_df['OUT'].notna()]
    y_times = patient_df_clean['TIME'].tolist()
    y_values = patient_df_clean['OUT'].astype(float).tolist()
    target_times = [0., 1., 3.]
    x_times = []
    for target in target_times:
        diffs = np.abs(patient_df_clean['TIME'] - target)
        closest_idx = diffs.idxmin()
        x_times.append(float(patient_df_clean.loc[closest_idx, 'TIME']))

    x_values = patient_df_clean[patient_df_clean['TIME'].isin(x_times)]['OUT'].astype(float).tolist()[:3]

    if not (x_values and len(x_times) == len(target_times)):
        return None

    y_true_times = y_times
    doses = patient_df['DOSE'].astype(float).tolist()[0]

    classe = 1 if (patient_df['CYP'] == 1).all() else 0
    traitement = 0 if float(patient_df['II'].values[0]) == 24 else 1

    static = [doses, traitement, classe]

    return {
        'times_val': y_times,
        'values_val': y_values,
        'y_true_times': y_true_times,
        'x_values': x_values,
        'x_times': x_times,
        'doses': doses,
        'static': static,
        'patient_id': patient_id,
        'dataset_number': 0,
        'auc_be': 0,  # Placeholder
        'auc_red': 0,  # Placeholder
        'others': [],  # Placeholder
    }


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

def extract_pccp(file_path, pharmac_path, plot=False):
    data_dict = {}
    df = load_and_clean_dataframe(file_path, sep=';', skipinitialspace=True, skiprows=1)
    df['ID_new'] = df['ID'].astype(int).astype(str).str.zfill(2) + df['PERI'].astype(str)
    ids_to_remove = df.loc[df['DV'] > 50000, 'ID_new'].unique()
    df = df[~df['ID_new'].isin(ids_to_remove)]
    
    df['DOSE'] = df['DOSE'] / df['DOSE'].max()
    df['DV'] = df['DV'] / 1000
    max_out = df['DV'].max()
    df['DV'] = df['DV'] / max_out
    
    unique_patient = df['ID_new'].unique()
    data_be = pd.DataFrame(extract_pharmac_data_from_file(pharmac_path))

    for patient_id in unique_patient:
        patient_df = df[df['ID_new'] == patient_id]
        patient_dict = build_patient_dictionary(patient_df, patient_id, max_out, None)
        if patient_dict:
            data_dict[patient_id] = patient_dict

    return data_dict, max_out

def extract_aadapt_tac(file_path, pharmac_path, auc_path, plot=False):
    data_dict = {}
    df = load_and_clean_dataframe(file_path, sep=';', skipinitialspace=True, skiprows=1)
    
    df = df[(df["nycth"] == "J") | (df["nycth"].isna())]
    df['ID_new'] = df['#ID'] + df['periode'].astype(str)
    df['OUT'] = pd.to_numeric(df['OUT'].replace('.', None), errors='coerce')
    df['DOSE'] = pd.to_numeric(df['DOSE'].replace('.', None), errors='coerce')
    df['TIME'] = pd.to_numeric(df['TIME'].replace('.', None), errors='coerce')
    ids_to_remove = df.loc[df['OUT'] > 50000, 'ID_new'].unique()
    df = df[~df['ID_new'].isin(ids_to_remove)]
    
    df['DOSE'] = df['DOSE'] / df['DOSE'].max()
    df['OUT2'] = df['OUT']
    max_out = df['OUT'].max()
    df['OUT2'] = df['OUT2'] / max_out
    df['OUT'] = df['OUT'] / max_out
    boxcox_transformed_data, best_lambda = stats.boxcox(df['OUT'].dropna())
    mask = (df['OUT'] > 0) & (df['OUT'].notna())
    df.loc[mask, 'OUT'] = boxcox_transformed_data

    unique_patient = df['ID_new'].unique()
    data_be = pd.DataFrame(extract_pharmac_data_from_file(pharmac_path))
    df_auc = pd.read_csv(auc_path)

    for patient_id in unique_patient:
        if patient_id in ['02PF-8J7', '0400-8J7', '05MT-8J7']:
            continue

        patient_df = df[df['ID_new'] == patient_id]
        patient_dict = build_patient_dictionary(patient_df, patient_id, max_out, best_lambda)
        if patient_dict:
            data_dict[patient_id] = patient_dict

    return data_dict, [max_out, best_lambda]

def extract_pccp_tac_2(file_path, pharmac_path, plot=False):
    data_dict = {}
    df = load_and_clean_dataframe(file_path, sep=';', skipinitialspace=True, skiprows=1)
    df['ID_new'] = df['#ID'] + df['visit'].astype(str)
    df['OUT'] = pd.to_numeric(df['OUT'].replace('.', None), errors='coerce')
    df['DOSE'] = pd.to_numeric(df['DOSE'].replace('.', None), errors='coerce')
    df['TIME'] = pd.to_numeric(df['TIME'].replace('.', None), errors='coerce') / 60
    ids_to_remove = df.loc[df['OUT'] > 50000, 'ID_new'].unique()
    df = df[~df['ID_new'].isin(ids_to_remove)]
    
    df['DOSE'] = df['DOSE'] / df['DOSE'].max()
    df['OUT'] = df['OUT'] / 1000
    max_out = df['OUT'].max()
    df['OUT'] = df['OUT'] / max_out
    
    unique_patient = df['ID_new'].unique()
    data_be = pd.DataFrame(extract_pharmac_data_from_file(pharmac_path))

    for patient_id in unique_patient:
        if patient_id in ['LM15M6', 'RE02M6']:
            continue
        
        patient_df = df[df['ID_new'] == patient_id]
        patient_dict = build_patient_dictionary(patient_df, patient_id, max_out, None)
        if patient_dict:
            data_dict[patient_id] = patient_dict

    return data_dict, max_out

def extract_pccp_tac(file_path, pharmac_path, plot=False):
    data_dict = {}
    df = load_and_clean_dataframe(file_path, sep=";", skipinitialspace=True, skiprows=0)
    df['ID_2'] = df['ID'].astype(str) + df['PERI'].astype(str)
    df['ID_new'] = df['ID'].astype(str) + df['PERI'].astype(str) + df['ET'].astype(str)
    df['OUT'] = pd.to_numeric(df['DV'].replace('.', None), errors='coerce')
    df['OUT2'] = pd.to_numeric(df['DV'].replace('.', None), errors='coerce')
    df['DOSE'] = pd.to_numeric(df['AMT'].replace('.', None), errors='coerce')
    df['TIME'] = pd.to_numeric(df['TIME'].replace('.', None), errors='coerce')
    ids_to_remove = df.loc[df['OUT'] > 50000, 'ID_new'].unique()
    df = df[~df['ID_new'].isin(ids_to_remove)]
    
    df['DOSE'] = df['DOSE'] / df['DOSE'].max()
    max_out = df['OUT'].max()
    df['OUT'] = df['OUT'] / max_out
    df['OUT2'] = df['OUT2'] / max_out
    boxcox_transformed_data, best_lambda = stats.boxcox(df[df['OUT'] != 0]['OUT'])
    mask = (df['OUT'] > 0) & (df['OUT'].notna())
    df.loc[mask, 'OUT'] = boxcox_transformed_data
    
    unique_patient = df['ID_new'].unique()
    data_be = pd.DataFrame(extract_pharmac_data_from_file(pharmac_path))

    for patient_id in unique_patient:
        patient_df = df[df['ID_new'] == patient_id]
        patient_dict = build_patient_dictionary(patient_df, patient_id, max_out, best_lambda)
        if patient_dict:
            data_dict[patient_id] = patient_dict

    return data_dict, [max_out, best_lambda]

def extract_gen_tac(train_data_path, test_data_path=None, plot=False, exp=""):
    data_dict = {}
    if train_data_path:
        df1 = load_and_clean_dataframe(train_data_path, sep=",", skiprows=0)
    if test_data_path and os.path.exists(test_data_path):
        df2 = load_and_clean_dataframe(test_data_path, sep=",", skiprows=0)
        df = pd.concat([df1, df2])
    else:
        df = df1

    df['ID_2'] = df['ID'].astype(str)
    df['ID_new'] = df['ID'].astype(str)
    df['OUT'] = pd.to_numeric(df['DV'].replace('.', None), errors='coerce')
    df['DOSE'] = pd.to_numeric(df['AMT'].replace('.', None), errors='coerce')
    df['TIME'] = pd.to_numeric(df['TIME'].replace('.', None), errors='coerce')
    ids_to_remove = df.loc[df['OUT'] > 50000, 'ID_new'].unique()
    df = df[~df['ID_new'].isin(ids_to_remove)]

    df['DOSE'] = df['DOSE'] / df['DOSE'].max()
    df['OUT'] = df['OUT']
    max_out = df['OUT'].max()
    df['OUT'] = df['OUT'] / max_out
    boxcox_transformed_data, best_lambda = stats.boxcox(df[df['OUT'] > 0]['OUT'])
    mask = (df['OUT'] > 0) & (df['OUT'].notna())
    df.loc[mask, 'OUT'] = boxcox_transformed_data

    unique_patient = df['ID_new'].unique()

    try:
        search_pattern = os.path.join(f"exp_run_all/{exp}", 'tacro_mapbayest_auc_*.csv')
        matching_files = glob.glob(search_pattern)
        filename = matching_files[-1] if matching_files else None
        if filename:
            data_be = pd.read_csv(filename)
        else:
            data_be = pd.DataFrame()
    except Exception as e:
        print(f"Error loading data_be: {e}")
        data_be = pd.DataFrame()

    for patient_id in unique_patient:
        patient_df = df[df['ID_new'] == patient_id]
        patient_dict = build_patient_dictionary(patient_df, patient_id, max_out, best_lambda)
        if patient_dict:
            data_dict[patient_id] = patient_dict

    return data_dict, [max_out, best_lambda]

import re 
import chardet

def extract_stimmugrep(file_path, pharmac_path, plot=False):
    data_dict = {}
    df = load_and_clean_dataframe(file_path, sep=';', encoding='ISO-8859-1', on_bad_lines='skip')
    df['ID_new'] = df['CODE_DOSAGE']
    
    df['DOSE'] = df['POSO_AMP_MATIN'] / df['POSO_AMP_MATIN'].max()

    data_be = pd.DataFrame(extract_pharmac_data_from_file(pharmac_path))
    
    all_unique_patient = df['ID_new'].unique()
    unique_patient = [patient_id for patient_id in all_unique_patient if patient_id != 'C03-P1']

    pattern_time = r'^T\d+$' 
    all_values = []
    for patient_id in unique_patient:
        patient_df = df[df['CODE_DOSAGE'] == patient_id]
        y_times = []
        y_values = []
        time_cols = [col for col in df.columns if re.match(pattern_time, col)]
            
        for time_col in time_cols:
            amp_col = time_col + '_AMP'
            if amp_col in patient_df.columns:
                amp_value = patient_df[amp_col].iloc[0]
                if pd.notna(amp_value):
                    time_value = patient_df[time_col].iloc[0]
                    if pd.notna(time_value):
                        y_times.append(float(time_value) / 60)
                        y_values.append(float(amp_value) / 1000)
                        all_values.append(float(amp_value) / 1000)
        
        if pd.notna(patient_df['POSO_TAC_VEILLE']).item():
            traitement = 1
        elif pd.notna(patient_df['POSO_CsA_VEILLE']).item():
            traitement = 0

        patient_be = data_be[data_be['Nom'] == patient_id]
        if patient_be.empty:
            continue
        
        patient_dict = build_patient_dictionary(patient_df, patient_id, max(all_values), None)
        if patient_dict:
            data_dict[patient_id] = patient_dict
            
    max_out = max(all_values)
    for patient_id in data_dict.keys():
        data_dict[patient_id]['x_values'] = [x / max_out for x in data_dict[patient_id]['x_values']]
        data_dict[patient_id]['values_val'] = [v / max_out for v in data_dict[patient_id]['values_val']]

    return data_dict, max_out
def extract_concept(file_path, pharmac_path_ciclo, pharmac_path_siro, plot=False):
    data_dict = {}
    df = load_and_clean_dataframe(file_path, sep=';', skipinitialspace=True, skiprows=1)
    df['TIME'] /= 60
    df['ID_new'] = df['#ID'].astype(str) + "_" + df['visit'].astype(str)
    df['OUT'] = pd.to_numeric(df['OUT'].replace('.', None), errors='coerce')
    df['DOSE'] = pd.to_numeric(df['DOSE'], errors='coerce')
    df['DOSE'] = df['DOSE'] / df['DOSE'].max()

    df['OUT'] = df['OUT'] / 1000
    max_out = df['OUT'].max()
    df['OUT'] = df['OUT'] / max_out

    data_be_ciclo = pd.DataFrame(extract_pharmac_data_from_file(pharmac_path_ciclo))
    data_be_siro = pd.DataFrame(extract_pharmac_data_from_file(pharmac_path_siro))
    all_treatments = {'csa': 0, 'Tacrolimus': 1, 'srl': 2}
    unique_patient = df['ID_new'].unique()

    for patient_id in unique_patient:
        patient_df = df[df['ID_new'] == patient_id]

        if not data_be_ciclo[data_be_ciclo['Nom'] == patient_id].empty:
            patient_be = data_be_ciclo[data_be_ciclo['Nom'] == patient_id]
        else:
            patient_be = data_be_siro[data_be_siro['Nom'] == patient_id]
        if patient_be.empty:
            continue

        patient_dict = build_patient_dictionary(patient_df, patient_id, max_out, None)
        if patient_dict:
            data_dict[patient_id] = patient_dict

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

def extract_pigrec(file_path, pharmac_path_ciclo, pharmac_path_tacro, plot=False):
    data_dict = {}
    df = load_and_clean_dataframe(file_path, sep=';', skipinitialspace=True, skiprows=1)
    df['ID_new'] = df['ID'].astype(str)
    df['OUT'] = pd.to_numeric(df['OUT'].replace('.', None), errors='coerce')
    df['DOSE'] = pd.to_numeric(df['DOSE'], errors='coerce')
    df['EVID'] = pd.to_numeric(df['EVID'], errors='coerce')
    df['DOSE'] = df['DOSE'] / df['DOSE'].max()
    
    max_out = df['OUT'].max()
    df['OUT'] = df['OUT'] / max_out

    unique_patient = df['ID_new'].unique()
    all_treatments = {'Ciclosporine': 0, 'Tacrolimus': 1, 'Sirolimus': 2}
    data_be_ciclo = pd.DataFrame(extract_pharmac_data_from_file(pharmac_path_ciclo))
    data_be_tacro = pd.DataFrame(extract_pharmac_data_from_file(pharmac_path_tacro))

    for patient_id in unique_patient:
        patient_df = df[df['ID_new'] == patient_id]

        if not data_be_ciclo[data_be_ciclo['Nom'] == patient_id].empty:
            patient_be = data_be_ciclo[data_be_ciclo['Nom'] == patient_id]
        else:
            patient_be = data_be_tacro[data_be_tacro['Nom'] == patient_id]
        if patient_be.empty:
            continue

        patient_dict = build_patient_dictionary(patient_df, patient_id, max_out, None)
        if patient_dict:
            data_dict[patient_id] = patient_dict

    return data_dict, max_out

def extract_ped(file_path, pharmac_path_ciclo, pharmac_path_siro, plot=False):
    data_dict = {}
    df = load_and_clean_dataframe(file_path, sep=",", skipinitialspace=True, skiprows=0)
    
    df['TIME'] = pd.to_numeric(df['TIME'], errors='coerce') / 60
    df['ID_new'] = df['id'].astype(str) + "_" + df['peri'].astype(str)
    df['MPA'] = pd.to_numeric(df['MPA'].replace('.', None), errors='coerce')
    
    df['DOSE'] = pd.to_numeric(df['Dose'], errors='coerce')
    df['DOSE'] = df['DOSE'] / df['DOSE'].max()

    max_out = df['MPA'].max()
    df['MPA'] = df['MPA'] / max_out

    unique_patient = df['ID_new'].unique()
    all_treatments = {'Ciclosporine': 0, 'Tacrolimus': 1, 'Sirolimus': 2, 'other': 3}
    data_be_ciclo = pd.DataFrame(extract_pharmac_data_from_file(pharmac_path_ciclo))
    data_be_siro = pd.DataFrame(extract_pharmac_data_from_file(pharmac_path_siro))

    for patient_id in unique_patient:
        patient_df = df[df['ID_new'] == patient_id]
        
        if not data_be_ciclo[data_be_ciclo['Nom'] == patient_id].empty:
            patient_be = data_be_ciclo[data_be_ciclo['Nom'] == patient_id]
        else:
            patient_be = data_be_siro[data_be_siro['Nom'] == patient_id]
        if patient_be.empty:
            continue

        patient_dict = build_patient_dictionary(patient_df, patient_id, max_out, None)
        if patient_dict:
            data_dict[patient_id] = patient_dict

    return data_dict, max_out

def extract_stablocine(file_path, pharmac_path, plot=False):
    data_dict = {}
    df = load_and_clean_dataframe(file_path, sep=';', skipinitialspace=True, skiprows=1)
    df['TIME'] = df['TIME_H'].astype(str)
    df['TIME'] = pd.to_numeric(df['TIME'], errors='coerce')
    df['ID_new'] = df['#ID'].astype(str)
    df['OUT'] = pd.to_numeric(df['OUT'].replace('.', None), errors='coerce')
    
    df['DOSE'] = df['DOSE'] / df['DOSE'].max()

    max_out = df['OUT'].max()
    df['OUT'] = df['OUT'] / max_out

    unique_patient = df['ID_new'].unique()
    data_be = pd.DataFrame(extract_pharmac_data_from_file(pharmac_path))
    for patient_id in unique_patient:
        patient_df = df[df['ID_new'] == patient_id]
        patient_be = data_be[data_be['Nom'] == patient_id]
        if patient_be.empty:
            continue

        patient_dict = build_patient_dictionary(patient_df, patient_id, max_out, None)
        if patient_dict:
            data_dict[patient_id] = patient_dict

    return data_dict, max_out

def extract_theo(file_path, plot=False):
    data_dict = {}
    df = load_and_clean_dataframe(file_path, sep=",", skipinitialspace=True, skiprows=0)
    df['ID_new'] = df['Subject'].astype(str)
    df['OUT'] = pd.to_numeric(df['conc'].replace('.', None), errors='coerce')
    df['DOSE'] = pd.to_numeric(df['Dose'].replace('.', None), errors='coerce')
    
    df['DOSE'] = df['DOSE'] / df['DOSE'].max()

    max_out = df['OUT'].max()
    df['OUT'] = df['OUT'] / max_out

    unique_patient = df['ID_new'].unique()
    
    for patient_id in unique_patient:
        patient_df = df[df['ID_new'] == patient_id]
        patient_dict = build_patient_dictionary(patient_df, patient_id, max_out, None)
        if patient_dict:
            data_dict[patient_id] = patient_dict
            
    return data_dict, max_out

def extract_tls(file_path, plot=False):
    data_dict = {}
    df = load_and_clean_dataframe(file_path, sep=";", skipinitialspace=True, skiprows=0)
    df['CORT_shifted'] = df.groupby('ID')['CORT'].shift()
    df['CORT_changed'] = (df['CORT'] != df['CORT_shifted']).fillna(True)
    df['CORT_group'] = df.groupby('ID')['CORT_changed'].cumsum().astype(int)
    df['DOSE'] = df['DOSE'] / df['DOSE'].max()
    df['CREA'] = df['CREA'] / df['CREA'].max()
    df['CORT'] = df['CORT'] / df['CORT'].max()
    df['ID_new'] = df['ID'].astype(str) + "_" + df['CORT'].astype(str)
    df.drop(columns=['CORT_shifted', 'CORT_changed', 'CORT_group'], inplace=True)
    max_out = df['OUT'].max()
    df['OUT'] = df['OUT'] / max_out

    unique_patient = df['ID_new'].unique()
    for patient_id in unique_patient:
        patient_df = df[df['ID_new'] == patient_id]
        patient_dict = build_patient_dictionary(patient_df, patient_id, max_out, None)
        if patient_dict:
            data_dict[patient_id] = patient_dict

    return data_dict, max_out

def extract_val_tacro(file_path, plot=False):
    # Load and clean the data
    df = pd.read_csv(file_path, skiprows=1)
    df['TIME'] = pd.to_numeric(df['TIME'], errors='coerce')
    df['OUT'] = pd.to_numeric(df['OUT'].replace('.', None), errors='coerce') * 10
    df['ID'] = pd.to_numeric(df['#ID'], errors='coerce')
    df = df.dropna(subset=['TIME', 'OUT', 'ID'])
    
    return df

def extract_in_tacro(file_path, plot=False):
    df = pd.read_csv(file_path)
    df['TIME'] = pd.to_numeric(df['TIME'], errors='coerce')
    df['OUT'] = pd.to_numeric(df['OUT'].replace('.', None), errors='coerce') / 100
    df['ID'] = pd.to_numeric(df['id'], errors='coerce')
    df['DOSE'] = pd.to_numeric(df['DOSE'], errors='coerce')
    df = df.dropna(subset=['TIME', 'OUT', 'ID'])

    return df

def extract_in_val_mmf(file_path, plot=False):
    data_dict = {}
    df = load_and_clean_dataframe(file_path, sep=";", skipinitialspace=True, skiprows=0)
    max_out = df['OUT'].max()
    df['OUT'] = df['OUT'] / max_out
    df['DOSE'] = df['DOSE'] / df['DOSE'].max()

    unique_patient = df['ID'].unique()
    for patient_id in unique_patient:
        patient_df = df[df['ID'] == patient_id]
        patient_dict = build_patient_dictionary(patient_df, patient_id, max_out, None)
        if patient_dict:
            data_dict[patient_id] = patient_dict

    return data_dict, max_out

def create_dataset(file_val, file_in):
    df_val = extract_val_tacro(file_val)
    df_in = extract_in_tacro(file_in)
    data_dict = {}
    unique_patient = df_in['ID'].unique()

    for patient_id in unique_patient:
        patient_df_in = df_in[df_in['ID'] == patient_id]
        patient_df_val = df_val[df_val['ID'] == patient_id]

        y_times = patient_df_val['TIME'].tolist()
        y_values = patient_df_val['OUT'].astype(float).tolist()
        x_values = patient_df_in['OUT'].astype(float).tolist()
        x_times = patient_df_in['TIME'].astype(float).tolist()
        
        if y_times and x_values and len(x_times) == 3 and len(y_times) == 17:
            doses = patient_df_in['DOSE'].astype(float).tolist()[0]
            data_dict[patient_id] = {
                'times_val': y_times,
                'values_val': y_values,
                'x_values': x_values,
                'x_times': x_times,
                'doses': doses
            }

    return data_dict


class TacroDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict, is_train=False, mean=None, std=None):
        self.data_dict = data_dict
        self.is_train = is_train
        self.mean = mean
        self.std = std
        self.patient_ids = list(self.data_dict.keys())

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        patient_dict = self.data_dict[patient_id]

        # Convert to tensors
        data = torch.from_numpy(np.array(patient_dict['values_val'])).float()
        data_t = torch.from_numpy(np.array(patient_dict['times_val'])).float()
        data_t_input = torch.from_numpy(np.array(patient_dict['x_times'])).float()
        data_input = torch.from_numpy(np.array(patient_dict['x_values'])).float()
        dose = torch.from_numpy(np.array(patient_dict['doses'])).float()
        static = torch.from_numpy(np.array(patient_dict.get('static', []))).float()
        auc_be = torch.from_numpy(np.array(patient_dict['auc_be'])).float()
        auc_red = torch.from_numpy(np.array(patient_dict['auc_red'])).float()
        y_true_times = torch.from_numpy(np.array(patient_dict['y_true_times'])).float()
        dataset = torch.from_numpy(np.array(patient_dict['dataset_number'])).float()
        others = torch.from_numpy(np.array(patient_dict.get('others', []))).float()

        return (
            data_t, data, data_t_input, data_input, static, auc_be,
            y_true_times, dataset, patient_id, others, auc_red, dose
        )


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