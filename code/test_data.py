import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def clean_dataframe(df):
    """Converts placeholder '.' to NaN and casts columns to numeric where appropriate."""
    df = df.copy()
    # Convert DV (Concentrations) and AMT (Doses) to numeric, coercing '.' to NaN
    df['DV'] = pd.to_numeric(df['DV'].replace('.', np.nan), errors='coerce')
    df['AMT'] = pd.to_numeric(df['AMT'].replace('.', np.nan), errors='coerce')
    df['TIME'] = pd.to_numeric(df['TIME'].replace('.', np.nan), errors='coerce')
    return df

def compare_datasets(classic_path, film_path):
    print("==================================================")
    print(" DATASET COMPARISON DIAGNOSTIC SCRIPT")
    print("==================================================")

    if not os.path.exists(classic_path) or not os.path.exists(film_path):
        import pdb; pdb.set_trace()
        print(f"Error: Could not find one or both files.")
        print(f"Classic: {classic_path}")
        print(f"FiLM: {film_path}")
        return

    # Load and clean
    df_classic = clean_dataframe(pd.read_csv(classic_path))
    df_film = clean_dataframe(pd.read_csv(film_path))

    # --- 1. Basic Structure ---
    print("\n[1] BASIC STRUCTURE & COUNTS")
    num_patients_c = df_classic['ID'].nunique()
    num_patients_f = df_film['ID'].nunique()
    
    print(f"Classic -> Total Rows: {len(df_classic)}, Unique Patients: {num_patients_c}")
    print(f"FiLM    -> Total Rows: {len(df_film)}, Unique Patients: {num_patients_f}")

    if 'VISIT' in df_film.columns:
        print(f"FiLM    -> Unique Visits per patient: {df_film['VISIT'].unique().tolist()}")
    else:
        print("WARNING: 'VISIT' column missing in FiLM dataset!")

    # --- 2. Extract Doses ---
    print("\n[2] DOSE (AMT) DISTRIBUTIONS")
    doses_c = df_classic['AMT'].dropna()
    doses_f = df_film['AMT'].dropna()
    doses_f_v1 = df_film[df_film['VISIT'] == 1]['AMT'].dropna() if 'VISIT' in df_film.columns else []

    print(f"Classic        -> Mean Dose: {doses_c.mean():.4f} mg | Min: {doses_c.min()} | Max: {doses_c.max()}")
    print(f"FiLM (Overall) -> Mean Dose: {doses_f.mean():.4f} mg | Min: {doses_f.min()} | Max: {doses_f.max()}")
    if len(doses_f_v1) > 0:
        print(f"FiLM (Visit 1) -> Mean Dose: {doses_f_v1.mean():.4f} mg")

    # --- 3. Extract Concentrations (DV) ---
    print("\n[3] CONCENTRATION (DV) DISTRIBUTIONS")
    dv_c = df_classic['DV'].dropna()
    dv_f = df_film['DV'].dropna()
    
    print(f"Classic -> Mean DV: {dv_c.mean():.4f} | Median: {dv_c.median():.4f} | Max: {dv_c.max():.4f}")
    print(f"FiLM    -> Mean DV: {dv_f.mean():.4f} | Median: {dv_f.median():.4f} | Max: {dv_f.max():.4f}")

    # --- 4. True AUC vs Predicted AUC (if applicable) ---
    print("\n[4] TRUE AUC DISTRIBUTIONS")
    if 'AUC' in df_classic.columns and 'AUC' in df_film.columns:
        # AUC is repeated per row, so we take the first unique one per patient/visit
        auc_c = df_classic.groupby('ID')['AUC'].first()
        auc_f_v1 = df_film[df_film['VISIT'] == 1].groupby('ID')['AUC'].first()
        print(f"Classic        -> Mean AUC: {auc_c.mean():.4f}")
        print(f"FiLM (Visit 1) -> Mean AUC: {auc_f_v1.mean():.4f}")
    
    # --- 5. Covariates Heterogeneity ---
    print("\n[5] COVARIATE HETEROGENEITY (HT, CYP, DRUG)")
    print("Classic Formulation Split:")
    print(df_classic.groupby('ID')['DRUG'].first().value_counts(normalize=True).to_string())
    print("FiLM Formulation Split:")
    print(df_film.groupby('ID')['DRUG'].first().value_counts(normalize=True).to_string())

    # --- 6. Visual Comparison ---
    print("\nGenerating visual comparison plot (dataset_comparison.png)...")
    
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Doses
    sns.histplot(doses_c, color='blue', alpha=0.5, label='Classic', ax=axes[0], stat='density', bins=10)
    sns.histplot(doses_f, color='orange', alpha=0.5, label='FiLM (All Visits)', ax=axes[0], stat='density', bins=10)
    axes[0].set_title('Dose (AMT) Distribution')
    axes[0].legend()

    # Plot 2: Concentrations (Log Scale to see tails)
    sns.kdeplot(dv_c, color='blue', fill=True, alpha=0.3, label='Classic', ax=axes[1])
    sns.kdeplot(dv_f, color='orange', fill=True, alpha=0.3, label='FiLM', ax=axes[1])
    axes[1].set_title('Concentration (DV) Density')
    axes[1].legend()

    # Plot 3: Time of Observations
    times_c = df_classic[df_classic['DV'].notna()]['TIME']
    times_f = df_film[df_film['DV'].notna()]['TIME']
    sns.kdeplot(times_c, color='blue', label='Classic', ax=axes[2])
    sns.kdeplot(times_f, color='orange', label='FiLM', ax=axes[2])
    axes[2].set_title('Observation Times Distribution')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('dataset_comparison.png')
    print("Plot saved successfully. Check dataset_comparison.png!")
    print("==================================================")


if __name__ == "__main__":
    # Ensure these paths point to where your generated datasets are saved
    CLASSIC_FILE = "./results/2/virtual_cohort_train.csv"
    FILM_FILE = "./results/exp_film_run/3/virtual_cohort_film_train.csv"
    
    # If they are inside the results/ folder, uncomment below:
    # CLASSIC_FILE = "./results/virtual_cohort_train.csv"
    # FILM_FILE = "./results/exp_film_run/virtual_cohort_film_train.csv"
    
    compare_datasets(CLASSIC_FILE, FILM_FILE)