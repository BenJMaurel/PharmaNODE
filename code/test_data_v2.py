import os
import torch
from torch.utils.data import DataLoader
import numpy as np

# Import your extraction, dataset, and collate functions
from lib.read_tacro import (
    extract_gen_tac, extract_gen_tac_film,
    TacroDataset, TacroFilmDataset,
    collate_fn_tacro, collate_fn_tacro_film
)

def analyze_tensor(name, tensor):
    """Helper to safely print stats of a tensor"""
    if tensor is None:
        print(f"  {name}: None")
        return
    
    # Convert boolean or integer masks to float for mean calculation
    t_float = tensor.float() if tensor.dtype != torch.float32 else tensor
    
    print(f"  {name}:")
    print(f"    Shape: {list(tensor.shape)}")
    print(f"    Mean : {t_float.mean().item():.4f}")
    print(f"    Max  : {t_float.max().item():.4f}")
    print(f"    Min  : {t_float.min().item():.4f}")

def compare_dataloaders(classic_train_path, classic_test_path, film_train_path, film_test_path):
    print("======================================================")
    print(" DATALOADER STATISTIC COMPARISON")
    print("======================================================")
    device = torch.device('cpu')

    # --- 1. Extraction Phase ---
    print("\n[1] EXTRACTION PHASE")
    try:
        # Load Classic
        print("Extracting Classic Data...")
        dict_c, scaler_c = extract_gen_tac([classic_train_path, classic_test_path], plot=False)
        print(f"  -> Classic Max_Out Scaler: {scaler_c[0]:.4f}")
        
        # Load FiLM
        print("Extracting FiLM Data...")
        dict_f, scaler_f = extract_gen_tac_film([film_train_path, film_test_path], exp=None)
        print(f"  -> FiLM Max_Out Scaler: {scaler_f[0]:.4f}")
    except Exception as e:
        print(f"Extraction Failed: {e}")
        return

    # --- 2. Dataset Initialization ---
    print("\n[2] DATASET INITIALIZATION")
    dataset_c = TacroDataset(dict_c)
    dataset_f = TacroFilmDataset(dict_f)
    print(f"  Classic Dataset Size: {len(dataset_c)} patients")
    print(f"  FiLM Dataset Size:    {len(dataset_f)} patients")

    # --- 3. DataLoader & Collation ---
    print("\n[3] DATALOADER BATCH STATISTICS (Batch Size = 128)")
    
    # Dummy args class for classic collate_fn
    class DummyArgs:
        pass
    
    dl_c = DataLoader(
        dataset_c, batch_size=128, shuffle=False, 
        collate_fn=lambda b: collate_fn_tacro(b, DummyArgs(), device)
    )
    
    dl_f = DataLoader(
        dataset_f, batch_size=128, shuffle=False, 
        collate_fn=lambda b: collate_fn_tacro_film(b, device)
    )

    # Fetch exactly one batch from each
    batch_c = next(iter(dl_c))
    batch_f = next(iter(dl_f))

    print("\n--- CLASSIC BATCH STATS ---")
    analyze_tensor("Observed Data (Encoder Input)", batch_c["observed_data"])
    analyze_tensor("Observed Timepoints", batch_c["observed_tp"])
    analyze_tensor("Data to Predict (Targets)", batch_c["data_to_predict"])
    analyze_tensor("Timepoints to Predict", batch_c["tp_to_predict"])
    analyze_tensor("Dose", batch_c["dose"])
    analyze_tensor("Static Covariates", batch_c["static"])
    
    print("\n--- FiLM BATCH STATS (Visit 1) ---")
    analyze_tensor("Observed Data V1 (Encoder Input)", batch_f["observed_data_v1"])
    analyze_tensor("Observed Timepoints V1", batch_f["observed_tp_v1"])
    analyze_tensor("Data to Predict V1 (Targets)", batch_f["data_to_predict_v1"])
    analyze_tensor("Timepoints to Predict V1", batch_f["tp_to_predict_v1"])
    analyze_tensor("Dose V1", batch_f["dose_v1"])
    analyze_tensor("Static Covariates V1", batch_f["static_v1"])
    
    print("\n--- GROUP 1 (Prograf) COMPARISON ---")
    # Check if the network is receiving fundamentally different inputs for Prograf
    static_c = batch_c["static"]
    static_f = batch_f["static_v1"]
    
    # Assuming treatment type is at index 1 of the static vector (0: Advagraf, 1: Prograf)
    mask_c_prograf = (static_c[:, 1] == 1)
    mask_f_prograf = (static_f[:, 1] == 1)
    
    if mask_c_prograf.sum() > 0:
        targets_c_prograf = batch_c["data_to_predict"][mask_c_prograf]
        print(f"  Classic Prograf Target Mean: {targets_c_prograf.float().mean().item():.4f}")
    
    if mask_f_prograf.sum() > 0:
        targets_f_prograf = batch_f["data_to_predict_v1"][mask_f_prograf]
        print(f"  FiLM V1 Prograf Target Mean: {targets_f_prograf.float().mean().item():.4f}")

    print("======================================================")

if __name__ == "__main__":
    # Adjust paths if your files are inside a specific directory
    C_TRAIN = "./results/2/virtual_cohort_train.csv"
    C_TEST  = "./results/2/virtual_cohort_test.csv"
    F_TRAIN = "./results/exp_film_run/3/virtual_cohort_film_train.csv"
    F_TEST  = "./results/exp_film_run/3/virtual_cohort_film_test.csv"
    
    compare_dataloaders(C_TRAIN, C_TEST, F_TRAIN, F_TEST)