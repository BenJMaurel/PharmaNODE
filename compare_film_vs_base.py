###########################
# Latent ODEs: Base vs FiLM Comparison Script
# Task: Standard Reconstruction (No FiLM Extrapolation)
###########################

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg') # Safe for environments without a display
import matplotlib.pyplot as plt
from scipy.special import inv_boxcox
from torch.utils.data import DataLoader
import torch

import lib.utils as utils
from lib.create_latent_ode_model import create_LatentODE_model
from lib.read_tacro import extract_gen_tac_film, TacroFilmDataset, collate_fn_tacro_film

# =========================================================================
# ARGUMENTS
# =========================================================================
parser = argparse.ArgumentParser('Compare Base vs FiLM Latent ODE (Reconstruction)')
parser.add_argument('--exp', type=str, default='exp_film_run', help="Folder name for the FiLM dataset")
parser.add_argument('--load-base', type=str, required=True, help="Path to the trained BASE model checkpoint (.ckpt)")
parser.add_argument('--load-film', type=str, required=True, help="Path to the trained FILM model checkpoint (.ckpt)")
parser.add_argument('-b', '--batch-size', type=int, default=1000)
parser.add_argument('--viz', action='store_true', help="Show and save comparison plots")

# Shared Architectural arguments
parser.add_argument('-l', '--latents', type=int, default=10)
parser.add_argument('--rec-dims', type=int, default=20)
parser.add_argument('--rec-layers', type=int, default=1)
parser.add_argument('--gen-layers', type=int, default=1)
parser.add_argument('-u', '--units', type=int, default=100)
parser.add_argument('-g', '--gru-units', type=int, default=100)
parser.add_argument('--z0-encoder', type=str, default='odernn')
parser.add_argument('--dataset', type=str, default='PK_Tacro')
parser.add_argument('--poisson', action='store_true')
parser.add_argument('--classif', action='store_true')
parser.add_argument('--linear-classif', action='store_true')
parser.add_argument('--use_gmm', action='store_true')
parser.add_argument('--use_flow', action='store_true')
parser.add_argument('--use_gmm_v', action='store_true')
parser.add_argument('-nc', '--n_components', type=int, default=4)
parser.add_argument('--extrap', action='store_true')

args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
utils.makedirs("results/")

def unscale_data(data_tensor, scaler_info):
    """Reverses the max_out and BoxCox scaling to get physiological concentrations."""
    data_np = data_tensor.detach().cpu().numpy() if torch.is_tensor(data_tensor) else data_tensor
    if scaler_info is None: return data_np
    
    max_out = scaler_info[0]
    best_lambda = scaler_info[1] if len(scaler_info) > 1 else None
    
    if best_lambda is not None:
        data_np = inv_boxcox(data_np, best_lambda)
        data_np = np.nan_to_num(data_np, nan=0.0) # Handle NaN from negative BoxCox inversions
        
    return data_np * max_out

def calculate_auc(concentrations, times, static_treatments):
    """
    Safely calculates AUC.
    For Prograf (static[:, 1] == 1), integrates only up to 12h.
    For Advagraf (static[:, 1] == 0), integrates up to 24h.
    """
    aucs = []
    
    for i in range(concentrations.shape[0]):
        # Check formulation: 1 = Prograf (12h), 0 = Advagraf (24h)
        is_prograf = bool(static_treatments[i, 1].item())
        cutoff_time = 12.0 if is_prograf else 24.0
        
        # Filter data to only include valid integration times
        valid_idx = times <= cutoff_time
        valid_times = times[valid_idx]
        valid_concs = concentrations[i, valid_idx]
        
        # Calculate Trapezoidal AUC
        aucs.append(np.trapz(valid_concs, valid_times))
        
    return np.array(aucs)

def plot_comparative_reconstruction(data_v1_obs, tp_v1_obs, data_v1_true, tp_v1_true, 
                                    pred_base, pred_film, dense_tp, doses_v1):
    """Generates a side-by-side plot comparing standard reconstruction for 4 patients."""
    fig = plt.figure(figsize=(16, 10), facecolor='white')
    n_plots = min(4, data_v1_obs.shape[0])
    
    for i in range(n_plots):
        ax = fig.add_subplot(2, 2, i+1)
        
        # 1. True Context (The 3 sparse points the models saw)
        ax.plot(tp_v1_obs, data_v1_obs[i, :], 'bX', label=f'Observed Inputs (Dose: {doses_v1[i]:.1f})', markersize=10, zorder=5)
        
        # 2. True Sparse Trajectory (Ground Truth)
        ax.plot(tp_v1_true, data_v1_true[i, :], 'g-', alpha=0.3, linewidth=4, label='True Sparse Trajectory')
        
        # 3. Base Model Dense Reconstruction
        ax.plot(dense_tp, pred_base[i, :], color='gray', linestyle='--', linewidth=2.5, label='Base Model Recon.')
        
        # 4. FiLM Model Dense Reconstruction (Standard path, no FiLM used)
        ax.plot(dense_tp, pred_film[i, :], color='purple', linestyle='--', linewidth=2.5, label='FiLM Model Recon.')
        
        ax.set_title(f'Patient {i+1} Standard Reconstruction', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('Concentration (ng/mL)')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('results/Base_vs_FiLM_Reconstruction.png', dpi=300)
    print("Saved comparative plot to results/Base_vs_FiLM_Reconstruction.png")

if __name__ == '__main__':
    print(f"Loading testing dataset from exp_run_all/{args.exp}/virtual_cohort_film_test.csv...")
    data_dict_test, scaler_info = extract_gen_tac_film(file_path=[f"exp_run_all/{args.exp}/virtual_cohort_film_test.csv"])
    test_dataset = TacroFilmDataset(data_dict_test)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: collate_fn_tacro_film(x, device))
    
    # ---------------------------------------------------------
    # 1. Load BASE Model
    # ---------------------------------------------------------
    print(f"Loading BASE model from {args.load_base}...")
    args.use_film = False
    z0_prior = torch.distributions.Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))
    model_base = create_LatentODE_model(args, input_dim=1, z0_prior=z0_prior, obsrv_std=torch.Tensor([0.01]).to(device), device=device)
    utils.get_ckpt_model(args.load_base, model_base, device)
    model_base.eval()

    # ---------------------------------------------------------
    # 2. Load FILM Model
    # ---------------------------------------------------------
    print(f"Loading FILM model from {args.load_film}...")
    args.use_film = True
    model_film = create_LatentODE_model(args, input_dim=1, z0_prior=z0_prior, obsrv_std=torch.Tensor([0.01]).to(device), device=device)
    utils.get_ckpt_model(args.load_film, model_film, device)
    model_film.eval()

    all_true_auc, all_base_auc, all_film_auc = [], [], []

    print("Evaluating Standard Reconstruction (Visit 1) on both models...")
    with torch.no_grad():
        for batch in test_dataloader:
            data_obs = batch["observed_data_v1"]
            tp_obs = batch["observed_tp_v1"]
            tp_pred = batch["tp_to_predict_v1"]
            dose = batch["dose_v1"]
            static = batch["static_v1"]
            
            # --- Get True AUC (Reference) directly from dataset ---
            # Extract auc_red and apply the scaling factor 
            dataset_idx = batch["patient_ids"] # Assuming this maps to original_indices
            # Actually, depending on your dataset class, you might need to pass auc_red through collate_fn.
            # Assuming you add "auc_red_v1" to collate_fn_tacro_film:
            auc_red = batch["auc_red_v1"] 
            
            # Unscale the ground truth AUC
            if scaler_info is not None:
                scaling_factor = scaler_info[0] # max_out
                auc_red_unscaled = auc_red.cpu().numpy() * scaling_factor
            else:
                auc_red_unscaled = auc_red.cpu().numpy()

            # --- Generate dense 100-point time grid for accurate integration ---
            dense_tp = utils.linspace_vector(tp_pred[0], torch.tensor(24.0), 100).to(device)

            # --- Model Predictions (100 samples averaged) ---
            pred_base, _ = model_base.get_reconstruction(dense_tp, data_obs, tp_obs, mask=None, dose=dose, static=static, n_traj_samples=100)
            pred_film, _ = model_film.get_reconstruction(dense_tp, data_obs, tp_obs, mask=None, dose=dose, static=static, n_traj_samples=100)

            # Average over the 100 latent trajectories and squeeze
            pred_base = pred_base.mean(dim=0).squeeze(-1) 
            pred_film = pred_film.mean(dim=0).squeeze(-1)

            # Unscale the predictions
            pred_base_unscaled = unscale_data(pred_base, scaler_info)
            pred_film_unscaled = unscale_data(pred_film, scaler_info)
            
            # Note: in test_model.py you zero out the tail for Prograf (12h) 
            # so that np.trapz over 24h naturally stops accumulating at 12h.
            # We do the exact same thing here:
            is_prograf = static[:, 1].bool()
            pred_base_unscaled[is_prograf, 50:] = 0.0
            pred_film_unscaled[is_prograf, 50:] = 0.0

            dense_tp_np = dense_tp.cpu().numpy()

            # Calculate Predicted AUCs
            b_aucs = np.array([np.trapz(p, dense_tp_np) for p in pred_base_unscaled])
            f_aucs = np.array([np.trapz(p, dense_tp_np) for p in pred_film_unscaled])
            
            all_true_auc.extend(auc_red_unscaled)
            all_base_auc.extend(b_aucs)
            all_film_auc.extend(f_aucs)

    # --- Compute Final Metrics Exactly like plot_auc ---
    all_true_auc = np.array(all_true_auc)
    all_base_auc = np.array(all_base_auc)
    all_film_auc = np.array(all_film_auc)

    # MPE (Bias) - No absolute values
    mpe_base = np.mean((all_true_auc - all_base_auc) / all_true_auc)
    mpe_film = np.mean((all_true_auc - all_film_auc) / all_true_auc)

    # RMSPE (Precision)
    rmspe_base = np.sqrt(np.mean(((all_true_auc - all_base_auc) / all_true_auc)**2))
    rmspe_film = np.sqrt(np.mean(((all_true_auc - all_film_auc) / all_true_auc)**2))

    print("\n" + "="*65)
    print(" SANITY CHECK: STANDARD RECONSTRUCTION (NO EXTRAPOLATION) ")
    print("="*65)
    print("1. BASE MODEL")
    print(f"   - MPE (Bias)          : {mpe_base * 100:.2f}%")
    print(f"   - RMSPE (Precision)   : {rmspe_base * 100:.2f}%")
    print("-" * 65)
    print("2. FILM MODEL (Standard Pipeline, FiLM Bypassed)")
    print(f"   - MPE (Bias)          : {mpe_film * 100:.2f}%")
    print(f"   - RMSPE (Precision)   : {rmspe_film * 100:.2f}%")
    print("="*65)
    
    # diff = ((rmse_film - rmse_base) / rmse_base) * 100
    # if diff > 0:
    #     print(f"Conclusion: FiLM model reconstruction is {abs(diff):.2f}% WORSE than Base model.")
    # else:
    #     print(f"Conclusion: FiLM model reconstruction is {abs(diff):.2f}% BETTER than Base model.")