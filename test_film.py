###########################
# Latent ODEs for Irregularly-Sampled Time Series
# FiLM Extrapolation Testing Script
###########################

import os
import sys
import matplotlib
matplotlib.use('Agg') # Safe for headless environments
import matplotlib.pyplot as plt
import argparse
import numpy as np
from scipy.special import inv_boxcox
from torch.utils.data import DataLoader
import random
import torch

import lib.utils as utils
from lib.create_latent_ode_model import create_LatentODE_model
from lib.read_tacro import extract_gen_tac_film, TacroFilmDataset, collate_fn_tacro_film, extract_pccp_tac_film
from lib.utils import get_device

# =========================================================================
# EXACT ARGUMENTS FROM test_model.py TO ENSURE SAFE MODEL RECONSTRUCTION
# =========================================================================
parser = argparse.ArgumentParser('Latent ODE FiLM Testing')
parser.add_argument('-n',  type=int, default=100, help="Size of the dataset")
parser.add_argument('--lr',  type=float, default=1e-2, help="Starting learning rate.")
parser.add_argument('-b', '--batch-size', type=int, default=2000)
parser.add_argument('--viz', action='store_true', help="Show plots while training")
parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--experiment', type = str, default = None, help="Fix experiment number for reproducibility")
parser.add_argument('--load', type=str, default=None, help="ID of the experiment to load for evaluation.")
parser.add_argument('-r', '--random-seed', type=int, default=1991, help="Random_seed")
parser.add_argument('--dataset', type=str, default='periodic', help="Dataset to load.")
parser.add_argument('-s', '--sample-tp', type=float, default=None)
parser.add_argument('-c', '--cut-tp', type=int, default=None)
parser.add_argument('--quantization', type=float, default=0.1)
parser.add_argument('--latent-ode', action='store_true', help="Run Latent ODE seq2seq model")
parser.add_argument('--use_gmm', action='store_true', help="Run Latent ODE with clusters inside latent space model")
parser.add_argument('--use_flow', action='store_true', help="Run Latent ODE with flow inside latent space model")
parser.add_argument('--use_gmm_v', action='store_true')
parser.add_argument('--z0-encoder', type=str, default='odernn', help="Type of encoder for Latent ODE model: odernn or rnn")
parser.add_argument('--classic-rnn', action='store_true')
parser.add_argument('--rnn-cell', default="gru")
parser.add_argument('--input-decay', action='store_true')
parser.add_argument('--ode-rnn', action='store_true')
parser.add_argument('--rnn-vae', action='store_true')
parser.add_argument('-l', '--latents', type=int, default=6, help="Size of the latent state")
parser.add_argument('-nc', '--n_components', type=int, default=4, help="Number of Gaussian within the latent space")
parser.add_argument('--rec-dims', type=int, default=20, help="Dimensionality of the recognition model (ODE or RNN).")
parser.add_argument('--rec-layers', type=int, default=1)
parser.add_argument('--gen-layers', type=int, default=1)
parser.add_argument('-u', '--units', type=int, default=100)
parser.add_argument('-g', '--gru-units', type=int, default=100)
parser.add_argument('--poisson', action='store_true')
parser.add_argument('--classif', action='store_true')
parser.add_argument('--linear-classif', action='store_true')
parser.add_argument('--extrap', action='store_true', help="Set extrapolation mode.")
parser.add_argument('-t', '--timepoints', type=int, default=100)
parser.add_argument('--max-t',  type=float, default=5.)
parser.add_argument('--noise-weight', type=float, default=0.01)
parser.add_argument('--seed', type = int, default = 15)
parser.add_argument('--exp', type=str, default='exp_film_run', help="Folder name for the FiLM dataset (inside exp_run_all/)")
parser.add_argument('--use_time', action='store_true', help="Run FiLM + Time Extrapolation training")
args = parser.parse_args()

# Force the film flag for internal logic
args.use_film = True 

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
        data_np = np.nan_to_num(data_np, nan=0.0) # Prevent NaN from negative inputs
        
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

if __name__ == '__main__':
    if args.load is None:
        print("Please provide a model checkpoint using the --load argument.")
        sys.exit(1)

    print(f"Loading testing dataset from exp_run_all/{args.exp}/virtual_cohort_film_test.csv...")
    
    # 1. Load Data
    data_dict_test, scaler_info = extract_gen_tac_film(
        file_path=[f"exp_run_all/{args.exp}/virtual_cohort_film_test.csv"]
    )
    
    # full_data_dict, scaler_info = extract_pccp_tac_film(plot=False)
        
    # 2. Get all patient/pair IDs and shuffle them randomly
    # all_ids = list(full_data_dict.keys())
    # # Set a seed if you want reproducible splits: random.seed(42)
    # random.seed(42)
    # random.shuffle(all_ids) 
    
    # # 3. Calculate the split index (80%)
    # split_idx = int(len(all_ids) * 0.8)
    # # test_ids = all_ids[:split_idx]
    # test_ids = all_ids
    # data_dict_test = {pid: full_data_dict[pid] for pid in test_ids}

    test_dataset = TacroFilmDataset(data_dict_test)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                                 collate_fn=lambda x: collate_fn_tacro_film(x, device))
    
    # 2. Create and Load Model
    print("Reconstructing model architecture...")
    z0_prior = torch.distributions.Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))
    obsrv_std = torch.Tensor([0.01]).to(device)
    
    model = create_LatentODE_model(args, input_dim=1, z0_prior=z0_prior, obsrv_std=obsrv_std, device=device)
    utils.get_ckpt_model(args.load, model, device)
    model.eval()
    
    all_true_auc_v1, all_pred_auc_v1 = [], []
    all_true_auc_v2, all_pred_auc_v2 = [], []

    print("Evaluating FiLM Extrapolation Model...")
    with torch.no_grad():
        for batch in test_dataloader:
            batch_dict = {
                "observed_data_v1": batch["observed_data_v1"],
                "observed_tp_v1": batch["observed_tp_v1"],
                "data_to_predict_v1": batch["data_to_predict_v1"],
                "tp_to_predict_v1": batch["tp_to_predict_v1"],
                "data_to_predict_v2": batch["data_to_predict_v2"],
                "tp_to_predict_v2": batch["tp_to_predict_v2"],
                "dose_v1": batch["dose_v1"],
                "others_v1" : batch["others_v1"],
                "dose_v2": batch["dose_v2"],
                "static_v1": batch["static_v1"],
                "auc_red_v1": batch["auc_red_v1"],
                "auc_red_v2": batch["auc_red_v2"],
                "delta_t": batch["delta_t"],
                "t_v1" : batch['t_v1'],
            }
            # FIX 1: Create a dense 100-point 24h timeline
            dense_tp = utils.linspace_vector(batch_dict["tp_to_predict_v1"][0], torch.tensor(24.0), 100).to(device)
            
            # FIX 2: Forward pass using dense_tp and 100 trajectories
            pred_v2, info = model.get_reconstruction_extrapolation(
                data_v1=batch_dict["observed_data_v1"],
                time_steps_v1=batch_dict["observed_tp_v1"],
                time_steps_v2=dense_tp,             # Pass dense timeline for V2
                dose_v1=batch_dict["dose_v1"],
                dose_v2=batch_dict["dose_v2"],
                time_steps_to_predict_v1=dense_tp,  # Pass dense timeline for V1
                static_v1=batch_dict["static_v1"],
                delta_t= batch_dict['delta_t'],
                t_v1 = batch_dict['t_v1'],
                n_traj_samples=100                  # Match test_model.py stochasticity
            )
            
            # FIX 3: Squeeze and average the 100 samples
            pred_v1 = info["pred_x_v1"].mean(dim=0).squeeze(-1)
            pred_v2 = pred_v2.mean(dim=0).squeeze(-1)
            
            # Extract true sparse data
            data_true_v1 = batch_dict["data_to_predict_v1"].squeeze(-1)
            data_true_v2 = batch_dict["data_to_predict_v2"].squeeze(-1)

            # Unscale Data
            true_v1_unscaled = unscale_data(data_true_v1, scaler_info)
            pred_v1_unscaled = unscale_data(pred_v1, scaler_info)
            true_v2_unscaled = unscale_data(data_true_v2, scaler_info)
            pred_v2_unscaled = unscale_data(pred_v2, scaler_info)
            
            # Numpy conversion
            dense_tp_np = dense_tp.cpu().numpy()
            tp_true_v1 = batch_dict["tp_to_predict_v1"].cpu().numpy()
            tp_true_v2 = batch_dict["tp_to_predict_v2"].cpu().numpy()
            static_treatments = batch_dict["static_v1"]

            # FIX 4: Zero out the tails for Prograf (12h formulation) on the dense predictions
            is_prograf = static_treatments[:, 1].bool()
            pred_v1_unscaled[is_prograf, 50:] = 0.0
            pred_v2_unscaled[is_prograf, 50:] = 0.0

            # Calculate True AUCs (from sparse data)
            t_aucs_v1 = batch_dict["auc_red_v1"]*scaler_info[0]
            t_aucs_v2 = batch_dict["auc_red_v2"]*scaler_info[0]
            
            # Calculate Predicted AUCs (from dense predictions)
            p_aucs_v1 = calculate_auc(pred_v1_unscaled, dense_tp_np, static_treatments)
            p_aucs_v2 = calculate_auc(pred_v2_unscaled, dense_tp_np, static_treatments)
            
            # Store
            all_true_auc_v1.extend(t_aucs_v1)
            all_pred_auc_v1.extend(p_aucs_v1)
            all_true_auc_v2.extend(t_aucs_v2)
            all_pred_auc_v2.extend(p_aucs_v2)

            # Plotting
            if args.viz:
                # Check if model has FiLM enabled before running
                from lib.plot_ot import plot_optimal_transport_trajectory, plot_dose_fan_trajectory
                from remove_dose import test_orthogonal_disentanglement
                test_orthogonal_disentanglement(model, batch_dict)
                plot_dose_fan_trajectory(model, batch_dict, patient_indices=[0, 1, 2,3,4,5, 6, 7, 8, 9 ,10, 11, 12], num_steps=20)
                plot_optimal_transport_trajectory(model, batch_dict, patient_idx=0, num_steps=30) 
                from various_plot import quantify_manifold_curvature, plot_dose_vector_field, plot_umap_zbio
                plot_umap_zbio(model, batch_dict, num_steps=20)
                plot_dose_vector_field(model, batch_dict, dose_step=5.0)
                quantify_manifold_curvature(model, batch_dict)
                # try:
                from lib.plotting import Visualizations
                viz = Visualizations(device)
                plot_name = f"test_film_extrapolation.png"
                print(f"Saving visualization to results/{plot_name}...")
                if isinstance(scaler_info, list):
                    scaler_dict = {'max_out': scaler_info[0]}
                    if len(scaler_info) > 1:
                        scaler_dict['best_lambda'] = scaler_info[1]
                elif not isinstance(scaler_info, dict):
                    scaler_dict = {'max_out': scaler_info} # Handles the float max_out case
                else:
                    scaler_dict = scaler_info
                
                viz.draw_all_plots_film(batch_dict, model, plot_name=plot_name, experimentID="results", save=True, scaler=scaler_dict)
                # except Exception as e:
                #     print(f"Could not generate plot: {e}")
                args.viz = False # Only plot once
    # --- Compute Final Metrics ---
    
    all_true_auc_v1 = np.array(all_true_auc_v1)
    all_pred_auc_v1 = np.array(all_pred_auc_v1)
    all_true_auc_v2 = np.array(all_true_auc_v2)
    all_pred_auc_v2 = np.array(all_pred_auc_v2)
    
    # FIX 5: Calculate MPE (Bias) and RMSPE (Precision) exactly like test_model.py
    # MPE (Bias) - No absolute values
    mpe_v1 = np.mean((all_true_auc_v1 - all_pred_auc_v1) / all_true_auc_v1)
    mpe_v2 = np.mean((all_true_auc_v2 - all_pred_auc_v2) / all_true_auc_v2)
    
    # RMSPE (Precision)
    rmspe_v1 = np.sqrt(np.mean(((all_true_auc_v1 - all_pred_auc_v1) / all_true_auc_v1)**2))
    rmspe_v2 = np.sqrt(np.mean(((all_true_auc_v2 - all_pred_auc_v2) / all_true_auc_v2)**2))

    print("\n" + "="*50)
    print("FINAL FILM EXTRAPOLATION RESULTS")
    print("="*50)
    print("VISIT 1 (Base Reconstruction Context):")
    print(f"  - MPE (Bias)          : {mpe_v1 * 100:.2f}%")
    print(f"  - RMSPE (Precision)   : {rmspe_v1 * 100:.2f}%")
    print("-" * 50)
    print("VISIT 2 (Zero-Shot Dose Extrapolation):")
    print(f"  - MPE (Bias)          : {mpe_v2 * 100:.2f}%")
    print(f"  - RMSPE (Precision)   : {rmspe_v2 * 100:.2f}%")
    print("="*50)