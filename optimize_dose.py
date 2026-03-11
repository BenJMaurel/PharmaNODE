###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Dose Optimization Script
#
# Given a trained FiLM model and a target AUC, this script finds the optimal
# dose for each patient by sweeping over candidate doses, integrating the
# modulated ODE trajectory, and selecting the dose whose predicted AUC is
# closest to the target. Results are plotted as predicted AUC vs. target AUC.
###########################

import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import numpy as np
from scipy.special import inv_boxcox
from torch.utils.data import DataLoader
import torch

import lib.utils as utils
from lib.create_latent_ode_model import create_LatentODE_model
from lib.read_tacro import (
    extract_gen_tac_film,
    TacroFilmDataset,
    collate_fn_tacro_film,
)
from lib.utils import get_device

# =========================================================================
# ARGUMENTS — mirrors test_film.py so that model reconstruction is identical
# =========================================================================
parser = argparse.ArgumentParser('Latent ODE — Dose Optimization')
parser.add_argument('-n',  type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('-b', '--batch-size', type=int, default=2000)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--save', type=str, default='experiments/')
parser.add_argument('--experiment', type=str, default=None)
parser.add_argument('--load', type=str, default=None,
                    help="Path to model checkpoint (required).")
parser.add_argument('-r', '--random-seed', type=int, default=1991)
parser.add_argument('--dataset', type=str, default='periodic')
parser.add_argument('-s', '--sample-tp', type=float, default=None)
parser.add_argument('-c', '--cut-tp', type=int, default=None)
parser.add_argument('--quantization', type=float, default=0.1)
parser.add_argument('--latent-ode', action='store_true')
parser.add_argument('--use_gmm', action='store_true')
parser.add_argument('--use_flow', action='store_true')
parser.add_argument('--use_gmm_v', action='store_true')
parser.add_argument('--z0-encoder', type=str, default='odernn')
parser.add_argument('--classic-rnn', action='store_true')
parser.add_argument('--rnn-cell', default='gru')
parser.add_argument('--input-decay', action='store_true')
parser.add_argument('--ode-rnn', action='store_true')
parser.add_argument('--rnn-vae', action='store_true')
parser.add_argument('-l', '--latents', type=int, default=6)
parser.add_argument('-nc', '--n_components', type=int, default=4)
parser.add_argument('--rec-dims', type=int, default=20)
parser.add_argument('--rec-layers', type=int, default=1)
parser.add_argument('--gen-layers', type=int, default=1)
parser.add_argument('-u', '--units', type=int, default=100)
parser.add_argument('-g', '--gru-units', type=int, default=100)
parser.add_argument('--poisson', action='store_true')
parser.add_argument('--classif', action='store_true')
parser.add_argument('--linear-classif', action='store_true')
parser.add_argument('--extrap', action='store_true')
parser.add_argument('-t', '--timepoints', type=int, default=100)
parser.add_argument('--max-t', type=float, default=5.)
parser.add_argument('--noise-weight', type=float, default=0.01)
parser.add_argument('--seed', type=int, default=15)
parser.add_argument('--exp', type=str, default='exp_film_run')
parser.add_argument('--use_time', action='store_true')

# =========================================================================
# DOSE OPTIMIZATION SPECIFIC ARGUMENTS
# =========================================================================
parser.add_argument('--target-auc', type=float, default=None,
                    help="Fixed target AUC for all patients (in ng/mL·h). "
                         "If None, the true Visit-2 AUC of each patient is used "
                         "as their individual target.")
parser.add_argument('--dose-min', type=float, default=0.05,
                    help="Lower bound of the candidate dose grid (mg).")
parser.add_argument('--dose-max', type=float, default=2.0,
                    help="Upper bound of the candidate dose grid (mg).")
parser.add_argument('--dose-steps', type=int, default=200,
                    help="Number of candidate doses to evaluate per patient.")
parser.add_argument('--n-traj', type=int, default=50,
                    help="Monte Carlo trajectory samples used to average predictions.")
parser.add_argument('--output-dir', type=str, default='results/',
                    help="Directory for output figures.")

args = parser.parse_args()
args.use_film = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
utils.makedirs(args.output_dir)


# =========================================================================
# HELPER FUNCTIONS
# =========================================================================

def unscale_data(data, scaler_info):
    """Reverses max-out and optional Box-Cox scaling."""
    data_np = data.detach().cpu().numpy() if torch.is_tensor(data) else np.asarray(data)
    if scaler_info is None:
        return data_np
    max_out = scaler_info[0]
    best_lambda = scaler_info[1] if len(scaler_info) > 1 else None
    if best_lambda is not None:
        data_np = inv_boxcox(data_np, best_lambda)
        data_np = np.nan_to_num(data_np, nan=0.0)
    return data_np * max_out


def calculate_auc_batch(concentrations_np, times_np, static_treatments):
    """
    Trapezoidal AUC per patient.
    Prograf (static[:, 1] == 1)  → integrate to 12 h.
    Advagraf (static[:, 1] == 0) → integrate to 24 h.
    """
    aucs = []
    for i in range(concentrations_np.shape[0]):
        is_prograf = bool(static_treatments[i, 1].item())
        cutoff = 12.0 if is_prograf else 24.0
        mask = times_np <= cutoff
        aucs.append(np.trapz(concentrations_np[i, mask], times_np[mask]))
    return np.array(aucs)


def predict_auc_for_dose(model, z0_encoded, dose_v1_tensor, dose_v2_scalar,
                         dense_tp, static_v1, scaler_info, n_traj,
                         batch_size, device):
    """
    Given a pre-encoded baseline state z0_encoded and a candidate target dose,
    apply the FiLM operator and integrate the ODE to obtain a predicted AUC.

    Parameters
    ----------
    z0_encoded : torch.Tensor  [1, 1, latent_dim] per-patient encoded state
    dose_v1_tensor : torch.Tensor  [batch] baseline doses
    dose_v2_scalar : float  candidate target dose
    dense_tp : torch.Tensor  [T] dense time grid for integration

    Returns
    -------
    np.ndarray  [batch] predicted AUCs for this candidate dose
    """
    dose_v2_tensor = torch.full_like(dose_v1_tensor, dose_v2_scalar)

    # Build context vector and apply FiLM
    c_doses = torch.stack([dose_v1_tensor, dose_v2_tensor], dim=-1).unsqueeze(0)
    c = torch.cat([c_doses, z0_encoded], dim=-1)

    gamma = model.film_gamma(c)
    beta  = model.film_beta(c)
    z0_modulated = z0_encoded * gamma + beta  # [1, batch, latent_dim]

    # Integrate ODE from modulated initial condition
    # model.diffeq_solver expects [n_traj, batch, latent_dim]
    z0_rep = z0_modulated.expand(n_traj, -1, -1)
    pred_z = model.diffeq_solver(z0_rep, dense_tp)          # [n_traj, batch, T, latent_dim]
    pred_x = model.decoder(pred_z).squeeze(-1)              # [n_traj, batch, T]
    pred_x = pred_x.mean(dim=0)                             # [batch, T]

    # Unscale
    pred_np = unscale_data(pred_x, scaler_info)

    # Zero out Prograf tails beyond 12 h
    is_prograf = static_v1[:, 1].bool().cpu().numpy()
    midpoint   = dense_tp.shape[0] // 2
    pred_np[is_prograf, midpoint:] = 0.0

    # AUC
    dense_np = dense_tp.cpu().numpy()
    return calculate_auc_batch(pred_np, dense_np, static_v1)


# =========================================================================
# MAIN
# =========================================================================

if __name__ == '__main__':

    if args.load is None:
        print("Please provide a model checkpoint via --load.")
        sys.exit(1)

    print(f"Loading test dataset from exp_run_all/{args.exp}/virtual_cohort_film_test.csv ...")
    data_dict_test, scaler_info = extract_gen_tac_film(
        file_path=[f"exp_run_all/{args.exp}/virtual_cohort_film_test.csv"]
    )

    test_dataset   = TacroFilmDataset(data_dict_test)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_fn_tacro_film(x, device)
    )

    # ---- Reconstruct model -------------------------------------------------
    print("Reconstructing model architecture ...")
    z0_prior  = torch.distributions.Normal(
        torch.Tensor([0.0]).to(device),
        torch.Tensor([1.0]).to(device)
    )
    obsrv_std = torch.Tensor([0.01]).to(device)
    model = create_LatentODE_model(
        args, input_dim=1,
        z0_prior=z0_prior,
        obsrv_std=obsrv_std,
        device=device
    )
    utils.get_ckpt_model(args.load, model, device)
    model.eval()

    # ---- Candidate dose grid -----------------------------------------------
    dose_grid = np.linspace(args.dose_min, args.dose_max, args.dose_steps)
    print(f"Dose grid: {args.dose_steps} candidates from "
          f"{args.dose_min} to {args.dose_max} mg")

    # Accumulators
    all_target_auc    = []   # individual target AUC per patient
    all_optimal_dose  = []   # dose selected by the model
    all_predicted_auc = []   # AUC achieved at the selected dose
    all_true_auc_v2   = []   # ground-truth Visit-2 AUC (reference)

    print("Running dose optimisation ...")
    with torch.no_grad():
        for batch in test_dataloader:

            data_v1    = batch["observed_data_v1"]    # [batch, T_obs, 1]
            tp_v1      = batch["observed_tp_v1"]       # [T_obs]
            dose_v1    = batch["dose_v1"]              # [batch]
            static_v1  = batch["static_v1"]            # [batch, n_static]
            auc_red_v1 = batch["auc_red_v1"]           # [batch]  (scaled)
            auc_red_v2 = batch["auc_red_v2"]           # [batch]  (scaled)

            batch_n   = data_v1.size(0)
            dense_tp  = utils.linspace_vector(
                tp_v1[0], torch.tensor(24.0), 100
            ).to(device)

            # ---- Encode baseline state z0 ----------------------------------
            seq_len      = data_v1.size(1)
            dose_exp     = dose_v1.view(-1, 1, 1).expand(-1, seq_len, 1)
            truth_w_mask = torch.cat([data_v1, dose_exp], dim=-1)

            # z0_encoded: [1, batch, latent_dim]
            z0_encoded, _ = model.encoder_z0(
                truth_w_mask, tp_v1,
                static=static_v1,
                run_backwards=True
            )

            # ---- Per-patient target AUC ------------------------------------
            if args.target_auc is not None:
                # Fixed clinical target (e.g. 200 ng/mL·h) for all patients
                target_aucs = np.full(batch_n, args.target_auc)
            else:
                # Use each patient's true Visit-2 AUC as their individual target
                target_aucs = (auc_red_v2 * scaler_info[0]).cpu().numpy()

            # ---- Grid search over candidate doses --------------------------
            # auc_matrix: [dose_steps, batch]
            auc_matrix = np.zeros((args.dose_steps, batch_n))

            for d_idx, dose_candidate in enumerate(dose_grid):
                auc_matrix[d_idx] = predict_auc_for_dose(
                    model        = model,
                    z0_encoded   = z0_encoded,
                    dose_v1_tensor = dose_v1,
                    dose_v2_scalar = dose_candidate,
                    dense_tp     = dense_tp,
                    static_v1    = static_v1,
                    scaler_info  = scaler_info,
                    n_traj       = args.n_traj,
                    batch_size   = batch_n,
                    device       = device
                )

            # ---- Select dose closest to target for each patient ------------
            # distance matrix: [dose_steps, batch]
            dist_matrix  = np.abs(auc_matrix - target_aucs[np.newaxis, :])
            best_dose_idx = dist_matrix.argmin(axis=0)           # [batch]

            optimal_doses  = dose_grid[best_dose_idx]            # [batch]
            predicted_aucs = auc_matrix[best_dose_idx,
                                        np.arange(batch_n)]       # [batch]
            true_aucs_v2   = (auc_red_v2 * scaler_info[0]).cpu().numpy()

            all_target_auc.extend(target_aucs.tolist())
            all_optimal_dose.extend(optimal_doses.tolist())
            all_predicted_auc.extend(predicted_aucs.tolist())
            all_true_auc_v2.extend(true_aucs_v2.tolist())

    # =========================================================================
    # RESULTS
    # =========================================================================
    all_target_auc    = np.array(all_target_auc)
    all_optimal_dose  = np.array(all_optimal_dose)
    all_predicted_auc = np.array(all_predicted_auc)
    all_true_auc_v2   = np.array(all_true_auc_v2)

    # Relative error between predicted AUC (at optimal dose) and target AUC
    rel_error = (all_predicted_auc - all_target_auc) / all_target_auc * 100
    mpe   = np.mean(rel_error)
    rmspe = np.sqrt(np.mean(rel_error ** 2))

    print("\n" + "=" * 55)
    print("DOSE OPTIMISATION RESULTS")
    print("=" * 55)
    print(f"  Patients evaluated   : {len(all_target_auc)}")
    print(f"  Dose range           : {args.dose_min}–{args.dose_max} mg "
          f"({args.dose_steps} steps)")
    print(f"  Target AUC mode      : "
          f"{'fixed = ' + str(args.target_auc) + ' ng/mL·h' if args.target_auc else 'individual (true V2 AUC)'}")
    print("-" * 55)
    print(f"  MPE   (bias)         : {mpe:.2f}%")
    print(f"  RMSPE (precision)    : {rmspe:.2f}%")
    print("=" * 55)

    # =========================================================================
    # FIGURE 1 — Predicted AUC at optimal dose vs. target AUC
    # =========================================================================
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(
        all_target_auc, all_predicted_auc,
        color='steelblue', alpha=0.7, edgecolors='white',
        linewidths=0.4, s=55, zorder=3,
        label='Patient (predicted vs. target)'
    )

    # Identity line (perfect dose selection)
    lims = [
        min(all_target_auc.min(), all_predicted_auc.min()) * 0.9,
        max(all_target_auc.max(), all_predicted_auc.max()) * 1.1,
    ]
    ax.plot(lims, lims, 'k--', linewidth=1.2, label='Identity (target = predicted)')

    # ±20% acceptance band
    ax.fill_between(
        lims,
        [l * 0.80 for l in lims],
        [l * 1.20 for l in lims],
        color='steelblue', alpha=0.10, label='±20% acceptance band'
    )

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Target AUC (ng/mL·h)', fontsize=12)
    ax.set_ylabel('Predicted AUC at Optimal Dose (ng/mL·h)', fontsize=12)
    ax.set_title(
        f'Model-Guided Dose Optimisation\n'
        f'RMSPE = {rmspe:.1f}%  |  MPE = {mpe:+.1f}%',
        fontsize=12
    )
    ax.legend(fontsize=9)
    ax.set_aspect('equal')
    plt.tight_layout()

    path1 = os.path.join(args.output_dir, 'dose_optimisation_scatter.png')
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    print(f"Saved: {path1}")

    # =========================================================================
    # FIGURE 2 — Distribution of selected optimal doses
    # =========================================================================
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(
        all_optimal_dose, bins=40,
        color='steelblue', alpha=0.75, edgecolor='white'
    )
    ax.axvline(
        np.median(all_optimal_dose), color='black',
        linestyle='--', linewidth=1.2,
        label=f'Median dose: {np.median(all_optimal_dose):.1f} mg'
    )
    ax.set_xlabel('Optimal Dose (mg)', fontsize=12)
    ax.set_ylabel('Number of Patients', fontsize=12)
    ax.set_title('Distribution of Model-Selected Optimal Doses', fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()

    path2 = os.path.join(args.output_dir, 'dose_optimisation_histogram.png')
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    print(f"Saved: {path2}")

    # =========================================================================
    # FIGURE 3 — AUC dose–response curve for a sample of individual patients
    # =========================================================================
    # Re-run the grid for a small patient subset to plot dose–response curves
    N_PATIENTS_TO_PLOT = min(6, len(all_optimal_dose))

    with torch.no_grad():
        # Use the first batch only for illustration
        batch = next(iter(test_dataloader))

        data_v1   = batch["observed_data_v1"]
        tp_v1     = batch["observed_tp_v1"]
        dose_v1   = batch["dose_v1"]
        static_v1 = batch["static_v1"]
        auc_red_v2 = batch["auc_red_v2"]

        dense_tp = utils.linspace_vector(
            tp_v1[0], torch.tensor(24.0), 100
        ).to(device)

        seq_len = data_v1.size(1)
        dose_exp = dose_v1.view(-1, 1, 1).expand(-1, seq_len, 1)
        truth_w_mask = torch.cat([data_v1, dose_exp], dim=-1)
        z0_encoded, _ = model.encoder_z0(
            truth_w_mask, tp_v1,
            static=static_v1, run_backwards=True
        )

        # Dose–response matrix for plotting subset
        subset_idx = list(range(N_PATIENTS_TO_PLOT))
        auc_curves = np.zeros((N_PATIENTS_TO_PLOT, args.dose_steps))

        for d_idx, dose_candidate in enumerate(dose_grid):
            batch_aucs = predict_auc_for_dose(
                model=model,
                z0_encoded=z0_encoded,
                dose_v1_tensor=dose_v1,
                dose_v2_scalar=dose_candidate,
                dense_tp=dense_tp,
                static_v1=static_v1,
                scaler_info=scaler_info,
                n_traj=args.n_traj,
                batch_size=data_v1.size(0),
                device=device
            )
            for p, pi in enumerate(subset_idx):
                auc_curves[p, d_idx] = batch_aucs[pi]

        target_aucs_subset = (auc_red_v2 * scaler_info[0]).cpu().numpy()

    fig, axes = plt.subplots(
        2, 3, figsize=(13, 8),
        sharex=False, sharey=False
    )
    axes = axes.flatten()

    for p, pi in enumerate(subset_idx):
        ax = axes[p]
        target = target_aucs_subset[pi]

        ax.plot(dose_grid, auc_curves[p], color='steelblue',
                linewidth=1.8, label='Predicted AUC')
        ax.axhline(target, color='crimson', linestyle='--',
                   linewidth=1.4, label=f'Target AUC\n({target:.1f} ng/mL·h)')

        # Mark the selected dose
        best_d_idx = np.argmin(np.abs(auc_curves[p] - target))
        best_d     = dose_grid[best_d_idx]
        best_auc   = auc_curves[p, best_d_idx]
        ax.axvline(best_d, color='black', linestyle=':', linewidth=1.2)
        ax.scatter([best_d], [best_auc], color='black', zorder=5, s=40,
                   label=f'Optimal dose\n({best_d:.1f} mg)')

        ax.set_xlabel('Dose (mg)', fontsize=9)
        ax.set_ylabel('Predicted AUC\n(ng/mL·h)', fontsize=9)
        ax.set_title(f'Patient {pi + 1}', fontsize=10)
        ax.legend(fontsize=7, loc='lower right')

    plt.suptitle(
        'Dose–Response Curves for Individual Patients\n'
        '(Dashed red = clinical target, dotted black = model-selected dose)',
        fontsize=11, y=1.01
    )
    plt.tight_layout()

    path3 = os.path.join(args.output_dir, 'dose_response_curves.png')
    fig.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path3}")

    print("\nDone. All outputs written to:", args.output_dir)