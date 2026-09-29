import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
import matplotlib.lines as mlines
import seaborn as sns

# ==========================================
# 1. NATURE M.I. AESTHETIC CONFIGURATION
# ==========================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,          # Small legend to prevent clutter
    "axes.linewidth": 0.8,         # Thinner axis lines
    "axes.spines.top": False,      # Remove top bounding box
    "axes.spines.right": False,    # Remove right bounding box
    "lines.linewidth": 1.2,
    "pdf.fonttype": 42,            # Ensures fonts are exported as text, not paths
    "ps.fonttype": 42
})

def plot_optimal_transport_trajectory(model, data_dict, patient_idx=0, num_steps=20, ax=None):
    """
    Visualizes the Optimal Transport trajectory of a single patient's latent state.
    Refactored to accept a matplotlib axis (ax) for multi-panel plotting.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
    else:
        fig = ax.figure

    model.eval()
    device = data_dict["observed_data_v1"].device
    
    with torch.no_grad():
        data_v1 = data_dict["observed_data_v1"][patient_idx:patient_idx+1]
        tp_v1 = data_dict["observed_tp_v1"]
        dose_v1 = data_dict["dose_v1"][patient_idx:patient_idx+1]
        
        static_v1 = data_dict.get("static_v1", None)
        if static_v1 is not None:
            static_v1 = static_v1[patient_idx:patient_idx+1]

        seq_len = data_v1.size(1)
        dose_expanded = dose_v1.view(-1, 1, 1).expand(-1, seq_len, 1)
        truth_w_mask = torch.cat((data_v1, dose_expanded), dim=-1)

        first_point_mu, _ = model.encoder_z0(truth_w_mask, tp_v1, static=static_v1, run_backwards=True)
        z0_old = first_point_mu 

        max_dose = max(data_dict["dose_v1"].max().item(), data_dict["dose_v2"].max().item()) * 1.5
        max_dose = max_dose if max_dose > 0 else 10.0
        
        dose_v2_range = torch.linspace(0, max_dose, num_steps).to(device)

        z0_trajectory, doses_used = [], []

        for dose_v2 in dose_v2_range:
            dose_v2_tensor = dose_v2.view(1) 
            c_doses = torch.stack([dose_v1, dose_v2_tensor], dim=1).to(device)
            c = torch.cat([c_doses.unsqueeze(0), z0_old], dim=-1)
            
            gamma = model.film_gamma(c)
            beta = model.film_beta(c)
            z0_new = (z0_old * gamma) + beta
            
            z0_trajectory.append(z0_new.squeeze().cpu().numpy())
            doses_used.append(dose_v2.item())

        z0_pca = PCA(n_components=2).fit_transform(np.array(z0_trajectory))
        pca_var = PCA(n_components=2).fit(np.array(z0_trajectory)).explained_variance_ratio_ * 100

        # Plotting
        ax.plot(z0_pca[:, 0], z0_pca[:, 1], color='gray', linestyle='--', alpha=0.5, zorder=1) 
        scatter = ax.scatter(z0_pca[:, 0], z0_pca[:, 1], c=doses_used, cmap='viridis', 
                             s=30, edgecolor='white', linewidth=0.5, zorder=2)
        
        orig_idx = np.argmin(np.abs(np.array(doses_used) - dose_v1.item()))
        ax.scatter(z0_pca[orig_idx, 0], z0_pca[orig_idx, 1], c='#E76F51', marker='*', 
                   s=120, edgecolor='white', linewidth=0.8, zorder=3,
                   label=f'Baseline V1 (Dose: {dose_v1.item():.2f})')

        ax.set_title(f'Linear Latent Trajectory\n(Patient {patient_idx})', pad=10)
        ax.set_xlabel(f'PCA 1 ({pca_var[0]:.1f}% Var)')
        ax.set_ylabel(f'PCA 2 ({pca_var[1]:.1f}% Var)')
        
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Target Dose')
        ax.legend(frameon=False, loc='best')

def plot_dose_fan_trajectory(model, data_dict, patient_indices=[0, 1, 2, 3, 4], num_steps=20, ax=None):
    """
    Visualizes the Optimal Transport 'Dose Fan' in the latent space.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
    else:
        fig = ax.figure

    model.eval()
    device = data_dict["observed_data_v1"].device
    
    max_dose = max(data_dict["dose_v1"].max().item(), data_dict["dose_v2"].max().item()) * 1.5
    max_dose = max_dose if max_dose > 0 else 10.0
    dose_v2_range = torch.linspace(0, max_dose, num_steps).to(device)
    
    all_trajectories, original_doses = [], []
    
    with torch.no_grad():
        for patient_idx in patient_indices:
            data_v1 = data_dict["observed_data_v1"][patient_idx:patient_idx+1]
            tp_v1 = data_dict["observed_tp_v1"]
            dose_v1 = data_dict["dose_v1"][patient_idx:patient_idx+1]
            
            static_v1 = data_dict.get("static_v1", None)
            if static_v1 is not None:
                static_v1 = static_v1[patient_idx:patient_idx+1]

            seq_len = data_v1.size(1)
            dose_expanded = dose_v1.view(-1, 1, 1).expand(-1, seq_len, 1)
            truth_w_mask = torch.cat((data_v1, dose_expanded), dim=-1)

            first_point_mu, _ = model.encoder_z0(truth_w_mask, tp_v1, static=static_v1, run_backwards=True)
            z0_old = first_point_mu 
            
            patient_trajectory = []
            original_doses.append(dose_v1.item())

            for dose_v2 in dose_v2_range:
                c = torch.cat([torch.stack([dose_v1, dose_v2.view(1)], dim=1).unsqueeze(0), z0_old], dim=-1)
                z0_new = (z0_old * model.film_gamma(c)) + model.film_beta(c)
                patient_trajectory.append(z0_new.squeeze().cpu().numpy())
                
            all_trajectories.append(np.array(patient_trajectory))

    pca = PCA(n_components=2)
    pca.fit(np.vstack(all_trajectories))
    pca_var = pca.explained_variance_ratio_ * 100

    markers = ['o', 's', '^', 'D', 'v', 'p', 'X', 'h']
    doses_np = dose_v2_range.cpu().numpy()
    
    for i, patient_idx in enumerate(patient_indices):
        z0_pca = pca.transform(all_trajectories[i])
        ax.plot(z0_pca[:, 0], z0_pca[:, 1], color='gray', alpha=0.3, zorder=1)
        
        scatter = ax.scatter(z0_pca[:, 0], z0_pca[:, 1], c=doses_np, cmap='viridis', 
                             marker=markers[i % len(markers)], s=20, edgecolor='white', linewidth=0.3, zorder=2)
        
        orig_idx = np.argmin(np.abs(doses_np - original_doses[i]))
        ax.scatter(z0_pca[orig_idx, 0], z0_pca[orig_idx, 1], c='#E76F51', marker='*', 
                   s=80, edgecolor='white', linewidth=0.5, zorder=3)

    ax.set_title('Curved Extrapolation Manifold\n(Multiple Phenotypes)', pad=10)
    ax.set_xlabel(f'Global PCA 1 ({pca_var[0]:.1f}% Var)')
    ax.set_ylabel(f'Global PCA 2 ({pca_var[1]:.1f}% Var)')
    
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Target Dose')
    
    star_marker = mlines.Line2D([], [], color='#E76F51', marker='*', linestyle='None',
                                markersize=8, markeredgecolor='white', label='Baseline V1')
    ax.legend(handles=[star_marker], frameon=False, loc='best')

# ==========================================
# RUN THE GENERATION & ASSEMBLE FIGURE 6
# ==========================================
def generate_figure_6(model, data_dict):
    """
    Creates the combined Figure 6 (Panels a and b side-by-side)
    as expected by a journal.
    """
    print("\n--- Generating Publication-Ready Figure 6 ---")
    
    # 183mm is a standard 2-column width in Nature. In inches: ~7.2 inches.
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    
    # Plot Panel A
    plot_optimal_transport_trajectory(model, data_dict, patient_idx=0, num_steps=20, ax=axes[0])
    
    # Plot Panel B (selecting a few diverse patients for clarity)
    plot_dose_fan_trajectory(model, data_dict, patient_indices=[0, 1, 2, 5, 8, 12], num_steps=20, ax=axes[1])
    
    # Add panel labels (a) and (b)
    axes[0].text(-0.15, 1.1, 'a', transform=axes[0].transAxes, fontsize=11, fontweight='bold', va='top')
    axes[1].text(-0.15, 1.1, 'b', transform=axes[1].transAxes, fontsize=11, fontweight='bold', va='top')

    plt.tight_layout()
    
    # Save as high-res vector PDF
    plt.savefig('Figure_6_Manifold_Transport.pdf', format='pdf', dpi=300, bbox_inches='tight', transparent=True)
    print("Saved -> Figure_6_Manifold_Transport.pdf")

# NOTE: Simply call generate_figure_6(model, data_dict) to run this.