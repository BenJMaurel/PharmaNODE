import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

def plot_optimal_transport_trajectory(model, data_dict, patient_idx=0, num_steps=20):
    """
    Visualizes the Optimal Transport trajectory of a single patient's latent state
    as the dose is continuously varied, adapted for the FiLM v1 -> v2 architecture.
    """
    print(f"\n--- Generating Optimal Transport Trajectory for Patient {patient_idx} ---")
    model.eval()
    
    # Use v1 keys since this is the source state for FiLM
    device = data_dict["observed_data_v1"].device
    
    with torch.no_grad():
        # 1. Extract the specific patient's data for Visit 1 (Source state)
        data_v1 = data_dict["observed_data_v1"][patient_idx:patient_idx+1]
        tp_v1 = data_dict["observed_tp_v1"]
        dose_v1 = data_dict["dose_v1"][patient_idx:patient_idx+1]
        
        static_v1 = None
        if "static_v1" in data_dict and data_dict["static_v1"] is not None:
            static_v1 = data_dict["static_v1"][patient_idx:patient_idx+1]

        # 2. Encode to get the initial latent state (z0_old)
        seq_len = data_v1.size(1)
        dose_expanded = dose_v1.view(-1, 1, 1).expand(-1, seq_len, 1)
        truth_w_mask = torch.cat((data_v1, dose_expanded), dim=-1)

        first_point_mu, first_point_std = model.encoder_z0(
            truth_w_mask, tp_v1, static=static_v1, run_backwards=True
        )
        
        # We use the mean (deterministic) for a clean trajectory visualization
        z0_old = first_point_mu 

        # 3. Define a continuous range of counterfactual doses
        # Sweep from 0 to 1.5x the maximum dose observed across both visits
        max_dose_v1 = data_dict["dose_v1"].max().item()
        max_dose_v2 = data_dict["dose_v2"].max().item()
        max_dose = max(max_dose_v1, max_dose_v2) * 1.5
        if max_dose == 0: max_dose = 10.0 # Fallback
        
        dose_v2_range = torch.linspace(0, max_dose, num_steps).to(device)

        z0_trajectory = []
        doses_used = []

        # 4. Transport the latent state across the dose manifold
        for dose_v2 in dose_v2_range:
            dose_v2_tensor = dose_v2.view(1) # Match batch shape
            
            # Form the context vector exactly as in get_reconstruction_extrapolation
            
            c_doses = torch.stack([dose_v1, dose_v2_tensor], dim=1).to(device)
            c_doses_exp = c_doses.unsqueeze(0) # [1, 1, 2] to match z0_old [1, 1, dim]
            
            c = torch.cat([c_doses_exp, z0_old], dim=-1)
            
            # Predict the Affine Transport Map parameters via FiLM
            gamma = model.film_gamma(c)
            beta = model.film_beta(c)

            # Apply the Bures-Wasserstein Optimal Transport map
            z0_new = (z0_old * gamma) + beta
            
            z0_trajectory.append(z0_new.squeeze().cpu().numpy())
            doses_used.append(dose_v2.item())

        z0_trajectory = np.array(z0_trajectory)

        # 5. Project the trajectory using PCA
        pca = PCA(n_components=2)
        z0_pca = pca.fit_transform(z0_trajectory)

        # 6. Plotting
        plt.figure(figsize=(10, 7))
        
        # Plot the trajectory line
        plt.plot(z0_pca[:, 0], z0_pca[:, 1], 'k--', alpha=0.5, zorder=1) 
        
        # Scatter the points colored by dose amount
        scatter = plt.scatter(
            z0_pca[:, 0], z0_pca[:, 1], 
            c=doses_used, cmap='viridis', 
            s=150, edgecolor='k', zorder=2
        )
        
        # Highlight the original administered dose
        orig_idx = np.argmin(np.abs(np.array(doses_used) - dose_v1.item()))
        plt.scatter(
            z0_pca[orig_idx, 0], z0_pca[orig_idx, 1], 
            c='red', marker='*', s=400, 
            label=f'Original Medicated State V1 (Dose: {dose_v1.item():.2f})', 
            edgecolor='k', zorder=3
        )

        plt.title(f'Continuous Optimal Transport Trajectory in Latent Space\n(Patient {patient_idx})', fontsize=14)
        plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)', fontsize=12)
        plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)', fontsize=12)
        cbar = plt.colorbar(scatter)
        cbar.set_label('Counterfactual Dose Amount', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Save or Show
        plt.tight_layout()
        plt.savefig(f'OT_Trajectory_Patient_{patient_idx}.png', dpi=300)
        plt.show()

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

def plot_dose_fan_trajectory(model, data_dict, patient_indices=[0, 1, 2], num_steps=20):
    """
    Visualizes the Optimal Transport 'Dose Fan' in the latent space.
    Plots counterfactual dose sweeps for multiple patients on a single global PCA projection 
    to visually demonstrate manifold curvature (diverging trajectories and varying step sizes).
    """
    print(f"\n--- Generating Dose Fan Plot for Patients: {patient_indices} ---")
    model.eval()
    
    device = data_dict["observed_data_v1"].device
    
    # Define a single global continuous range of counterfactual doses for fair comparison
    max_dose_v1 = data_dict["dose_v1"].max().item()
    max_dose_v2 = data_dict["dose_v2"].max().item()
    max_dose = max(max_dose_v1, max_dose_v2) * 1.5
    if max_dose == 0: max_dose = 10.0 # Fallback
    
    dose_v2_range = torch.linspace(0, max_dose, num_steps).to(device)
    
    all_trajectories = []
    original_doses = []
    
    with torch.no_grad():
        # 1. Generate trajectories for all selected patients
        for patient_idx in patient_indices:
            # Extract the specific patient's data for Visit 1 (Source state)
            data_v1 = data_dict["observed_data_v1"][patient_idx:patient_idx+1]
            tp_v1 = data_dict["observed_tp_v1"]
            dose_v1 = data_dict["dose_v1"][patient_idx:patient_idx+1]
            
            static_v1 = None
            if "static_v1" in data_dict and data_dict["static_v1"] is not None:
                static_v1 = data_dict["static_v1"][patient_idx:patient_idx+1]

            # Encode to get the initial latent state (z0_old)
            seq_len = data_v1.size(1)
            dose_expanded = dose_v1.view(-1, 1, 1).expand(-1, seq_len, 1)
            truth_w_mask = torch.cat((data_v1, dose_expanded), dim=-1)

            first_point_mu, _ = model.encoder_z0(
                truth_w_mask, tp_v1, static=static_v1, run_backwards=True
            )
            
            z0_old = first_point_mu # Use deterministic mean for trajectory
            
            patient_trajectory = []
            original_doses.append(dose_v1.item())

            # Transport the latent state across the dose manifold
            for dose_v2 in dose_v2_range:
                dose_v2_tensor = dose_v2.view(1) # Match batch shape
                
                # Form context vector
                c_doses = torch.stack([dose_v1, dose_v2_tensor], dim=1).to(device)
                c_doses_exp = c_doses.unsqueeze(0) # [1, 1, 2] to match z0_old [1, 1, dim]
                c = torch.cat([c_doses_exp, z0_old], dim=-1)
                
                # Predict the Affine Transport Map parameters via FiLM
                gamma = model.film_gamma(c)
                beta = model.film_beta(c)

                # Apply the Bures-Wasserstein Optimal Transport map
                z0_new = (z0_old * gamma) + beta
                patient_trajectory.append(z0_new.squeeze().cpu().numpy())
                
            all_trajectories.append(np.array(patient_trajectory))

    # 2. Fit a Global PCA on ALL patients' trajectories combined
    # This is critical to prove curvature; we must share the coordinate space
    combined_trajectories = np.vstack(all_trajectories)
    pca = PCA(n_components=2)
    pca.fit(combined_trajectories)

    # 3. Plotting
    plt.figure(figsize=(11, 8))
    
    # Define a list of markers to differentiate patients if needed
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X']
    
    for i, patient_idx in enumerate(patient_indices):
        # Project this specific patient's trajectory into the global PCA space
        z0_pca = pca.transform(all_trajectories[i])
        
        # Plot the trajectory line (the "Fan" blade)
        plt.plot(z0_pca[:, 0], z0_pca[:, 1], 'k-', alpha=0.3, zorder=1)
        
        # Scatter the points colored by dose amount
        doses_np = dose_v2_range.cpu().numpy()
        scatter = plt.scatter(
            z0_pca[:, 0], z0_pca[:, 1], 
            c=doses_np, cmap='viridis', 
            marker=markers[i % len(markers)],
            s=100, edgecolor='k', alpha=0.9, zorder=2,
            label=f'Patient {patient_idx} trajectory'
        )
        
        # Highlight the original administered dose (Baseline state)
        orig_idx = np.argmin(np.abs(doses_np - original_doses[i]))
        plt.scatter(
            z0_pca[orig_idx, 0], z0_pca[orig_idx, 1], 
            c='red', marker='*', s=350, 
            edgecolor='white', linewidth=1.5, zorder=3
        )

    # Styling and Annotations
    var_explained = pca.explained_variance_ratio_ * 100
    plt.title('Emergent Curvature of the Pharmacological Manifold\nContinuous Dose Transport Across Diverse Patient Phenotypes', fontsize=16, pad=15)
    plt.xlabel(f'Global PCA Component 1 ({var_explained[0]:.1f}% Variance)', fontsize=13)
    plt.ylabel(f'Global PCA Component 2 ({var_explained[1]:.1f}% Variance)', fontsize=13)
    
    # Add Colorbar for Dose
    cbar = plt.colorbar(scatter)
    cbar.set_label('Counterfactual Dose Amount (mg)', fontsize=13)
    
    # Custom legend handling (combine patient markers and the red star)
    handles, labels = plt.gca().get_legend_handles_labels()
    import matplotlib.lines as mlines
    star_marker = mlines.Line2D([], [], color='red', marker='*', linestyle='None',
                                markersize=15, markeredgecolor='white', label='Original Medicated State (V1)')
    handles.append(star_marker)
    plt.legend(handles=handles, fontsize=11, loc='best', framealpha=0.9)
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('Dose_Fan_Manifold_Curvature.png', dpi=300)
    plt.show()
    
    print("\nObservation Guide for the Paper:")
    print("- If the lines are perfectly parallel and dot-spacing is identical, the manifold is FLAT.")
    print("- If the lines fan out (different angles) OR dot-spacing varies per patient, the manifold is CURVED.")