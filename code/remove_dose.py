import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

def test_orthogonal_disentanglement(model, data_dict, num_patients=50, num_steps=50):
    """
    Tests disentanglement by projecting counterfactual dose sweeps onto a patient-specific 
    orthogonal basis (via PCA) and evaluating the predictive power of the components.
    """
    print("\n" + "="*60)
    print(" EXPERIMENT A (REVISED): ORTHOGONAL MANIFOLD DISENTANGLEMENT")
    print("="*60)
    model.eval()
    device = data_dict["observed_data_v1"].device
    
    r2_dose_dir_list = []
    r2_ortho_dir_list = []

    # Cap num_patients to available batch size
    batch_size = data_dict["observed_data_v1"].size(0)
    num_patients = min(num_patients, batch_size)
    
    with torch.no_grad():
        for patient_idx in range(num_patients):
            # 1. Extract patient V1 data
            data_v1 = data_dict["observed_data_v1"][patient_idx:patient_idx+1]
            tp_v1 = data_dict["observed_tp_v1"]
            dose_v1 = data_dict["dose_v1"][patient_idx:patient_idx+1]
            static_v1 = data_dict.get("static_v1", None)
            if static_v1 is not None:
                static_v1 = static_v1[patient_idx:patient_idx+1]

            # 2. Encode to get z0_old
            seq_len = data_v1.size(1)
            dose_expanded = dose_v1.view(-1, 1, 1).expand(-1, seq_len, 1)
            truth_w_mask = torch.cat((data_v1, dose_expanded), dim=-1)
            
            z0_old, _ = model.encoder_z0(truth_w_mask, tp_v1, static=static_v1, run_backwards=True)

            # 3. Generate a continuous sweep of new doses
            max_dose = max(data_dict["dose_v1"].max().item(), data_dict["dose_v2"].max().item()) * 1.5
            if max_dose == 0: max_dose = 10.0
            dose_v2_range = torch.linspace(0, max_dose, num_steps).to(device)
            
            z0_trajectory = []
            doses_used = []
            
            # 4. Generate the counterfactual states
            for dose_v2 in dose_v2_range:
                dose_v2_tensor = dose_v2.view(1)
                c_doses = torch.stack([dose_v1, dose_v2_tensor], dim=1).to(device)
                c_doses_exp = c_doses.unsqueeze(0) 
                
                c = torch.cat([c_doses_exp, z0_old], dim=-1)
                gamma = model.film_gamma(c)
                beta = model.film_beta(c)
                
                z0_new = (z0_old * gamma) + beta
                z0_trajectory.append(z0_new.squeeze().cpu().numpy())
                doses_used.append(dose_v2.item())
                
            Z_sweep = np.array(z0_trajectory)
            y_doses = np.array(doses_used)
            
            # 5. Create patient-specific orthogonal basis using PCA
            pca = PCA()
            Z_pca = pca.fit_transform(Z_sweep)
            # Z_pca = pca.transform(z0_full_np.reshape(1,-1))
            dose_vector = pca.components_[0]
            z_dose = Z_pca[:, 0:1]  # Only the first dimension
            z0_bio_np = Z_pca[:, 1:]
            # Formula: z_bio = z_full - projection of z_full onto dose_vector
            projection_magnitude = np.dot(Z_sweep, dose_vector)
            z0_bio_np = Z_sweep - (projection_magnitude.reshape(-1,1) * dose_vector.reshape(1,-1))
            
            # 6. Split into "Dose Dimension" (PC1) and "Orthogonal Subspace" (PC2 to N)
            X_dose_dir = Z_pca[:, 0:1]  # Only the first dimension
            X_ortho_dir = Z_pca[:, 1:]  # All other orthogonal dimensions
            
            # 7. Predict the sweeping dose using Ridge Regression (Linear)
            # Test A: Can PC1 predict the dose?
            reg_dose = Ridge(alpha=0.1).fit(Z_pca, y_doses)
            r2_dose = r2_score(y_doses, reg_dose.predict(Z_pca))
            
            # Test B: Can the orthogonal subspace predict the dose?
            reg_ortho = Ridge(alpha=0.1).fit(z0_bio_np, y_doses)
            r2_ortho = r2_score(y_doses, reg_ortho.predict(z0_bio_np))
            
            r2_dose_dir_list.append(r2_dose)
            r2_ortho_dir_list.append(r2_ortho)

    # 8. Aggregate and Print Results
    mean_r2_dose = np.mean(r2_dose_dir_list)
    mean_r2_ortho = np.mean(r2_ortho_dir_list)
    
    print(f"Evaluated across {num_patients} patients (Patient-specific basis).")
    print(f"--> Mean R^2 using ONLY the 1D Dose Direction (PC1):     {mean_r2_dose:.4f}")
    print(f"--> Mean R^2 using the Orthogonal Subspace (PC2-PC_n):   {mean_r2_ortho:.4f}")
    print("="*60 + "\n")