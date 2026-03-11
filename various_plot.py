# Save as plot_quiver.py
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

def plot_dose_vector_field(model, batch_dict, dose_step=5.0):
    model.eval()
    device = batch_dict["observed_data_v1"].device
    
    with torch.no_grad():
        # 1. Encode full batch
        data_v1 = batch_dict["observed_data_v1"]
        tp_v1 = batch_dict["observed_tp_v1"]
        dose_v1 = batch_dict["dose_v1"]
        static = batch_dict["static_v1"]
        
        # Logic to match model's expected input [Batch, Time, Dim] [cite: 1213, 1217]
        seq_len = data_v1.size(1)
        dose_expanded = dose_v1.view(-1, 1, 1).expand(-1, seq_len, 1)
        truth_w_mask = torch.cat((data_v1, dose_expanded), dim=-1)
        
        # Get z0_old [Batch, 1, Latent_Dim] [cite: 1218, 1219]
        z0_old, _ = model.encoder_z0(truth_w_mask, tp_v1, static=static, run_backwards=True)
        
        # 2. Setup transport context for the WHOLE batch
        # Shift dose by dose_step
        dose_v2 = dose_v1 + dose_step 
        
        # c_doses must be [Batch, 1, 2] to concatenate with z0_old [Batch, 1, Latent_Dim] 
        c_doses = torch.stack([dose_v1, dose_v2], dim=-1).unsqueeze(0) 
        
        # Correct Concatenation: [Batch, 1, 2 + Latent_Dim]
        c = torch.cat([c_doses, z0_old], dim=-1)
        
        # Get FiLM parameters [cite: 1161, 1187, 1188]
        gamma = model.film_gamma(c)
        beta = model.film_beta(c)
        
        # Apply Bures-Wasserstein transport [cite: 1149, 1170]
        z0_new = (z0_old * gamma) + beta
        
        # Convert to numpy for PCA [cite: 675, 732]
        diff = (z0_new - z0_old).squeeze(0).cpu().numpy()
        base = z0_old.squeeze(0).cpu().numpy()

    # 3. Visualization
    pca = PCA(n_components=2)
    coords = pca.fit_transform(base)
    # Project the vectors into PCA space [cite: 676, 732]
    vectors = pca.transform(base + diff) - coords 

    plt.figure(figsize=(10, 8))
    q = plt.quiver(coords[:, 0], coords[:, 1], vectors[:, 0], vectors[:, 1], 
                   np.linalg.norm(diff, axis=1), cmap='viridis', alpha=0.8)
    plt.colorbar(q, label='Transport Magnitude in Latent Space')
    plt.title(f"Dose Vector Field (Dose Increase: +{dose_step}mg)\nEvidence of Phenotype-Dependent Transport")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig("manifold_quiver.png")
    plt.close() # Avoid memory issues in headless environments
# Save as analyze_curvature.py
import torch
import numpy as np
import matplotlib.pyplot as plt

def quantify_manifold_curvature(model, batch_dict, dose_step = 5.0):
    """
    Quantifies curvature by calculating the angle between transport vectors 
    of different patients. Flat manifold = 0 degree variance.
    """
    model.eval()
    with torch.no_grad():
        data_v1 = batch_dict["observed_data_v1"]
        tp_v1 = batch_dict["observed_tp_v1"]
        dose_v1 = batch_dict["dose_v1"]
        static = batch_dict["static_v1"]
        
        seq_len = data_v1.size(1)
        truth_w_mask = torch.cat((data_v1, dose_v1.view(-1, 1, 1).expand(-1, seq_len, 1)), dim=-1)
        z0_old, _ = model.encoder_z0(truth_w_mask, tp_v1, static=static, run_backwards=True)
        
        # 2. Transport everyone by a fixed +dose_step
        dose_v2 = dose_v1 + dose_step
        c_doses = torch.stack([dose_v1, dose_v2], dim=1)
        c = torch.cat([c_doses.unsqueeze(0), z0_old], dim=-1)
        
        gamma = model.film_gamma(c)
        beta = model.film_beta(c)
        z0_new = (z0_old * gamma) + beta
        
        # Calculate displacement vector
        diff = (z0_new - z0_old).squeeze().cpu().numpy()
        base = z0_old.squeeze().cpu().numpy()
        
        # Normalize vectors for Cosine Similarity [cite: 692, 693]
        norms = np.linalg.norm(diff, axis=1, keepdims=True)
        unit_vectors = diff / (norms + 1e-9)
        
        # Compute all-pairs cosine similarity
        cos_sim = np.dot(unit_vectors, unit_vectors.T)
        # Convert to angles
        angles = np.arccos(np.clip(cos_sim, -1.0, 1.0)) * (180 / np.pi)
        
        # Get only the unique pairs (upper triangle)
        unique_angles = angles[np.triu_indices(angles.shape[0], k=1)]
        
    plt.figure(figsize=(8, 5))
    plt.hist(unique_angles, bins=50, color='crimson', alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(unique_angles), color='black', linestyle='--', 
                label=f'Mean Divergence: {np.mean(unique_angles):.2f}°')
    plt.title("Angular Divergence of Dose Transport Vectors")
    plt.xlabel("Angle between Patient Dose-Vectors (Degrees)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig("curvature_histogram.png")
    plt.close()
    return np.mean(unique_angles)

# Save as plot_grid_deformation.py
def plot_grid_deformation(model, dose_start=2.0, dose_end=5.0):
    """
    Applies the transport map to a perfectly uniform grid of latent points.
    Visualizing how a square grid becomes 'warped' is the classic way to show curvature.
    """
    # 1. Create a 2D grid in latent space (assuming D=10, we vary 2 dimensions)
    x = np.linspace(-2, 2, 10)
    y = np.linspace(-2, 2, 10)
    xv, yv = np.meshgrid(x, y)
    
    grid_points = np.zeros((100, 10)) # Latent Dim D=10
    grid_points[:, 0] = xv.flatten()
    grid_points[:, 1] = yv.flatten()
    z0_grid = torch.tensor(grid_points, dtype=torch.float32).unsqueeze(1).to(device)

    # 2. Apply FiLM
    d1 = torch.ones(100, 1) * dose_start
    d2 = torch.ones(100, 1) * dose_end
    c = torch.cat([torch.stack([d1, d2], dim=1), z0_grid], dim=-1)
    
    z0_deformed = (z0_grid * model.film_gamma(c)) + model.film_beta(c)
    
    # 3. Plot Original vs Deformed
    z0_def_np = z0_deformed.squeeze().detach().cpu().numpy()
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(grid_points[:, 0], grid_points[:, 1], c='blue', s=10)
    plt.title(f"Original Bio-Grid (Dose {dose_start})")
    
    plt.subplot(1, 2, 2)
    plt.scatter(z0_def_np[:, 0], z0_def_np[:, 1], c='red', s=10)
    plt.title(f"Transported Grid (Dose {dose_end})")
    plt.savefig("grid_deformation.png")

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
try:
    from ripser import ripser
    from persim import plot_diagrams
except ImportError:
    print("Warning: Please install ripser and persim via 'pip install scikit-tda' to run Persistent Homology.")
def plot_geodesic_vs_euclidean(model, data_dict, num_steps=20, n_neighbors=5):
    """
    Computes Euclidean vs Geodesic distances on the purified z_bio manifold to detect
    non-linear curvature (e.g., ribbon or horseshoe shapes).
    """
    model.eval()
    device = data_dict["observed_data_v1"].device
    batch_size = data_dict["observed_data_v1"].size(0)
    
    Z_bio_list = []
    
    print(f"Extracting z_bio for {batch_size} patients for Geodesic analysis...")
    with torch.no_grad():
        for patient_idx in range(batch_size):
            # 1. Extract patient V1 data
            data_v1 = data_dict["observed_data_v1"][patient_idx:patient_idx+1]
            tp_v1 = data_dict["observed_tp_v1"]
            dose_v1 = data_dict["dose_v1"][patient_idx:patient_idx+1]
            static_v1 = data_dict.get("static_v1", None)
            if static_v1 is not None:
                static_v1 = static_v1[patient_idx:patient_idx+1]

            # 2. Encode to get full z0
            seq_len = data_v1.size(1)
            dose_expanded = dose_v1.view(-1, 1, 1).expand(-1, seq_len, 1)
            truth_w_mask = torch.cat((data_v1, dose_expanded), dim=-1)
            
            z0_full, _ = model.encoder_z0(truth_w_mask, tp_v1, static=static_v1, run_backwards=True)
            z0_full_np = z0_full.squeeze().cpu().numpy()

            # 3. Sweep counterfactual doses to find the local Dose Direction
            max_dose = max(data_dict["dose_v1"].max().item(), data_dict["dose_v2"].max().item()) * 1.5
            if max_dose == 0: max_dose = 10.0
            dose_v2_range = torch.linspace(0, max_dose, num_steps).to(device)
            
            z0_trajectory = []
            for dose_v2 in dose_v2_range:
                dose_v2_tensor = dose_v2.view(1)
                c_doses = torch.stack([dose_v1, dose_v2_tensor], dim=1).to(device)
                c = torch.cat([c_doses.unsqueeze(0), z0_full], dim=-1)
                
                gamma = model.film_gamma(c)
                beta = model.film_beta(c)
                z0_new = (z0_full * gamma) + beta
                z0_trajectory.append(z0_new.squeeze().cpu().numpy())
                
            Z_sweep = np.array(z0_trajectory)
            
            # 4. Use PCA to project out the Dose Vector (PC1)
            # pca = PCA()
            # pca.fit(Z_sweep)
            # Z_pca = pca.transform(z0_full_np.reshape(1, -1))
            
            # # 5. Extract z_bio by dropping PC1
            # z0_bio_np = Z_pca[:, 1:] 
            # Z_bio_list.append(z0_bio_np)
            pca = PCA()
            pca.fit(Z_sweep)
            dose_vector = pca.components_[0]
            Z_pca = pca.transform(z0_full_np.reshape(1,-1))
            # z_dose = Z_pca[:, 0:1]  # Only the first dimension
            # z0_bio_np = Z_pca[:, 1:]
            # Formula: z_bio = z_full - projection of z_full onto dose_vector
            projection_magnitude = np.dot(z0_full_np, dose_vector)
            z0_bio_np = z0_full_np - (projection_magnitude * dose_vector)
            Z_bio_list.append(z0_bio_np)
            
    # Combine into a single matrix [Batch, Latent_Dim - 1]
    X_bio = np.array(Z_bio_list)

    # --- Distance Calculations ---
    # Calculate Euclidean distances
    euclidean_dists = squareform(pdist(X_bio, metric='euclidean'))
    
    # Calculate Geodesic distances via KNN graph
    # mode='distance' ensures edge weights equal the Euclidean distance between neighbors
    knn_graph = kneighbors_graph(X_bio, n_neighbors=n_neighbors, mode='distance', include_self=False)
    geodesic_dists = shortest_path(csgraph=knn_graph, directed=False)
    
    # Flatten the upper triangle of the matrices to plot pairwise distances
    upper_tri_indices = np.triu_indices(batch_size, k=1)
    euc_flat = euclidean_dists[upper_tri_indices]
    geo_flat = geodesic_dists[upper_tri_indices]
    
    # Filter out infinite distances (disconnected components in the KNN graph)
    valid_mask = ~np.isinf(geo_flat)
    euc_flat = euc_flat[valid_mask]
    geo_flat = geo_flat[valid_mask]

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    
    # Plot Geodesic vs Euclidean
    plt.scatter(euc_flat, geo_flat, alpha=0.5, s=10, c='purple')
    
    # Plot the y=x line (where Geodesic == Euclidean, meaning flat space)
    max_val = max(euc_flat.max(), geo_flat.max())
    plt.plot([0, max_val], [0, max_val], 'k--', label='Flat Space (y=x)')
    
    plt.title("Geodesic vs Euclidean Distance on $z_{bio}$ Manifold")
    plt.xlabel("Euclidean Distance (Straight Line)")
    plt.ylabel("Geodesic Distance (Along Manifold)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.savefig("results/geodesic_vs_euclidean_zbio.png")
    plt.close()
    print("Successfully saved Geodesic analysis to results/geodesic_vs_euclidean_zbio.png")
def plot_persistent_homology(model, batch_dict, num_steps=20):
    """
    Performs Topological Data Analysis (Persistent Homology) on the purified z_bio manifold.
    Calculates and plots the persistence diagram to detect clusters (H0) and loops/holes (H1).
    """
    model.eval()
    device = batch_dict["observed_data_v1"].device
    batch_size = batch_dict["observed_data_v1"].size(0)
    
    Z_bio_list = []
    
    print(f"Extracting z_bio for {batch_size} patients for TDA...")
    with torch.no_grad():
        for patient_idx in range(batch_size):
            # 1. Extract patient V1 data
            data_v1 = batch_dict["observed_data_v1"][patient_idx:patient_idx+1]
            tp_v1 = batch_dict["observed_tp_v1"]
            dose_v1 = batch_dict["dose_v1"][patient_idx:patient_idx+1]
            static_v1 = batch_dict.get("static_v1", None)
            if static_v1 is not None:
                static_v1 = static_v1[patient_idx:patient_idx+1]

            # 2. Encode to get full z0
            seq_len = data_v1.size(1)
            dose_expanded = dose_v1.view(-1, 1, 1).expand(-1, seq_len, 1)
            truth_w_mask = torch.cat((data_v1, dose_expanded), dim=-1)
            
            z0_full, _ = model.encoder_z0(truth_w_mask, tp_v1, static=static_v1, run_backwards=True)
            z0_full_np = z0_full.squeeze().cpu().numpy()

            # 3. Sweep counterfactual doses to find the local Dose Direction
            max_dose = max(batch_dict["dose_v1"].max().item(), batch_dict["dose_v2"].max().item()) * 1.5
            if max_dose == 0: max_dose = 10.0
            dose_v2_range = torch.linspace(0, max_dose, num_steps).to(device)
            
            z0_trajectory = []
            for dose_v2 in dose_v2_range:
                dose_v2_tensor = dose_v2.view(1)
                c_doses = torch.stack([dose_v1, dose_v2_tensor], dim=1).to(device)
                c = torch.cat([c_doses.unsqueeze(0), z0_full], dim=-1)
                
                gamma = model.film_gamma(c)
                beta = model.film_beta(c)
                z0_new = (z0_full * gamma) + beta
                z0_trajectory.append(z0_new.squeeze().cpu().numpy())
                
            Z_sweep = np.array(z0_trajectory)
            
            # 4. Use PCA to project out the Dose Vector (PC1)
            # pca = PCA()
            # pca.fit(Z_sweep)
            # Z_pca = pca.transform(z0_full_np.reshape(1, -1))
            
            # # 5. Extract z_bio by dropping PC1
            # z0_bio_np = Z_pca[:, 1:] 
            # Z_bio_list.append(z0_bio_np)
            pca = PCA()
            pca.fit(Z_sweep)
            dose_vector = pca.components_[0]
            Z_pca = pca.transform(z0_full_np.reshape(1,-1))
            # z_dose = Z_pca[:, 0:1]  # Only the first dimension
            # z0_bio_np = Z_pca[:, 1:]
            # Formula: z_bio = z_full - projection of z_full onto dose_vector
            projection_magnitude = np.dot(z0_full_np, dose_vector)
            z0_bio_np = z0_full_np - (projection_magnitude * dose_vector)
            # 5. Extract z_bio by dropping PC1
            Z_bio_list.append(z0_bio_np)
            
    # Combine into a single matrix [Batch, Latent_Dim - 1]
    z_bio_np = np.array(Z_bio_list)

    # 6. Run Persistent Homology using Ripser
    try:
        # maxdim=1 computes H0 (connected components) and H1 (1D loops)
        diagrams = ripser(z_bio_np, maxdim=1)['dgms']
        
        # 7. Plot the persistence diagram
        plt.figure(figsize=(8, 6))
        plot_diagrams(diagrams, show=False)
        plt.title("Persistence Diagram of Purified $z_{bio}$ Manifold")
        
        plt.savefig("results/persistence_diagram_zbio.png")
        plt.close()
        print("Successfully saved Topological Data Analysis plot to results/persistence_diagram_zbio.png")
        
    except NameError:
        print("Skipping TDA plot: 'ripser' or 'persim' is not available.")

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

try:
    import umap
except ImportError:
    print("Warning: Please install umap via 'pip install umap-learn' to run UMAP projections.")

def plot_umap_zbio(model, batch_dict, num_steps=20):
    """
    Projects the purified z_bio manifold into 2D using UMAP.
    Colors the resulting manifold by intrinsic clearance to visualize the biological gradient.
    """
    model.eval()
    device = batch_dict["observed_data_v1"].device
    batch_size = batch_dict["observed_data_v1"].size(0)
    
    Z_bio_list = []
    clearance_list = []
    
    print(f"Extracting z_bio for {batch_size} patients for UMAP projection...")
    with torch.no_grad():
        for patient_idx in range(batch_size):
            # 1. Extract patient V1 data
            data_v1 = batch_dict["observed_data_v1"][patient_idx:patient_idx+1]
            tp_v1 = batch_dict["observed_tp_v1"]
            dose_v1 = batch_dict["dose_v1"][patient_idx:patient_idx+1]
            static_v1 = batch_dict.get("static_v1", None)
            if static_v1 is not None:
                static_v1 = static_v1[patient_idx:patient_idx+1]
                
            # Extract clearance for coloring (Pure Biology)
            try:
                clearance = batch_dict['others_v1'][patient_idx][0].item()
            except (KeyError, IndexError):
                clearance = 0.0 # Fallback if unavailable
            clearance_list.append(clearance)

            # 2. Encode to get full z0
            seq_len = data_v1.size(1)
            dose_expanded = dose_v1.view(-1, 1, 1).expand(-1, seq_len, 1)
            truth_w_mask = torch.cat((data_v1, dose_expanded), dim=-1)
            
            z0_full, _ = model.encoder_z0(truth_w_mask, tp_v1, static=static_v1, run_backwards=True)
            z0_full_np = z0_full.squeeze().cpu().numpy()

            # 3. Sweep counterfactual doses to find the local Dose Direction
            max_dose = max(batch_dict["dose_v1"].max().item(), batch_dict["dose_v2"].max().item()) * 1.5
            if max_dose == 0: max_dose = 10.0
            dose_v2_range = torch.linspace(0, max_dose, num_steps).to(device)
            
            z0_trajectory = []
            for dose_v2 in dose_v2_range:
                dose_v2_tensor = dose_v2.view(1)
                c_doses = torch.stack([dose_v1, dose_v2_tensor], dim=1).to(device)
                c = torch.cat([c_doses.unsqueeze(0), z0_full], dim=-1)
                
                gamma = model.film_gamma(c)
                beta = model.film_beta(c)
                z0_new = (z0_full * gamma) + beta
                z0_trajectory.append(z0_new.squeeze().cpu().numpy())
                
            Z_sweep = np.array(z0_trajectory)
            
            # 4. Project out the Dose Vector (PC1)
            pca = PCA()
            pca.fit(Z_sweep)
            dose_vector = pca.components_[0]
            Z_pca = pca.transform(z0_full_np.reshape(1,-1))
            # z_dose = Z_pca[:, 0:1]  # Only the first dimension
            # z0_bio_np = Z_pca[:, 1:]
            # Formula: z_bio = z_full - projection of z_full onto dose_vector
            projection_magnitude = np.dot(z0_full_np, dose_vector)
            z0_bio_np = z0_full_np - (projection_magnitude.reshape(-1,1) * dose_vector.reshape(1,-1))
            # 5. Extract z_bio by dropping PC1
            Z_bio_list.append(z0_bio_np.squeeze(0))
            
            
    # Combine into a single matrix [Batch, Latent_Dim - 1]
    X_bio = np.array(Z_bio_list)
    clearances = np.array(clearance_list)

    # 6. Fit UMAP
    print("Fitting UMAP...")
    try:
        # n_neighbors controls how UMAP balances local vs global structure
        # min_dist controls how tightly UMAP packs points together
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        umap_embedding = reducer.fit_transform(X_bio)
        
        # 7. Plotting
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], 
                              c=clearances, cmap='coolwarm', s=25, alpha=0.8, edgecolors='none')
        
        plt.colorbar(scatter, label='Intrinsic Clearance (Metabolism Rate)')
        plt.title("UMAP Projection of Purified $z_{bio}$ Manifold")
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        plt.grid(True, linestyle=':', alpha=0.4)
        
        plt.savefig("results/umap_zbio_manifold.png")
        plt.close()
        print("Successfully saved UMAP projection to results/umap_zbio_manifold.png")
        
    except NameError:
        print("Skipping UMAP plot: 'umap' module not loaded.")