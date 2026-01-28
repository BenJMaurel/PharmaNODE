###########################
# Latent ODEs with Normalizing Flows - Interactive Visualization
# DEBUG VERSION: Includes safety checks for Flow Rendering
###########################

import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import itertools
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
from scipy.special import inv_boxcox
import time
import argparse
import numpy as np
import pandas as pd
from random import SystemRandom
import plotly.graph_objects as go
import plotly.express as px
import io
import base64

# Imports for Dash
from dash import Dash, dcc, html, Input, Output, no_update

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import lib.utils as utils
from lib.create_latent_ode_model import create_LatentODE_model
from lib.parse_datasets import parse_datasets

# --- Argument Parser ---
parser = argparse.ArgumentParser('Latent ODE Flow Viz')
parser.add_argument('-n',  type=int, default=100)
parser.add_argument('--lr',  type=float, default=1e-2)
parser.add_argument('-b', '--batch-size', type=int, default=2000)
parser.add_argument('--save', type=str, default='experiments/')
parser.add_argument('--load', type=str, default=None, required=True)
parser.add_argument('-r', '--random-seed', type=int, default=1991)
parser.add_argument('--dataset', type=str, default='periodic')
parser.add_argument('-s', '--sample-tp', type=float, default=None)
parser.add_argument('-c', '--cut-tp', type=int, default=None)
parser.add_argument('--quantization', type=float, default=0.1)

# Model Specifics
parser.add_argument('--latent-ode', action='store_true')
parser.add_argument('--z0-encoder', type=str, default='odernn')
parser.add_argument('-l', '--latents', type=int, default=6)
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
parser.add_argument('--max-t',  type=float, default=5.)
parser.add_argument('--noise-weight', type=float, default=0.01)
parser.add_argument('--seed', type = int, default = 15)

# Keep old args to prevent crash
parser.add_argument('--use-gmm', action='store_true')
parser.add_argument('--use-gmm-v', action='store_true')
parser.add_argument('-nc', '--n_components', type=int, default=4)
parser.add_argument('--classic-rnn', action='store_true')
parser.add_argument('--rnn-cell', default="gru")
parser.add_argument('--input-decay', action='store_true')
parser.add_argument('--ode-rnn', action='store_true')
parser.add_argument('--rnn-vae', action='store_true')

# Flow args
parser.add_argument('--use-flow', action='store_true') 
parser.add_argument('--flow-layers', type=int, default=4)

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
file_name = os.path.basename(__file__)[:-3]
utils.makedirs(args.save)

# --- Robust Contour Function ---
def get_density_contours(x, y):
    """
    Computes density contour lines for 2D data using Gaussian KDE.
    Now dynamically calculates levels to ensure visibility.
    """
    # 1. Fit KDE
    positions = np.vstack([x, y])
    try:
        kernel = gaussian_kde(positions)
    except Exception as e:
        print(f"KDE Fitting Failed: {e}")
        return []

    # 2. Create Grid
    xmin, xmax = x.min(), x.max() 
    ymin, ymax = y.min(), y.max()
    # Add 10% padding
    x_pad = (xmax - xmin) * 0.1
    y_pad = (ymax - ymin) * 0.1
    X, Y = np.mgrid[xmin-x_pad:xmax+x_pad:100j, ymin-y_pad:ymax+y_pad:100j]
    positions_grid = np.vstack([X.ravel(), Y.ravel()])
    
    # 3. Evaluate Z
    Z = np.reshape(kernel(positions_grid).T, X.shape)
    
    # 4. Dynamic Levels (Key Change)
    z_max = Z.max()
    # Create levels at 10%, 30%, 50%, 70%, 90% of max density
    levels = [0.1 * z_max, 0.3 * z_max, 0.5 * z_max, 0.7 * z_max, 0.9 * z_max]
    
    # 5. Extract Contours
    fig_temp, ax_temp = plt.subplots()
    CS = ax_temp.contour(X, Y, Z, levels=levels)
    
    contour_lines = []
    # Matplotlib 3.8+ compatibility
    if hasattr(CS, "get_paths"):
        paths = CS.get_paths()
    else:
        # Older Matplotlib versions
        paths = []
        for collection in CS.collections:
            paths.extend(collection.get_paths())

    for path in paths:
        for poly in path.to_polygons(closed_only=False):
            contour_lines.append(poly)
            
    plt.close(fig_temp)
    return contour_lines

#####################################################################################################

if __name__ == '__main__':
    args.experiment = args.load
    ckpt_path = os.path.join(args.save, "experiment_" + str(args.load) + '.ckpt')    
    
    data_obj = parse_datasets(args, device)
    input_dim = data_obj["input_dim"]   
    classif_per_tp = data_obj.get("classif_per_tp", False)
    n_labels = data_obj.get("n_labels", 1) if args.classif else 1

    obsrv_std = torch.Tensor([0.01]).to(device)
    z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))
    
    model = create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, device, 
        classif_per_tp = classif_per_tp, n_labels = n_labels)
    
    utils.get_ckpt_model(ckpt_path, model, device)
    print(f"Model {args.load} loaded.")

    # --- Inference ---
    all_reconstructions = []
    all_data = []
    all_latent_z0_means = []
    all_labels = []
    
    with torch.no_grad():
        data_dict = utils.get_next_batch(data_obj["test_dataloader"])
        
        data =  data_dict["data_to_predict"]
        time_steps = data_dict["tp_to_predict"]
        observed_data =  data_dict["observed_data"]
        observed_time_steps = data_dict["observed_tp"]
        observed_mask = data_dict["observed_mask"]
        dose = data_dict["dose"]
        static = data_dict["static"]
        patient_id = data_dict['patient_id']
        dataset = data_dict['dataset_number']
        
        device = utils.get_device(time_steps)
        time_steps_to_predict = utils.linspace_vector(time_steps[0], torch.tensor(24.), 100).to(device)
        
        reconstructions, info = model.get_reconstruction(time_steps_to_predict, 
            observed_data, observed_time_steps, dose=dose, static=static, 
            mask=observed_mask, n_traj_samples=50)
        
        # Customize labels as needed
        labels_raw = 2 * static[:,1] + static[:,2]
        all_labels.append(labels_raw)
        
        latent_z0_mean = info["first_point"][0]
        all_latent_z0_means.append(latent_z0_mean)
        all_reconstructions.append(reconstructions)
        all_data.append(data)

    # --- Scaling ---
    original_indices = dataset.cpu().numpy().astype(int)
    scaling_factor = data_obj['max_out']['max_out'][original_indices]
    
    if 'best_lambda' in data_obj['max_out'].keys():
        lam = data_obj['max_out']['best_lambda'][0]
        data = inv_boxcox(data, lam)
        reconstructions = inv_boxcox(reconstructions, lam)
        reconstructions = torch.nan_to_num(reconstructions, nan=0.0)

    data = data.detach().numpy() * scaling_factor[:, np.newaxis, np.newaxis]
    reconstructions = reconstructions.detach().numpy() * scaling_factor[:, np.newaxis, np.newaxis]
    
    all_labels = list(itertools.chain(*all_labels))
    all_latent_z0_means = torch.cat(all_latent_z0_means, dim=0).cpu().numpy().squeeze(0)
    
    # --- PCA ---
    print("Running PCA on Latent Means...")
    reducer = PCA(n_components=2)
    embedding_real = reducer.fit_transform(all_latent_z0_means)

    # --- Flow Sampling ---
    embedding_flow = None
    
    # DEBUG: Check if flow exists
    if hasattr(model, 'flow'):
        print("Sampling from Flow Prior...")
        with torch.no_grad():
            u_samples = torch.randn(5000, model.latent_dim).to(device)
            z_flow_samples, _ = model.flow(u_samples)
            z_flow_samples = z_flow_samples.cpu().numpy()
        
        print(f"Sampled {len(z_flow_samples)} flow points.")
        embedding_flow = reducer.transform(z_flow_samples)
    else:
        print("!!! WARNING: Model does not have 'flow' attribute. Is --use-flow set?")

    # --- Tooltip Helper ---
    ts_to_predict_np = time_steps_to_predict.cpu().numpy()
    ts_gt_np = time_steps.cpu().numpy()

    def plot_to_base64_url(index):
        fig_hover, ax_hover = plt.subplots(figsize=(4, 2.5), dpi=100)
        rec_mean = np.mean(reconstructions[:, index, :, 0], axis=0)
        ax_hover.plot(ts_to_predict_np, rec_mean, label='Pred', color='red')
        try:
            gt = data[index, :, 0]
            mask_gt = gt != 0 
            if np.any(mask_gt):
                ax_hover.scatter(ts_gt_np[mask_gt], gt[mask_gt], label='Obs', color='black', s=10)
        except Exception:
            pass
        p_id_str = patient_id[index] if index < len(patient_id) else str(index)
        ax_hover.set_title(f'Pat ID: {p_id_str}', fontsize=10)
        ax_hover.legend(fontsize=8)
        plt.tight_layout()
        buf = io.BytesIO()
        fig_hover.savefig(buf, format="png")
        plt.close(fig_hover)
        encoded_image = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{encoded_image}"

    # --- Plotly ---
    df = pd.DataFrame({
        'x': embedding_real[:, 0],
        'y': embedding_real[:, 1],
        'patient_id': [str(p) for p in patient_id],
        'label': [int(l) for l in all_labels],
        'global_index': list(range(len(patient_id)))
    })

    print("Building Interactive Plotly Figure...")
    fig = go.Figure()

    # Layer A: Flow Contours
    if embedding_flow is not None:
        print("Computing Density Contours...")
        contours = get_density_contours(embedding_flow[:,0], embedding_flow[:,1])
        
        if len(contours) > 0:
            print(f"Generated {len(contours)} contour segments.")
            for i, vertices in enumerate(contours):
                fig.add_trace(go.Scatter(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    mode='lines',
                    line=dict(color='black', width=1.5, dash='dash'),
                    hoverinfo='skip',
                    name='Flow Density' if i == 0 else None,
                    showlegend=(i==0)
                ))
        else:
            print("!!! WARNING: No contours generated. Plotting raw points instead.")
            # Fallback: Plot raw flow points if contours fail
            fig.add_trace(go.Scatter(
                x=embedding_flow[:, 0],
                y=embedding_flow[:, 1],
                mode='markers',
                marker=dict(color='gray', size=2, opacity=0.3),
                name='Raw Flow Samples'
            ))

    # Layer B: Real Patients
    unique_lbls = sorted(df['label'].unique())
    for lbl in unique_lbls:
        mask = df['label'] == lbl
        subset = df[mask]
        fig.add_trace(go.Scatter(
            x=subset['x'],
            y=subset['y'],
            mode='markers',
            marker=dict(size=8, line=dict(width=1, color='white')),
            name=f"Group {lbl}",
            customdata=np.stack((subset['patient_id'], subset['global_index']), axis=-1)
        ))

    fig.update_layout(
        title='Latent Space: Flow Prior vs Patient Embeddings',
        template='plotly_white',
        clickmode='event+select'
    )

    app = Dash(__name__)
    app.layout = html.Div([
        dcc.Graph(id="graph", figure=fig, clear_on_unhover=True),
        dcc.Tooltip(id="tooltip"),
    ])

    @app.callback(
        Output("tooltip", "show"),
        Output("tooltip", "bbox"),
        Output("tooltip", "children"),
        Input("graph", "hoverData"),
    )
    def display_hover(hoverData):
        if hoverData is None: return False, no_update, no_update
        pt = hoverData["points"][0]
        if "customdata" not in pt: return False, no_update, no_update

        bbox = pt["bbox"]
        p_id = pt["customdata"][0]
        g_idx = int(pt["customdata"][1])
        img_src = plot_to_base64_url(g_idx)

        children = html.Div([
            html.Img(src=img_src, style={"width": "100%"}),
            html.P(f"ID: {p_id}", style={"font-weight": "bold"})
        ], style={'width': '280px', 'padding': '10px'})

        return True, bbox, children

    print("App running...")
    app.run()