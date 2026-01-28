###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import os
import sys
import matplotlib

import matplotlib.pyplot
import matplotlib.pyplot as plt
import itertools
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from scipy.special import inv_boxcox
import matplotlib.colors as mcolors
import time
import datetime
import argparse
import numpy as np
import pandas as pd
from random import SystemRandom
from sklearn import model_selection
import umap
import plotly.graph_objects as go
import plotly.express as px
import io
import base64

# Imports for Dash
from dash import Dash
from dash import dcc, html, Input, Output, no_update
from plotly.offline import init_notebook_mode, iplot # For running in offline mode if needed

# Initialize plotly for notebook usage (if not already done)
# init_notebook_mode(connected=True)

import torch
import torch.nn as nn
from torch.nn.functional import relu
import torch.optim as optim

import lib.utils as utils
from lib.plotting import *

from lib.rnn_baselines import *
from lib.ode_rnn import *
from lib.create_latent_ode_model import create_LatentODE_model
from lib.parse_datasets import parse_datasets
from lib.ode_func import ODEFunc, ODEFunc_w_Poisson
from lib.diffeq_solver import DiffeqSolver
from mujoco_physics import HopperPhysics
from lib.latent_ode import LatentODEGMM_V
from lib.utils import compute_loss_all_batches
# Generative model for noisy data based on ODE
parser = argparse.ArgumentParser('Latent ODE')
parser.add_argument('-n',  type=int, default=100, help="Size of the dataset")
parser.add_argument('--lr',  type=float, default=1e-2, help="Starting learning rate.")
parser.add_argument('-b', '--batch-size', type=int, default=2000)
parser.add_argument('--viz', action='store_true', help="Show plots while training")

parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--experiment', type = str, default = None, help="Fix experiment number for reproducibility")
parser.add_argument('--load', type=str, default=None, help="ID of the experiment to load for evaluation. If None, run a new experiment.")
parser.add_argument('-r', '--random-seed', type=int, default=1991, help="Random_seed")

parser.add_argument('--dataset', type=str, default='periodic', help="Dataset to load. Available: physionet, activity, hopper, periodic")
parser.add_argument('-s', '--sample-tp', type=float, default=None, help="Number of time points to sub-sample."
	"If > 1, subsample exact number of points. If the number is in [0,1], take a percentage of available points per time series. If None, do not subsample")

parser.add_argument('-c', '--cut-tp', type=int, default=None, help="Cut out the section of the timeline of the specified length (in number of points)."
	"Used for periodic function demo.")

parser.add_argument('--quantization', type=float, default=0.1, help="Quantization on the physionet dataset."
	"Value 1 means quantization by 1 hour, value 0.1 means quantization by 0.1 hour = 6 min")

parser.add_argument('--use-gmm', action='store_true', help="Run Latent ODE with clusters inside latent space model")
parser.add_argument('--use-gmm-v', action='store_true', help="Run Latent ODE with clusters inside latent space model")
parser.add_argument('-nc', '--n_components', type=int, default=4, help="Number of Gaussian within the latent space")
parser.add_argument('--latent-ode', action='store_true', help="Run Latent ODE seq2seq model")
parser.add_argument('--z0-encoder', type=str, default='odernn', help="Type of encoder for Latent ODE model: odernn or rnn")

parser.add_argument('--classic-rnn', action='store_true', help="Run RNN baseline: classic RNN that sees true points at every point. Used for interpolation only.")
parser.add_argument('--rnn-cell', default="gru", help="RNN Cell type. Available: gru (default), expdecay")
parser.add_argument('--input-decay', action='store_true', help="For RNN: use the input that is the weighted average of impirical mean and previous value (like in GRU-D)")

parser.add_argument('--ode-rnn', action='store_true', help="Run ODE-RNN baseline: RNN-style that sees true points at every point. Used for interpolation only.")

parser.add_argument('--rnn-vae', action='store_true', help="Run RNN baseline: seq2seq model with sampling of the h0 and ELBO loss.")

parser.add_argument('-l', '--latents', type=int, default=6, help="Size of the latent state")
parser.add_argument('--rec-dims', type=int, default=20, help="Dimensionality of the recognition model (ODE or RNN).")

parser.add_argument('--rec-layers', type=int, default=1, help="Number of layers in ODE func in recognition ODE")
parser.add_argument('--gen-layers', type=int, default=1, help="Number of layers in ODE func in generative ODE")

parser.add_argument('-u', '--units', type=int, default=100, help="Number of units per layer in ODE func")
parser.add_argument('-g', '--gru-units', type=int, default=100, help="Number of units per layer in each of GRU update networks")

parser.add_argument('--poisson', action='store_true', help="Model poisson-process likelihood for the density of events in addition to reconstruction.")
parser.add_argument('--classif', action='store_true', help="Include binary classification loss -- used for Physionet dataset for hospiral mortality")

parser.add_argument('--linear-classif', action='store_true', help="If using a classifier, use a linear classifier instead of 1-layer NN")
parser.add_argument('--extrap', action='store_true', help="Set extrapolation mode. If this flag is not set, run interpolation mode.")

parser.add_argument('-t', '--timepoints', type=int, default=100, help="Total number of time-points")
parser.add_argument('--max-t',  type=float, default=5., help="We subsample points in the interval [0, args.max_tp]")
parser.add_argument('--noise-weight', type=float, default=0.01, help="Noise amplitude for generated traejctories")
parser.add_argument('--seed', type = int, default = 15, help="Fix seed for reproducibility")

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
file_name = os.path.basename(__file__)[:-3]
utils.makedirs(args.save)

def get_ellipse_curve(center, cov, scale=2.0, n_points=100):
    """
    Generates x,y points for an ellipse iso-contour.
    center: [x, y]
    cov: [[sxx, sxy], [syx, syy]] (2x2 covariance matrix)
    scale: Scale factor (2.0 approx 95% confidence interval)
    """
    # Eigendecomposition to get orientation and width/height
    vals, vecs = np.linalg.eigh(cov)
    
    # Calculate width and height (std devs)
    # vals can be slightly negative due to precision, clamp to 0
    vals = np.maximum(vals, 0)
    width, height = 2 * scale * np.sqrt(vals)
    
    # Rotation angle of the ellipse
    angle = np.arctan2(vecs[1, 0], vecs[0, 0])
    
    # Generate parametric ellipse
    t = np.linspace(0, 2*np.pi, n_points)
    xt = 0.5 * width * np.cos(t)
    yt = 0.5 * height * np.sin(t)
    
    # Rotate
    x_rot = xt * np.cos(angle) - yt * np.sin(angle)
    y_rot = xt * np.sin(angle) + yt * np.cos(angle)
    
    # Translate to center
    return x_rot + center[0], y_rot + center[1]

#####################################################################################################

if __name__ == '__main__':
    args.experiment = args.load
    experimentID = args.load
    if experimentID is None:
        # Make a new experiment ID
        experimentID = int(SystemRandom().random()*100000)
    ckpt_path = os.path.join(args.save, "experiment_" + str(experimentID) + '.ckpt')    
    start = time.time()
    print("Sampling dataset of {} training examples".format(args.n))

    input_command = sys.argv
    ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
    if len(ind) == 1:
        ind = ind[0]
        input_command = input_command[:ind] + input_command[(ind+2):]
    input_command = " ".join(input_command) 
    utils.makedirs("results/")  
    ##################################################################
    data_obj = parse_datasets(args, device)
    input_dim = data_obj["input_dim"]   
    classif_per_tp = False
    if ("classif_per_tp" in data_obj):
        # do classification per time point rather than on a time series as a whole
        classif_per_tp = data_obj["classif_per_tp"] 
    if args.classif and (args.dataset == "hopper" or args.dataset == "periodic"):
        raise Exception("Classification task is not available for MuJoCo and 1d datasets")  
    n_labels = 1
    if args.classif:
        if ("n_labels" in data_obj):
            n_labels = data_obj["n_labels"]
        else:
            raise Exception("Please provide number of labels for classification task")  
    ##################################################################
    # Create the model
    obsrv_std = 0.01
    if args.dataset == "hopper":
        obsrv_std = 1e-3    
    obsrv_std = torch.Tensor([obsrv_std]).to(device)    
    z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))    
    if args.rnn_vae:
        if args.poisson:
            print("Poisson process likelihood not implemented for RNN-VAE: ignoring --poisson") # Create RNN-VAE model
        model = RNN_VAE(input_dim, args.latents, 
    		device = device, 
    		rec_dims = args.rec_dims, 
    		concat_mask = True, 
    		obsrv_std = obsrv_std,
    		z0_prior = z0_prior,
    		use_binary_classif = args.classif,
    		classif_per_tp = classif_per_tp,
    		linear_classifier = args.linear_classif,
    		n_units = args.units,
    		input_space_decay = args.input_decay,
    		cell = args.rnn_cell,
    		n_labels = n_labels,
    		train_classif_w_reconstr = (args.dataset == "physionet")
    		).to(device)    
    elif args.classic_rnn:
        if args.poisson:
            print("Poisson process likelihood not implemented for RNN: ignoring --poisson") 
        if args.extrap:
            raise Exception("Extrapolation for standard RNN not implemented")
        # Create RNN model
        model = Classic_RNN(input_dim, args.latents, device, 
    		concat_mask = True, obsrv_std = obsrv_std,
    		n_units = args.units,
    		use_binary_classif = args.classif,
    		classif_per_tp = classif_per_tp,
    		linear_classifier = args.linear_classif,
    		input_space_decay = args.input_decay,
    		cell = args.rnn_cell,
    		n_labels = n_labels,
    		train_classif_w_reconstr = (args.dataset == "physionet")
    		).to(device)
    elif args.ode_rnn:
        # Create ODE-GRU model
        n_ode_gru_dims = args.latents

        if args.poisson:
            print("Poisson process likelihood not implemented for ODE-RNN: ignoring --poisson") 
        if args.extrap:
            raise Exception("Extrapolation for ODE-RNN not implemented")    
        ode_func_net = utils.create_net(n_ode_gru_dims, n_ode_gru_dims, 
    		n_layers = args.rec_layers, n_units = args.units, nonlinear = nn.Tanh)  
        rec_ode_func = ODEFunc(
    		input_dim = input_dim, 
    		latent_dim = n_ode_gru_dims,
    		ode_func_net = ode_func_net,
    		device = device).to(device) 
        z0_diffeq_solver = DiffeqSolver(input_dim, rec_ode_func, "euler", args.latents, 
    		odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device)

        model = ODE_RNN(input_dim, n_ode_gru_dims, device = device, 
    		z0_diffeq_solver = z0_diffeq_solver, n_gru_units = args.gru_units,
    		concat_mask = True, obsrv_std = obsrv_std,
    		use_binary_classif = args.classif,
    		classif_per_tp = classif_per_tp,
    		n_labels = n_labels,
    		train_classif_w_reconstr = (args.dataset == "physionet")
    		).to(device)
    elif args.latent_ode:
        model = create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, device, 
    		classif_per_tp = classif_per_tp,
    		n_labels = n_labels)
    else:
    
        raise Exception("Model not specified")  
    ##################################################################  
    # if args.viz:
    # 	viz = Visualizations(device)    
    ##################################################################

    #Load checkpoint and evaluate the model
    if args.load is not None:
        utils.get_ckpt_model(ckpt_path, model, device)  
    ##################################################################
    # Training  
    log_path = "logs/" + file_name + "_" + str(experimentID) + ".log"
    if not os.path.exists("logs/"):
        utils.makedirs("logs/")
    num_batches = data_obj["n_test_batches"]
    all_reconstructions = []
    all_data = []
    all_latent_z0_var = []
    all_latent_z0_means = []
    all_latent_trajectories = []
    all_labels = []
    # for itr in range(1, num_batches):
    for itr in range(0,1):
        with torch.no_grad():
            data_dict = utils.get_next_batch(data_obj["test_dataloader"])
            data =  data_dict["data_to_predict"]
            time_steps = data_dict["tp_to_predict"]
            mask = data_dict["mask_predicted_data"]
            auc_be = data_dict["auc_be"]
            auc_red = data_dict["auc_red"]
            observed_data =  data_dict["observed_data"]
            observed_time_steps = data_dict["observed_tp"]
            observed_mask = data_dict["observed_mask"]
            y_true_times = data_dict['y_true_times']
            dose = data_dict["dose"]
            static = data_dict["static"]
            others = data_dict["others"]
            dataset = data_dict['dataset_number']
            patient_id = data_dict['patient_id']
            device = get_device(time_steps)
            if isinstance(model, LatentODE) or isinstance(model, LatentODEGMM) or isinstance(model, LatentODEGMM_V):
    	        # sample at the original time points
                time_steps_to_predict = utils.linspace_vector(time_steps[0], torch.tensor(24.), 100).to(device)
            reconstructions, info = model.get_reconstruction(time_steps_to_predict, 
    			observed_data, observed_time_steps, dose = dose, static = static, mask = observed_mask, n_traj_samples = 100)
            # all_labels.append(static[:,1])
            all_labels.append(2*static[:,1] + static[:,2])
    		# all_labels.append(auc_red[:,0])
            latent_z0_var = info["first_point"][1]
            latent_z0_mean = info["first_point"][0]
            latent_traj = info["latent_traj"]
            all_latent_trajectories.append(latent_traj)
            all_latent_z0_means.append(latent_z0_mean.squeeze(0))
            all_latent_z0_var.append(latent_z0_var.squeeze(0))
            all_data.append(data)
            all_reconstructions.append(reconstructions)
        original_indices = dataset.cpu().numpy().astype(int)
    	# original_indices = static[:,1].cpu().numpy().astype(int)
        unique_values = np.unique(original_indices)

    # reconstructions[:, :, 50:, :] = 0
    train_full = 1
    if len(unique_values)==1 or train_full == 1:
        new_indices = original_indices
    else:
        new_indices = np.searchsorted(unique_values, original_indices)
    scaling_factor = data_obj['max_out']['max_out'][new_indices]
    # scaling_factor = data_obj['max_out'][dataset.cpu().numpy().astype(int)]
    # scaling_factor = data_obj['max_out'][(all_labels[0]).cpu().numpy().astype(int)]

    # scaling_factor = data_obj['max_out']
    # mean, std = data_obj['dataset_train'].mean, data_obj['dataset_train'].std
    # data = data.detach().numpy()*data_obj["max_out"]
    # reconstructions = reconstructions.detach().numpy()*data_obj["max_out"]

    # data = ((data.detach()*std) + mean).numpy()
    # reconstructions = ((reconstructions.detach()*std) + mean).numpy()
    # data = torch.cat(all_data)
    # reconstructions = torch.cat(all_reconstructions)
    if 'best_lambda' in data_obj['max_out'].keys():
        data = inv_boxcox(data, data_obj['max_out']['best_lambda'][0])
        reconstructions = inv_boxcox(reconstructions, data_obj['max_out']['best_lambda'][0])
        reconstructions = torch.nan_to_num(reconstructions, nan=0.0)
    reconstructions[:, static[:,1].bool(), 50:, :] = 0
    if torch.mean(auc_red)!=0.0:
        if len(auc_red.shape) == 1:
            auc_red = (auc_red*scaling_factor)
        elif len(auc_red.shape) == 2:
            auc_red = (auc_red*scaling_factor[:, np.newaxis]).squeeze(-1)
    # 	# pass
    # else:
    # 	auc_red = None
    data = data.detach().numpy()*scaling_factor[:, np.newaxis, np.newaxis]
    reconstructions = reconstructions.detach().numpy()*scaling_factor[:, np.newaxis, np.newaxis]
    # reconstructions = reconstructions.detach().numpy()
    # data = data.detach().numpy()
    # reconstructions = reconstructions.detach().numpy()
    try:
        auc_be = auc_be.detach().numpy()*scaling_factor[:, np.newaxis]
    except:
        auc_be = auc_be.detach().numpy()*scaling_factor[:, np.newaxis, np.newaxis]
    # Create a figure
    all_labels = list(itertools.chain(*all_labels))
    fig = plt.figure()
    # # Add a single subplot (1 row, 1 column, 1st plot)
    ax = fig.add_subplot(1, 1, 1)
    fig, ax = plt.subplots(figsize=(10, 7))
    # error, rmse, cat_metrics = plot_auc(ax, data, reconstructions, y_true_times, time_steps_to_predict, auc_red = auc_red, auc_be = None, labels = all_labels, patient_ids=patient_id)
    error, rmse, cat_metrics, error_be, rmse_be, cat_be_metrics, indices_a_thresh = plot_auc(ax, data, reconstructions, y_true_times, time_steps_to_predict, auc_be = auc_be, auc_red = auc_red,labels = all_labels, patient_ids=patient_id, args = args, plot_all = False, save_csv = False)

    all_latent_z0_means = np.concatenate(all_latent_z0_means, axis=0)
    all_latent_z0_var = np.concatenate(all_latent_z0_var, axis=0)
    all_latent_trajectories = np.concatenate(all_latent_trajectories, axis=1)

    # Before UMAP, reshape the trajectories data
    # Shape of all_latent_trajectories: [n_traj_samples, n_patients, n_timepoints, n_latents]
    # We want to run UMAP on all latent states, so we need to reshape it
    # to [n_traj_samples * n_patients * n_timepoints, n_latents]
    n_traj_samples, n_patients, n_timepoints, n_latents = all_latent_trajectories.shape
    all_latent_states = np.mean(all_latent_trajectories, axis = 0)
    all_latent_states = all_latent_states.reshape(-1, n_latents)
    print("Step 1: Running UMAP on all latent states...")
    # mapper = umap.UMAP(
    #     n_neighbors=min(30, len(all_latent_states)-1),          # How many neighbors to consider for local structure
    #     min_dist=0.1,            # How tightly to pack points together
    #     n_components=2,          # Target dimensions
    #     random_state=42,
    # ).fit(all_latent_states[:,0,:])
    pca = PCA(n_components=2)
    pca.fit(all_latent_states)

    # # Now transform the trajectories
    embedding_trajectories = pca.transform(all_latent_states).reshape(n_patients, n_timepoints, 2)
    # embedding_trajectories = np.concatenate([pca.transform(all_latent_states[:,i,:]).reshape(n_patients,1, 2) for i in range(n_timepoints)], axis = 1)
    
    # For now, we only take the mean over the trajectory samples
    # embedding_trajectories = np.mean(embedding_trajectories, axis=0)
    # Now transform the trajectories
    # embedding_trajectories = np.concatenate([mapper.transform(all_latent_states[:,i,:]).reshape(n_patients,1, 2) for i in range(n_timepoints)], axis = 1)
    # # For now, we only take the mean over the trajectory samples
    # embedding_trajectories = np.mean(embedding_trajectories, axis=0)
    print("UMAP embedding created.")
    # Step 2: Pre-generate plot images for hover tooltips
    print("Step 2: Generating time-series plot images for each data point...")
    hover_images = []
    num_samples = reconstructions.shape[1]
    ts_to_predict = time_steps_to_predict.cpu().numpy()
    ts_gt = time_steps

    # Define the hover template. This is HTML that tells Plotly what to show.
    hovertemplate = """
    <b>Patient ID:</b> %{customdata[0]}<br>
    <b>Label Value:</b> %{customdata[1]:.3f}<br><br>
    %{customdata[2]}
    <extra></extra>
    """ # <extra></extra> hides the trace name

    def plot_to_base64_url(index):
        """Generates a PNG of the time series plot for a given index and returns a data URL."""
        fig_hover, ax_hover = plt.subplots(figsize=(4, 2.5), dpi=100)
        
        # Get the data for the specific patient
        reconstruction_mean = np.mean(reconstructions[:, index, :, 0], axis=0)
        
        ax_hover.plot(ts_to_predict, reconstruction_mean, label='Reconstruction', color='red')
        try:
            ground_truth = data[index, :, 0]
            ax_hover.plot(time_steps, ground_truth, label='Ground Truth', color='black', linestyle='--')
        except:
            pass
        ax_hover.set_title(f'Patient ID: {df.iloc[index]["patient_id"]}', fontsize=10)
        ax_hover.set_xlabel('Time', fontsize=8)
        ax_hover.set_ylabel('Value', fontsize=8)
        ax_hover.legend(fontsize=8)
        ax_hover.grid(True, alpha=0.5)
        plt.tight_layout()

        # Save plot to an in-memory buffer and encode it
        buf = io.BytesIO()
        fig_hover.savefig(buf, format="png")
        plt.close(fig_hover) # IMPORTANT: Close figure to free memory
        
        encoded_image = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{encoded_image}"
    
    

    # Step 3: Prepare data for Plotly using a Pandas DataFrame
    print("Step 3: Assembling data into a DataFrame for Plotly...")
    # Create a DataFrame for each patient and their trajectory
    df_list = []
    for i in range(n_patients):
        df_patient = pd.DataFrame({
            'x': embedding_trajectories[i, :, 0],
            'y': embedding_trajectories[i, :, 1],
            'time': ts_to_predict,
            'patient_id': patient_id[i],
            'label': all_labels[i]
        })
        df_list.append(df_patient)
    df = pd.concat(df_list, ignore_index=True)

    # Step 4: Create the interactive scatter plot
    print("Step 4: Building the interactive figure...")
    fig = go.Figure()
    # Add trajectories for each patient
    for i in range(n_patients):
        df_patient = df[df['patient_id'] == patient_id[i]]
        
        # fig.add_trace(go.Scatter(
        #     x=df_patient['x'], y=df_patient['y'],
        #     mode='lines',
        #     line=dict(color=px.colors.qualitative.Plotly[int(all_labels[i]) % len(px.colors.qualitative.Plotly)], width=1),
        #     name=f"Patient {patient_id[i]}"
        # ))
        # Add markers for the start of the trajectory
        fig.add_trace(go.Scatter(
            x=[df_patient['x'].iloc[0]], y=[df_patient['y'].iloc[0]],
            mode='markers',
            marker=dict(
                color=px.colors.qualitative.Plotly[int(all_labels[i]) % len(px.colors.qualitative.Plotly)],
                size=8,
                symbol='circle'
            ),
            name=f"Start Patient {patient_id[i]}"
        ))
    fig.update_traces(hoverinfo="none", hovertemplate=None)
    fig.update_layout(title='Interactive UMAP of Latent Space (Hover for Details and Plots)')

   # --- NEW: Plot ACCURATE Rotated Ellipses with Explicit Angle ---
    if hasattr(model, 'prior_means') and hasattr(model, 'prior_logvars'):
        print("Projecting and plotting GMM prior modes...")
        
        # 1. Get Model Parameters in U-Space (Disentangled)
        # prior_means are in U-space
        means_u = model.prior_means.detach().cpu().numpy() 
        # prior_logvars are in U-space
        vars_u = np.exp(model.prior_logvars.detach().cpu().numpy()) 

        # 2. Map U-Space -> Z-Space (Physical)
        if hasattr(model, 'latent_rotation'):
            # Model has Linear Flow: z = R^-1 * u
            # We need the Inverse of the rotation weight
            # Weight shape is [Out, In] -> [Dim, Dim]
            W = model.latent_rotation.weight.detach().cpu().numpy() # Shape [Dim, Dim]
            try:
                W_inv = np.linalg.inv(W)
            except np.linalg.LinAlgError:
                print("Warning: Rotation matrix is singular. Using Identity for visualization.")
                W_inv = np.eye(W.shape[0])
            # Transform Means: z = u @ (R^-1).T
            means_z = means_u @ W_inv.T
            
            # Transform Covariances: 
            # Cov(z) = W^-1 * Cov(u) * (W^-1).T
            covs_z = []
            for i in range(len(vars_u)):
                Sigma_u = np.diag(vars_u[i])
                Sigma_z = W_inv @ Sigma_u @ W_inv.T
                covs_z.append(Sigma_z)
        else:
            # Standard Diagonal Model (No Rotation): z = u
            means_z = means_u
            covs_z = [np.diag(v) for v in vars_u]
        # 3. Project Z-Space -> 2D PCA Space
        # PCA was fitted on Z-space data
        W_pca = pca.components_ # Shape [2, n_latents]
        
        # Project Means
        modes_2d = pca.transform(means_z)

        # Loop over each cluster mode
        for i in range(len(modes_2d)):
            # A. Project the Covariance Matrix to 2D
            # Sigma_2d = W_pca @ Sigma_z @ W_pca.T
            Sigma_z = covs_z[i]
            cov_2d = W_pca @ Sigma_z @ W_pca.T
            
            # B. Get the Curve Coordinates using the Helper Function
            # Note: Ensure get_ellipse_curve is defined in your utils or scope
            x_elli, y_elli = get_ellipse_curve(modes_2d[i], cov_2d, scale=2.0)
            
            # C. Plot the Curve
            fig.add_trace(go.Scatter(
                x=x_elli,
                y=y_elli,
                mode='lines',
                line=dict(color='black', width=2, dash='dot'),
                name=f'Mode {i} (95% CI)',
                hoverinfo='skip', # purely visual
                showlegend=False
            ))

        # D. Plot the Centers
        fig.add_trace(go.Scatter(
            x=modes_2d[:, 0],
            y=modes_2d[:, 1],
            mode='markers',
            marker=dict(symbol='x', size=12, color='black', line=dict(width=2, color='white')),
            name='GMM Centers',
            hoverinfo='skip'
        ))
    # -----------------------------------------
    if hasattr(model, 'prior_means') and hasattr(model, 'prior_cov_tril'):
        print("Projecting and plotting Full-Rank GMM prior modes...")
        
        # 1. Get Model Parameters (CPU/Numpy)
        prior_means_np = model.prior_means.detach().cpu().numpy()
        
        # Get Cholesky Factors L
        # We must replicate the "positive diagonal" logic from the model class
        L_params = model.prior_cov_tril.detach().cpu()
        latent_dim = L_params.shape[1]
        
        # Create mask for diagonal
        diag_mask = torch.eye(latent_dim).unsqueeze(0).repeat(len(L_params), 1, 1)
        
        # Reconstruct valid L: Lower triangle + Exp(Diagonal)
        # L = L_tril * (1 - mask) + exp(L_diag) * mask
        L_tril = torch.tril(L_params)
        L_valid = L_tril * (1 - diag_mask) + torch.exp(L_tril * diag_mask) * diag_mask
        
        # Compute Full Covariance: Sigma = L @ L.T
        # Shape: [n_components, latent_dim, latent_dim]
        prior_covs_np = (L_valid @ L_valid.transpose(1, 2)).numpy()
        
        # 2. PCA Projection Matrix (W)
        # pca.components_ has shape [n_components_pca (2), n_latents]
        W = pca.components_ 
        
        # 3. Project Means to 2D
        # pca.transform handles the centering automatically for means
        modes_2d = pca.transform(prior_means_np)

        # 4. Loop over each cluster to Project Covariance and Plot
        for i in range(len(modes_2d)):
            # A. Project the Full Covariance Matrix to 2D
            # Formula: Cov_2d = W @ Sigma_3d @ W.T
            # Shape: [2, D] @ [D, D] @ [D, 2] -> [2, 2]
            Sigma_full = prior_covs_np[i]
            cov_2d = W @ Sigma_full @ W.T
            
            # B. Get the Curve Coordinates 
            # (Assumes you have the get_ellipse_curve function defined)
            x_elli, y_elli = get_ellipse_curve(modes_2d[i], cov_2d, scale=2.0)
            
            # C. Plot the Curve
            fig.add_trace(go.Scatter(
                x=x_elli,
                y=y_elli,
                mode='lines',
                line=dict(color='black', width=2, dash='dot'),
                name=f'Mode {i} (95% CI)',
                hoverinfo='skip',
                showlegend=False
            ))

        # D. Plot the Centers
        fig.add_trace(go.Scatter(
            x=modes_2d[:, 0],
            y=modes_2d[:, 1],
            mode='markers',
            marker=dict(symbol='x', size=12, color='black', line=dict(width=2, color='white')),
            name='GMM Centers',
            hoverinfo='skip'
        ))
        fig.update_traces(hoverinfo="none", hovertemplate=None)
    app = Dash(__name__)
    app.layout = html.Div([
        # The graph component
        dcc.Graph(id="umap-graph", figure=fig, clear_on_unhover=True),
        # The custom tooltip component
        dcc.Tooltip(id="graph-tooltip"),
    ])

    @app.callback(
        Output("graph-tooltip", "show"),
        Output("graph-tooltip", "bbox"),
        Output("graph-tooltip", "children"),
        Input("umap-graph", "hoverData"),
    )
    def display_hover_plot(hoverData):
        # If not hovering, hide the tooltip
        if hoverData is None:
            return False, no_update, no_update

        # Get the index and bounding box of the hovered point
        hover_point = hoverData["points"][0]
        bbox = hover_point["bbox"]
        point_index = hover_point["pointNumber"]

        # Generate the plot for the hovered point
        image_url = plot_to_base64_url(point_index)
        patient_info = df.iloc[point_index]

        # Define the content of the tooltip
        children = html.Div([
            html.Img(src=image_url, style={"width": "100%"}),
            html.Hr(),
            html.P(f"Patient ID: {patient_info['patient_id']}"),
            html.P(f"Label Value: {patient_info['label']:.3f}"),
        ], style={'width': '300px', 'white-space': 'normal'})

        return True, bbox, children
    
    print("Step 5: Displaying the interactive plot. You can zoom, pan, and hover!")
    # This will open in your browser or display in a notebook
    app.run()