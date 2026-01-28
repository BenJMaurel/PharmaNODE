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
from fit_gaussian import generate_with_gmm_prior

from lib.utils import compute_loss_all_batches
matplotlib.use('TkAgg')

# Generative model for noisy data based on ODE
parser = argparse.ArgumentParser('Latent ODE GMM')
parser.add_argument('-n',  type=int, default=100, help="Size of the dataset")
parser.add_argument('--lr',  type=float, default=1e-2, help="Starting learning rate.")
parser.add_argument('-b', '--batch-size', type=int, default=2000)
parser.add_argument('--viz', action='store_true', help="Show plots while training")

parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="ID of the experiment to load for evaluation. If None, run a new experiment.")
parser.add_argument('--experiment', type = str, default = None, help="Fix experiment number for reproducibility")
parser.add_argument('-r', '--random-seed', type=int, default=1991, help="Random_seed")

parser.add_argument('--dataset', type=str, default='periodic', help="Dataset to load. Available: physionet, activity, hopper, periodic")
parser.add_argument('-s', '--sample-tp', type=float, default=None, help="Number of time points to sub-sample."
	"If > 1, subsample exact number of points. If the number is in [0,1], take a percentage of available points per time series. If None, do not subsample")

parser.add_argument('-c', '--cut-tp', type=int, default=None, help="Cut out the section of the timeline of the specified length (in number of points)."
	"Used for periodic function demo.")

parser.add_argument('--quantization', type=float, default=0.1, help="Quantization on the physionet dataset."
	"Value 1 means quantization by 1 hour, value 0.1 means quantization by 0.1 hour = 6 min")

parser.add_argument('--use-gmm', action='store_true', help="Run Latent ODE with clusters inside latent space model")
parser.add_argument('--latent-ode', action='store_true', help="Run Latent ODE seq2seq model")

parser.add_argument('--z0-encoder', type=str, default='odernn', help="Type of encoder for Latent ODE model: odernn or rnn")

parser.add_argument('--classic-rnn', action='store_true', help="Run RNN baseline: classic RNN that sees true points at every point. Used for interpolation only.")
parser.add_argument('--rnn-cell', default="gru", help="RNN Cell type. Available: gru (default), expdecay")
parser.add_argument('--input-decay', action='store_true', help="For RNN: use the input that is the weighted average of impirical mean and previous value (like in GRU-D)")

parser.add_argument('--ode-rnn', action='store_true', help="Run ODE-RNN baseline: RNN-style that sees true points at every point. Used for interpolation only.")

parser.add_argument('--rnn-vae', action='store_true', help="Run RNN baseline: seq2seq model with sampling of the h0 and ELBO loss.")

parser.add_argument('-l', '--latents', type=int, default=6, help="Size of the latent state")
parser.add_argument('--n_components', type=int, default=4, help="Number of Gaussian within the latent space")
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

#####################################################################################################

if __name__ == '__main__':  
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
            print("Poisson process likelihood not implemented for RNN-VAE: ignoring --poisson") 
        # Create RNN-VAE model
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
    all_labels = []
    all_latent_trajectories = []
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
            dataset = data_dict['dataset_number']
            patient_id = data_dict['patient_id']
            device = get_device(time_steps)
            if isinstance(model, LatentODE) or isinstance(model, LatentODEGMM):
                # sample at the original time points
                # time_steps_to_predict = utils.linspace_vector(time_steps[0], torch.tensor(24.), 100).to(device)
                time_steps_to_predict = torch.tensor([0.0, 0.33, 0.67, 1., 1.5, 2.0, 3.0, 4.0, 6.0, 9.0, 12.0, 24.0], device=device)
                # time_steps_to_predict = torch.arange(0.5, 168.0 + 0.1, 0.1, device=device)
            reconstructions, info = model.get_reconstruction(time_steps_to_predict, 
                observed_data, observed_time_steps, dose = dose, static = static, mask = observed_mask, n_traj_samples = 100)
            all_labels.append(static[:,1])
            # all_labels.append(auc_red[:,0])
            latent_z0_var = info["first_point"][1]
            latent_z0_mean = info["first_point"][0]
            latent_traj = info["latent_traj"]
            all_latent_trajectories.append(latent_traj)
            all_latent_z0_means.append(latent_z0_mean.squeeze(0))
            all_latent_z0_var.append(latent_z0_var.squeeze(0))
            all_data.append(data)
            all_reconstructions.append(reconstructions)
    all_latent_z0_means = np.concatenate(all_latent_z0_means, axis=0)
    all_latent_z0_var = np.concatenate(all_latent_z0_var, axis=0)
    all_latent_trajectories = np.concatenate(all_latent_trajectories, axis=1)
    n_traj_samples, n_patients, n_timepoints, n_latents = all_latent_trajectories.shape
    all_latent_states = np.mean(all_latent_trajectories, axis = 0)
    all_latent_states = all_latent_states.reshape(-1, n_latents)
    NUM_COMPONENTS = 2 # Example: maybe you have 5 distinct types of data
    # means_z0 = torch.zeros(10)
    # sigma_z0 = torch.ones(10)
    samples = generate_with_gmm_prior(all_latent_z0_means, n_components=NUM_COMPONENTS, num_samples = 100)
    n_traj_samples = len(samples[0])
    mean = torch.mean(samples)
    std = torch.std(samples)
    # to_sample = utils.sample_standard_gaussian(means_z0, sigma_z0)
    # samples = to_sample.sample([n_traj_samples, 1, args.l]).squeeze(-1)
    # samples = (torch.randn(1, n_traj_samples, 10)* std) +mean # 10 being the dimension of the latent space
    time_steps_to_predict = torch.tensor([0.0, 0.33, 0.67, 1., 1.5, 2.0, 3.0, 4.0, 6.0, 9.0, 12.0, 24.0], device=device)
    # time_steps_to_predict = utils.linspace_vector(torch.tensor(0.0), torch.tensor(24.), 100).to(device)
    # time_steps_to_predict = torch.arange(0.5, 24.0 + 0.1, 0.1, device=device)
    sol_y = model.diffeq_solver(samples.unsqueeze(0), time_steps_to_predict)
    pred_x = model.decoder(sol_y)
    # test = model.sample_traj_from_prior(time_steps_to_predict, n_traj_samples = 100)
    # test = test.detach().numpy()
    scaling_factor = data_obj['max_out']['max_out'][0]
    if 'best_lambda' in data_obj['max_out'].keys():
        new_test = inv_boxcox(pred_x.detach().numpy(), data_obj['max_out']['best_lambda'][0])
        new_test = new_test*scaling_factor
        data = inv_boxcox(data, data_obj['max_out']['best_lambda'][0])
        reconstructions = inv_boxcox(reconstructions, data_obj['max_out']['best_lambda'][0])
        reconstructions = torch.nan_to_num(reconstructions, nan=0.0)
        reconstructions = reconstructions.detach().numpy()*scaling_factor
        reconstructions = np.mean(reconstructions[:, :, :], axis=0)
    # all_latent_z0_means = samples
    reconstructions = np.concatenate((reconstructions.squeeze(-1), new_test.squeeze(0).squeeze(-1)), axis = 0)
    time_steps_col = time_steps_to_predict.detach().cpu().numpy().reshape(-1, 1)
    new_test_2d = new_test.squeeze().T
    combined_data_to_save = np.concatenate((time_steps_col, new_test_2d), axis=1)
    
    headers = ['time'] + [f'sample_{i}' for i in range(new_test_2d.shape[1])]
    
    df_to_save = pd.DataFrame(combined_data_to_save, columns=headers)
    csv_filename = 'exp_run_all/' + str(args.experiment) +'/new_test_data.csv'
    df_to_save.to_csv(csv_filename, index=False)
    print(f"--- Data saved to {csv_filename} ---")

    # 3. Reload from CSV
    # reloaded_df = pd.read_csv(csv_filename)
    # print(f"--- Data reloaded from {csv_filename} ---")
    # # 4. (Optional) Restore data to original shapes
    # # Restore time_steps (as numpy array)
    # reloaded_time_steps = reloaded_df['time'].values
    # reloaded_new_test_2d = reloaded_df.drop('time', axis=1).values
    print("Step 1: Running UMAP on the latent means...")
    # mapper = umap.UMAP(
    #     n_neighbors=20,          # How many neighbors to consider for local structure
    #     min_dist=0.1,            # How tightly to pack points together
    #     n_components=2,          # Target dimensions
    #     random_state=42,
    # ).fit(all_latent_z0_means)
    pca = PCA(n_components=2)
    pca.fit(all_latent_states)
    # embedding_real = pca.transform(all_latent_z0_means)
    # embedding = pca.transform(samples)
    n_traj_samples, n_patients, n_timepoints, n_latents = sol_y.shape
    embedding = pca.transform(sol_y.detach().numpy().reshape(-1, n_latents)).reshape(n_patients, n_timepoints, 2)
    # all_latent_z0_means = samples
    print("UMAP embedding created.")
    # Step 2: Pre-generate plot images for hover tooltips
    print("Step 2: Generating time-series plot images for each data point...")
    hover_images = []
    num_samples = samples.shape[0]
    ts_to_predict = time_steps_to_predict.cpu().numpy()
    # Step 3: Prepare data for Plotly using a Pandas DataFrame
    print("Step 3: Assembling data into a DataFrame for Plotly...")
    # embedding = np.concatenate((embedding_real, embedding), axis = 0)
    df_list = []

    for i in range(n_patients):
        df_patient = pd.DataFrame({
            'x': embedding[i, :, 0],
            'y': embedding[i, :, 1],
            'time': ts_to_predict,
            'patient_id': i
        })
        df_list.append(df_patient)
    df = pd.concat(df_list, ignore_index=True)
    # Step 4: Create the interactive scatter plot
    print("Step 4: Building the interactive figure...")
    # Define the hover template. This is HTML that tells Plotly what to show.
    hovertemplate = """
    <b>Patient ID:</b> %{customdata[0]}<br>
    <b>Label Value:</b> %{customdata[1]:.3f}<br><br>
    %{customdata[2]}
    <extra></extra>
    """ # <extra></extra> hides the trace name
    # Helper function to generate a plot and convert it to a Base64 image URL
    def plot_to_base64_url(index):
        """Generates a PNG of the time series plot for a given index and returns a data URL."""
        fig_hover, ax_hover = plt.subplots(figsize=(4, 2.5), dpi=100)
        # Get the data for the specific patient
        reconstruction_mean = reconstructions[index, :]
        
        ax_hover.plot(ts_to_predict, reconstruction_mean, label='Reconstruction', color='red')
        # try:
        #     ground_truth = data[index, :, 0]
        #     ax_hover.plot(time_steps, ground_truth, label='Ground Truth', color='black', linestyle='--')
        # except:
        #     pass
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
    fig = go.Figure()
    # Add trajectories for each patient
    for i in range(n_patients):
        df_patient = df[df['patient_id'] == i]
        
        fig.add_trace(go.Scatter(
            x=df_patient['x'], y=df_patient['y'],
            mode='lines',
            name=f"Patient {i}"
        ))
        # Add markers for the start of the trajectory
        fig.add_trace(go.Scatter(
            x=[df_patient['x'].iloc[0]], y=[df_patient['y'].iloc[0]],
            mode='markers',
            marker=dict(
                size=8,
                symbol='circle'
            ),
            name=f"Start Patient {i}"
        ))
    fig.update_traces(hoverinfo="none", hovertemplate=None)
    fig.update_layout(title='Interactive UMAP of Latent Space (Hover for Details and Plots)')

    # Define the callback to update the toolti
    app = Dash(__name__)
    app.layout = html.Div([
        # The graph component
        dcc.Graph(id="umap-graph", figure=fig, clear_on_unhover=True),
        # The custom tooltip component
        dcc.Tooltip(id="graph-tooltip"),
    ])
    # Define the callback to update the tooltip
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
        ], style={'width': '300px', 'white-space': 'normal'})
        return True, bbox, children
    print("Step 5: Displaying the interactive plot. You can zoom, pan, and hover!")
    # This will open in your browser or display in a notebook
    app.run(port=8051)
