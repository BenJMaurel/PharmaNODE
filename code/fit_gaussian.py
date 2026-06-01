###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import os
import sys
import matplotlib
# matplotlib.use('TkAgg')
# matplotlib.use('Agg')
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
from sklearn.mixture import GaussianMixture
import torch
import torch.nn as nn
from torch.nn.functional import relu
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
import numpy as np
import matplotlib.colors as mcolors

import lib.utils as utils
from lib.plotting import *

from lib.rnn_baselines import *
from lib.ode_rnn import *
from lib.create_latent_ode_model import create_LatentODE_model
from lib.parse_datasets import parse_datasets
from lib.ode_func import ODEFunc, ODEFunc_w_Poisson
from lib.diffeq_solver import DiffeqSolver


from lib.utils import compute_loss_all_batches
# matplotlib.use('TkAgg')
# Generative model for noisy data based on ODE
parser = argparse.ArgumentParser('Latent ODE')
parser.add_argument('-n',  type=int, default=100, help="Size of the dataset")
parser.add_argument('--lr',  type=float, default=1e-2, help="Starting learning rate.")
parser.add_argument('-b', '--batch-size', type=int, default=2000)
parser.add_argument('--viz', action='store_true', help="Show plots while training")

parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="ID of the experiment to load for evaluation. If None, run a new experiment.")
parser.add_argument('-r', '--random-seed', type=int, default=1991, help="Random_seed")
parser.add_argument('--experiment', type = str, default = None, help="Fix experiment number for reproducibility")

parser.add_argument('--dataset', type=str, default='periodic', help="Dataset to load. Available: physionet, activity, hopper, periodic")
parser.add_argument('-s', '--sample-tp', type=float, default=None, help="Number of time points to sub-sample."
	"If > 1, subsample exact number of points. If the number is in [0,1], take a percentage of available points per time series. If None, do not subsample")

parser.add_argument('-c', '--cut-tp', type=int, default=None, help="Cut out the section of the timeline of the specified length (in number of points)."
	"Used for periodic function demo.")

parser.add_argument('--quantization', type=float, default=0.1, help="Quantization on the physionet dataset."
	"Value 1 means quantization by 1 hour, value 0.1 means quantization by 0.1 hour = 6 min")

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
#####################################################################################################

def visualize_gmm_on_pca(latent_means, gmm_model, title="GMM Clusters in Latent Space (PCA Projection)"):
    """
    Visualizes GMM clusters projected onto the first two principal components.

    Args:
        latent_means (np.ndarray): The high-dimensional latent data.
        gmm_model (GaussianMixture): The fitted GMM model.
        title (str): The title for the plot.
    """
    # Step 1: Reduce dimensionality with PCA (ACP)
    print("Projecting latent means onto 2D using PCA...")
    pca = PCA(n_components=2)
    latent_means_pca = pca.fit_transform(latent_means)
    
    # Predict the cluster for each data point
    labels = gmm_model.predict(latent_means)
    
    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Use a distinct color for each cluster
    colors = plt.cm.viridis(np.linspace(0, 1, gmm_model.n_components))

    # Step 2: Create a scatter plot of the projected data
    print("Creating scatter plot of projected data points...")
    for i in range(gmm_model.n_components):
        points = latent_means_pca[labels == i]
        ax.scatter(points[:, 0], points[:, 1], s=15, color=colors[i], alpha=0.6, label=f'Cluster {i}')

    # Step 3: Draw the ellipses representing the GMM components
    print("Drawing ellipses for each Gaussian component...")
    for i in range(gmm_model.n_components):
        # Project the GMM component's covariance matrix into the PCA space
        # C_pca = P^T * C_orig * P, where P is the matrix of principal components
        cov_orig = gmm_model.covariances_[i]
        p_components = pca.components_
        cov_pca = p_components @ cov_orig @ p_components.T
        
        # Project the GMM component's mean into the PCA space
        mean_pca = pca.transform(gmm_model.means_[i].reshape(1, -1)).flatten()
        
        # Calculate eigenvalues and eigenvectors of the projected covariance
        # to determine the ellipse's shape and orientation
        eigenvalues, eigenvectors = np.linalg.eigh(cov_pca)
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        
        # Use eigenvalues to set the width and height of the ellipse
        # (2 standard deviations, covering ~95% of the data for that component)
        width, height = 2 * np.sqrt(5.991 * eigenvalues) # 5.991 is the chi-squared value for 2 dof at p=0.05
        
        # Create and add the ellipse patch
        ellipse = Ellipse(xy=mean_pca, width=width, height=height, angle=angle,
                          facecolor=colors[i], alpha=0.25)
        ax.add_patch(ellipse)
        
        # Plot the center of the Gaussian
        ax.scatter(mean_pca[0], mean_pca[1], s=150, c='white', edgecolors='black', marker='X', zorder=5)


    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Principal Component 1", fontsize=12)
    ax.set_ylabel("Principal Component 2", fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    output_filename = "gmm_clusters_pca.png"
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    plt.close(fig)

def generate_with_gmm_prior(latent_means_numpy, n_components=5, num_samples=64):
    """
    Generates new samples from a VAE using a GMM-fitted prior.

    Args:
        vae_model: The trained VAE model. Must have a .decode() method.
        latent_means_tensor: A tensor of shape (num_data_points, latent_dim) 
                             containing the latent means of the training data.
        n_components (int): The number of clusters to use for the GMM.
        num_samples (int): The number of new samples to generate.

    Returns:
        A tensor containing the generated samples.
    """
    print("--- Starting GMM-based generation ---")
    
    # Ensure the model is on the correct device and in eval mode

    # 1. Prepare data: Convert latent means tensor to NumPy array
    print("Step 1: Converting latent data to NumPy...")
    # 2. Fit the GMM
    print(f"Step 2: Fitting GMM with {n_components} components...")
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(latent_means_numpy)
    print("GMM fit complete.")
	# --- Call the visualization function with your data and fitted model ---
    visualize_gmm_on_pca(latent_means_numpy, gmm)
    # 3. Sample new latent vectors from the GMM
    print(f"Step 3: Sampling {num_samples} new latent vectors from the GMM...")
    new_latent_vectors_numpy, _ = gmm.sample(n_samples=num_samples)

    # 4. Convert back to tensors and decode
    print("Step 4: Decoding vectors to generate new data...")
    new_latent_vectors_tensor = torch.from_numpy(new_latent_vectors_numpy).float().to(device)
    return new_latent_vectors_tensor



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
	all_labels = []
	# for itr in range(1, num_batches):
	for itr in range(0,1):
		with torch.no_grad():
			data_dict = utils.get_next_batch(data_obj["train_dataloader"])
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
			if isinstance(model, LatentODE):
				# sample at the original time points
				time_steps_to_predict = utils.linspace_vector(time_steps[0], torch.tensor(24.), 100).to(device)
			reconstructions, info = model.get_reconstruction(time_steps_to_predict, 
				observed_data, observed_time_steps, dose = dose, static = static, mask = observed_mask, n_traj_samples = 100)
			all_labels.append(static[:,1])
			# all_labels.append(auc_red[:,0])
			latent_z0_var = info["first_point"][1]
			latent_z0_mean = info["first_point"][0]
			all_latent_z0_means.append(latent_z0_mean.squeeze(0))
			all_latent_z0_var.append(latent_z0_var.squeeze(0))
			all_data.append(data)
			all_reconstructions.append(reconstructions)
			
	all_latent_z0_means = np.concatenate(all_latent_z0_means, axis=0)
	all_latent_z0_var = np.concatenate(all_latent_z0_var, axis=0)

	NUM_COMPONENTS = 2 # Example: maybe you have 5 distinct types of data

	print(f"Fitting GMM with {NUM_COMPONENTS} components...")
	print(generate_with_gmm_prior(all_latent_z0_means, n_components=2))
	print("GMM fitting complete!")
# gmm = GaussianMixture(n_components=NUM_COMPONENTS, covariance_type='full', random_state=42)
# gmm.fit(all_latent_z0_means)


