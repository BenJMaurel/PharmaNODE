import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot
import matplotlib.pyplot as plt


from torch.utils.tensorboard import SummaryWriter

import time
import datetime
import argparse
import numpy as np
import pandas as pd
from random import SystemRandom
from sklearn import model_selection
import random

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

from lib.utils import compute_loss_all_batches, initialize_gmm_with_kmeans, initialize_gmm_with_kmeans_v
# Generative model for noisy data based on ODE
parser = argparse.ArgumentParser('Latent ODE')
parser.add_argument('-n',  type=int, default=100, help="Size of the dataset")
parser.add_argument('--niters', type=int, default=1000)
parser.add_argument('--lr',  type=float, default=1e-2, help="Starting learning rate.")
parser.add_argument('-b', '--batch-size', type=int, default=1)
parser.add_argument('--viz', action='store_true', help="Show plots while training")

parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="ID of the experiment to load for evaluation. If None, run a new experiment.")
parser.add_argument('-r', '--random-seed', type=int, default=1991, help="Random_seed")

parser.add_argument('--dataset', type=str, default='periodic', help="Dataset to load. Available: physionet, activity, hopper, periodic")
parser.add_argument('-s', '--sample-tp', type=float, default=None, help="Number of time points to sub-sample."
	"If > 1, subsample exact number of points. If the number is in [0,1], take a percentage of available points per time series. If None, do not subsample")

parser.add_argument('-c', '--cut-tp', type=int, default=None, help="Cut out the section of the timeline of the specified length (in number of points)."
	"Used for periodic function demo.")

parser.add_argument('--quantization', type=float, default=0.1, help="Quantization on the physionet dataset."
	"Value 1 means quantization by 1 hour, value 0.1 means quantization by 0.1 hour = 6 min")

parser.add_argument('--use_film', action='store_true', help="Run End-to-End FiLM Extrapolation training")
parser.add_argument('--use_time', action='store_true', help="Run FiLM + Time Extrapolation training")
parser.add_argument('--latent-ode', action='store_true', help="Run Latent ODE seq2seq model")
parser.add_argument('--use_gmm', action='store_true', help="Run Latent ODE with clusters inside latent space model")
parser.add_argument('--use_flow', action='store_true', help="Run Latent ODE with clusters inside latent space model")
parser.add_argument('--use_gmm_v', action='store_true', help="Run Latent ODE with clusters inside latent space model")
parser.add_argument('--z0-encoder', type=str, default='odernn', help="Type of encoder for Latent ODE model: odernn or rnn")

parser.add_argument('--classic-rnn', action='store_true', help="Run RNN baseline: classic RNN that sees true points at every point. Used for interpolation only.")
parser.add_argument('--rnn-cell', default="gru", help="RNN Cell type. Available: gru (default), expdecay")
parser.add_argument('--input-decay', action='store_true', help="For RNN: use the input that is the weighted average of impirical mean and previous value (like in GRU-D)")

parser.add_argument('--ode-rnn', action='store_true', help="Run ODE-RNN baseline: RNN-style that sees true points at every point. Used for interpolation only.")

parser.add_argument('--rnn-vae', action='store_true', help="Run RNN baseline: seq2seq model with sampling of the h0 and ELBO loss.")

parser.add_argument('-l', '--latents', type=int, default=6, help="Size of the latent state")
parser.add_argument('-nc', '--n_components', type=int, default=4, help="Number of Gaussian within the latent space")
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
parser.add_argument('--noise-weight', type=float, default=0.04, help="Noise amplitude for generated traejctories")
parser.add_argument('--seed', type = int, default = 15, help="Fix seed for reproducibility")
parser.add_argument('--experiment', type = str, default = None, help="Fix experiment number for reproducibility")
parser.add_argument('--patience', type=int, default=10000, help='Patience for early stopping')

parser.add_argument('--smoothing_factor', type=float, default=0.99, help='Smoothing factor for EMA of test MSE')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
file_name = os.path.basename(__file__)[:-3]
utils.makedirs(args.save)

#####################################################################################################

if __name__ == '__main__':
	# torch.manual_seed(args.random_seed)
	# np.random.seed(args.random_seed)
	torch.manual_seed(args.seed)
	seed = np.random.seed(args.seed)
	experimentID = args.load

	if experimentID is None:
		# Make a new experiment ID
		experimentID = int(args.experiment)
		if experimentID is None:
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
	if args.use_film:
		from lib.read_tacro import extract_gen_tac_film, TacroFilmDataset, collate_fn_tacro_film,
		from torch.utils.data import DataLoader
		ckpt_path = os.path.join(args.save, "experiment_film_" + str(experimentID) + '.ckpt')
        # Load paired dataset
		data_dict_train, scale = extract_gen_tac_film(file_path=[f"exp_run_all/exp_film_run/{args.experiment}/virtual_cohort_film_train.csv"])
		data_dict_test, _ = extract_gen_tac_film(file_path=[f"exp_run_all/exp_film_run/{args.experiment}/virtual_cohort_film_test.csv"])
		
		max_out = {}
		max_out['best_lambda'] = np.array(scale[1])
		max_out['max_out']=  np.array(scale[0])
		dataset_train = TacroFilmDataset(data_dict_train)
		dataset_test = TacroFilmDataset(data_dict_test)
        
		train_dataloader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: collate_fn_tacro_film(x, device))
		test_dataloader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: collate_fn_tacro_film(x, device))
		
		data_obj = {
            "input_dim": 1,
            "train_dataloader": utils.inf_generator(train_dataloader),
            "test_dataloader": utils.inf_generator(test_dataloader),
            "n_train_batches": len(train_dataloader),
            "n_test_batches": len(test_dataloader),
            "max_out" : max_out
        }
	else:
        # Your normal dataset parsing
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

	if args.viz:
		viz = Visualizations(device)

	##################################################################
	
	#Load checkpoint and evaluate the model
	if args.load is not None:
		utils.get_ckpt_model(ckpt_path, model, device)
		# exit()
	##################################################################
	# Training

	log_path = "logs/" + file_name + "_" + str(experimentID) + ".log"
	if not os.path.exists("logs/"):
		utils.makedirs("logs/")
	logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
	logger.info(input_command)

	optimizer = optim.Adamax(model.parameters(), lr=args.lr)

	num_batches = data_obj["n_train_batches"]
	
	tensorboard_log_dir = os.path.join(args.save, "runs", str(experimentID))
	writer = SummaryWriter(tensorboard_log_dir)
	print(f"TensorBoard logs will be saved to: {tensorboard_log_dir}")

print("----------------------------------------------------------------")
print("Starting GMM Warm-Start Strategy")
print("Phase 1: Pre-training Encoder on Reconstruction Loss (MSE) only")
print("----------------------------------------------------------------")


if args.use_gmm or args.use_gmm_v or args.use_flow:
	n_warmup_epochs = 20
	num_batches = data_obj["n_train_batches"]

	for warm_itr in range(n_warmup_epochs * num_batches):
		optimizer.zero_grad()
		utils.update_learning_rate(optimizer, decay_rate=0.999, lowest=args.lr/10)
		
		batch_dict = utils.get_next_batch(data_obj["train_dataloader"])
    
		if args.use_film:
			
			train_res = model.compute_film_losses(batch_dict, n_traj_samples=3, scaler = max_out)
            # Optional: Print statement to watch the dual loss
			# if itr % num_batches == 0:
			# 	print(f"Epoch {itr//num_batches} | V1 Loss: {train_res['rec_loss_v1']:.4f} | V2 Loss: {train_res['rec_loss_v2']:.4f}")
		else:
			train_res = model.compute_all_losses(batch_dict, n_traj_samples=3)
            
		train_res["loss"].backward()
		optimizer.step()
		
		if warm_itr % num_batches == 0:
			print(f"[Warmup] Epoch {warm_itr // num_batches}/{n_warmup_epochs} - MSE: {train_res['mse'].item():.4f}")
	# 2. Run K-Means Initialization
	print("----------------------------------------------------------------")
	print("Phase 2: Initializing GMM Priors with K-Means on Pre-trained Embeddings")
	print("----------------------------------------------------------------")
	if args.use_gmm:
		initialize_gmm_with_kmeans(model, data_obj, device, n_batches=data_obj["n_test_batches"])
	elif args.use_gmm_v: 
		initialize_gmm_with_kmeans(model, data_obj, device, n_batches=data_obj["n_test_batches"])
	print("----------------------------------------------------------------")
	print("Phase 3: Starting Full Training (Reconstruction + KL + Diversity)")
	print("----------------------------------------------------------------")

### Early Stopping Initialization ###
# Add this to your argument parser: parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
patience = args.patience
smoothing_factor = args.smoothing_factor # e.g., 0.1
early_stop_counter = 0
unfreeze_epoch = 2000  # Adjustable: Wait until clusters are stable
unfreeze_iter = unfreeze_epoch * num_batches

smoothed_test_mse = float('inf') # Initialize to infinity for the first EMA calculation
best_smoothed_test_mse = float('inf') # Tracks the best smoothed MSE encountered
# Define a path for the best model checkpoint
best_ckpt_path = ckpt_path.replace('.ckpt', '_best.ckpt')

optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr)
utils.update_learning_rate(optimizer, decay_rate=1.0, lowest=args.lr) # Reset LR
### End of Initialization ###

for itr in range(1, num_batches * (args.niters + 1)):
    optimizer.zero_grad()
    utils.update_learning_rate(optimizer, decay_rate = 0.999, lowest = args.lr / 10)
    wait_until_kl_inc = 20
    # model.prior_means.data
    if itr // num_batches < wait_until_kl_inc:
        kl_coef = 0.
    else:
        kl_coef = (1-0.99** (itr // num_batches - wait_until_kl_inc))
    if itr == unfreeze_iter and args.use_gmm_v:
        initialize_gmm_with_kmeans(model, data_obj, device, n_batches=data_obj["n_test_batches"])
        model.latent_rotation.weight.requires_grad = True
        
        # 2. Add Parametrization (Orthogonality) NOW if you want strict rotation
        # (Optional but recommended if using the 'Parametrization' method)
        # torch.nn.utils.parametrizations.orthogonal(model.latent_rotation, "weight")
        # 3. Re-initialize Optimizer to include the new parameter
        optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr * 0.1) # Lower LR for fine-tuning
        
    # epoch = itr // num_batches
    # wait_until_kl_inc = 0
    # max_kl_beta = 20.0 # <--- Cap at 5.0 instead of 1.0
    
    # if epoch < wait_until_kl_inc:
    #     kl_coef = 0.
    # else:
    #     # Reaches 5.0 quickly
    #     kl_coef = min(max_kl_beta, (epoch - wait_until_kl_inc) / 1)
    # --- START OF BATCH FETCHING ---
    if args.use_film:
        batch_dict = utils.get_next_batch_film(data_obj["train_dataloader"])
        train_res = model.compute_film_losses(batch_dict, n_traj_samples=3, kl_coef=kl_coef, max_out = data_obj["max_out"])
    else:
        batch_dict = utils.get_next_batch(data_obj["train_dataloader"])
        train_res = model.compute_all_losses(batch_dict, n_traj_samples=3, kl_coef=kl_coef)
    # train_res = model.compute_all_losses(batch_dict, n_traj_samples = 3, kl_coef = kl_coef)
            
    train_res["loss"].backward()
    optimizer.step()

    n_iters_to_viz = 10
    if itr % (n_iters_to_viz * num_batches) == 0:
        with torch.no_grad():

            test_res = compute_loss_all_batches(model, 
                data_obj["test_dataloader"], args,
                n_batches = data_obj["n_test_batches"],
                experimentID = experimentID,
                device = device,
                n_traj_samples = 10, kl_coef = kl_coef, data_obj = data_obj)
            if hasattr(args, 'use_film') and args.use_film:
                # Custom logging for FiLM showing V1 and V2 losses explicitly
                
                message = 'Epoch {:04d} [FiLM Test] | Total Loss {:.6f} | V1 MSE (Context) {:.6f} | V2 MSE (Extrap) {:.6f} | KL {:.4f} | auc {:6f}'.format(
                    itr // num_batches, 
                    test_res["loss"].item() if hasattr(test_res["loss"], 'item') else test_res["loss"],
                    test_res["rec_loss_v1"].item() if hasattr(test_res["rec_loss_v1"], 'item') else test_res["rec_loss_v1"],
                    test_res["rec_loss_v2"].item() if hasattr(test_res["rec_loss_v2"], 'item') else test_res["rec_loss_v2"],
                    test_res["kl_loss"].item() if hasattr(test_res["kl_loss"], 'item') else test_res["kl_loss"],
                    test_res["rmse_auc"].item() if hasattr(test_res["rmse_auc"], 'item') else test_res["rmse_auc"]
                )
            else:
                # Original logging
                message = 'Epoch {:04d} [Test seq (cond on sampled tp)] | Loss {:.6f} | Likelihood {:.6f} | KL fp {:.4f} | FP STD {:.4f}|'.format(
                    itr // num_batches, 
                    test_res["loss"].item() if hasattr(test_res["loss"], 'item') else test_res["loss"], 
                    test_res["likelihood"].item() if hasattr(test_res["likelihood"], 'item') else test_res["likelihood"], 
                    test_res["kl_first_p"].item() if hasattr(test_res["kl_first_p"], 'item') else test_res["kl_first_p"], 
                    test_res["std_first_p"].item() if hasattr(test_res["std_first_p"], 'item') else test_res["std_first_p"]
                )
        
            logger.info("Experiment " + str(experimentID))
            logger.info(message)
            logger.info("KL coef: {}".format(kl_coef))
            logger.info("Train loss (one batch): {}".format(train_res["loss"].detach()))
            logger.info("Train CE loss (one batch): {}".format(train_res["ce_loss"].detach()))
            
            if "auc" in test_res:
                logger.info("Classification AUC (TEST): {:.4f}".format(test_res["auc"]))

            # --- MSE and Early Stopping Logic (Modified for Smoothing) ---
            if "mse" in test_res:
                current_raw_mse = test_res["mse"].item() # Get the raw current MSE
                logger.info("Test MSE (raw): {:.4f}".format(current_raw_mse))
                
                ### Calculate Exponential Moving Average of Test MSE ###
                if smoothed_test_mse == float('inf'): # For the very first evaluation
                    smoothed_test_mse = current_raw_mse
                else:
                    smoothed_test_mse = (current_raw_mse * smoothing_factor) + \
                                        (smoothed_test_mse * (1 - smoothing_factor))
                
                logger.info(f"Test MSE (smoothed with alpha={smoothing_factor}): {smoothed_test_mse:.4f}")

                ### EARLY STOPPING CHECK against Smoothed MSE ###
                if smoothed_test_mse < best_smoothed_test_mse:
                    best_smoothed_test_mse = smoothed_test_mse
                    early_stop_counter = 0
                    logger.info(f"New best smoothed test MSE: {best_smoothed_test_mse:.4f}. Saving best model to {best_ckpt_path}")
                    # Save the model state when a new best smoothed MSE is achieved
                    torch.save({
                        'args': args,
                        'state_dict': model.state_dict(),
                        'epoch': itr//num_batches,
                        'raw_test_mse_at_best': current_raw_mse, # Store raw MSE for context
                        'smoothed_test_mse': best_smoothed_test_mse
                    }, best_ckpt_path)
                else:
                    early_stop_counter += 1
                    logger.info(f"Smoothed Test MSE did not improve. Early stopping counter: {early_stop_counter}/{patience}")

                if early_stop_counter >= patience and itr > 2000:
                    logger.info("Early stopping triggered: Smoothed Test MSE has not improved for {} epochs.".format(patience))
                    break # Exit the main training loop
                ### END OF EARLY STOPPING CHECK ###

                for i in range(4):
                    logger.info("Test MSE group_{}: {:.4f}".format(i,test_res["mse_cond"][i]))

            if "accuracy" in train_res:
                logger.info("Classification accuracy (TRAIN): {:.4f}".format(train_res["accuracy"]))

            if "accuracy" in test_res:
                logger.info("Classification accuracy (TEST): {:.4f}".format(test_res["accuracy"]))

            if "pois_likelihood" in test_res:
                logger.info("Poisson likelihood: {}".format(test_res["pois_likelihood"]))

            if "ce_loss" in test_res:
                logger.info("CE loss: {}".format(test_res["ce_loss"]))

            # --- TensorBoard Logging ---
            writer.add_scalar('Loss/train_total_loss', train_res["loss"].item(), itr)
            if "mse" in train_res:
                writer.add_scalar('MSE/train', train_res["mse"].item(), itr)
                for i in range(4):
                    logger.info("train MSE group_{}: {:.4f}".format(i,train_res["mse_cond"][i]))
            if "ce_loss" in train_res and not torch.isnan(train_res["ce_loss"]):
                    writer.add_scalar('Loss/train_ce_loss', train_res["ce_loss"].item(), itr)

            writer.add_scalar('Loss/test_total_loss', test_res["loss"].item(), itr)
            writer.add_scalar('Likelihood/test', test_res["likelihood"].item(), itr)

            if "mse" in test_res:
                writer.add_scalar('MSE/test_raw', current_raw_mse, itr) # Log raw MSE
                # writer.add_scalar('MSE/test_smoothed', smoothed_test_mse, itr) # Log smoothed MSE
            if "rmse_auc" in test_res:
                writer.add_scalar('MSE/rmse_auc', test_res["rmse_auc"].item(), itr) # Log raw MSE
            if "ce_loss" in test_res and not torch.isnan(test_res["ce_loss"]):
                logger.info("CE loss (TEST): {}".format(test_res["ce_loss"]))
                writer.add_scalar('Loss/test_ce_loss', test_res["ce_loss"].item(), itr)
            writer.flush()
        
        # --- Save Latest Checkpoint ---
        torch.save({
            'args': args,
            'state_dict': model.state_dict(),
        }, ckpt_path)


        # Plotting
        if args.viz and itr % (10 * num_batches) == 0:
            with torch.no_grad():
                print("plotting....")
                plot_id = itr // num_batches // n_iters_to_viz
                
                if hasattr(args, 'use_film') and args.use_film:
                    test_dict = utils.get_next_batch_film(data_obj["test_dataloader"])
                    
                    viz.draw_all_plots_film(test_dict, model, 
                        plot_name = file_name + "_" + str(experimentID) + "_film_{:03d}".format(plot_id) + ".png",
                        experimentID = experimentID, save=True,
                        scaler=data_obj.get("max_out", None))
                else:
                    test_dict = utils.get_next_batch(data_obj["test_dataloader"])
                    if isinstance(model, LatentODE) and (args.dataset in ["periodic", "PK_Example", 'PK_Tacro', 'PK_MMF']):
                        viz.draw_all_plots_one_dim(test_dict, model, 
                            plot_name = file_name + "_" + str(experimentID) + "_{:03d}".format(plot_id) + ".png",
                            experimentID = experimentID, save=True)
                
                plt.pause(0.01)

torch.save({
    'args': args,
    'state_dict': model.state_dict(),
}, ckpt_path)

writer.close()
print("Training complete. TensorBoard writer closed.")
if best_smoothed_test_mse != float('inf'):
    print(f"Best model (based on smoothed MSE) saved to {best_ckpt_path} with smoothed Test MSE: {best_smoothed_test_mse:.4f}")
else:
    print("Training finished, but no 'best' model was saved as MSE was not tracked or training ended early.")
