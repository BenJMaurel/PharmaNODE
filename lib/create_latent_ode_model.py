###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Modified version of Yulia Rubanova
###########################

import os
import numpy as np

import torch
import torch.nn as nn
from torch.nn.functional import relu

from . import utils
from .latent_ode import LatentODE, LatentODEGMM, LatentODEGMM_V, LatentODEFlow
from .encoder_decoder import *
from .diffeq_solver import DiffeqSolver
from .ode_func import ODEFunc, ODEFunc_w_Poisson

from torch.distributions.normal import Normal


#####################################################################################################

def create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, device, 
	classif_per_tp = False, n_labels = 1):

	dim = args.latents
	if args.poisson:
		lambda_net = utils.create_net(dim, input_dim, 
			n_layers = 1, n_units = args.units, nonlinear = nn.Tanh)

		# ODE function produces the gradient for latent state and for poisson rate
		ode_func_net = utils.create_net(dim * 2, args.latents * 2, 
			n_layers = args.gen_layers, n_units = args.units, nonlinear = nn.Tanh)

		gen_ode_func = ODEFunc_w_Poisson(
			input_dim = input_dim, 
			latent_dim = args.latents * 2,
			ode_func_net = ode_func_net,
			lambda_net = lambda_net,
			device = device).to(device)
	else:
		dim = args.latents 
		ode_func_net = utils.create_net(dim, args.latents, 
			n_layers = args.gen_layers, n_units = args.units, nonlinear = nn.Tanh)

		gen_ode_func = ODEFunc(
			input_dim = input_dim, 
			latent_dim = args.latents, 
			ode_func_net = ode_func_net,
			device = device).to(device)

	z0_diffeq_solver = None
	n_rec_dims = args.rec_dims
	# enc_input_dim = (int(input_dim) + 3) * 2 # we add the static values then concatenate the mask
	enc_input_dim = int(input_dim) + 1 # we concatenate the dose and other (dynamic) covariates

	gen_data_dim = input_dim

	z0_dim = args.latents
	if args.poisson:
		z0_dim += args.latents # predict the initial poisson rate

	if args.z0_encoder == "odernn":
		ode_func_net = utils.create_net(n_rec_dims, n_rec_dims, 
			n_layers = args.rec_layers, n_units = args.units, nonlinear = nn.Tanh)

		rec_ode_func = ODEFunc(
			input_dim = enc_input_dim, 
			latent_dim = n_rec_dims,
			ode_func_net = ode_func_net,
			device = device).to(device)

		z0_diffeq_solver = DiffeqSolver(enc_input_dim, rec_ode_func, "euler", args.latents, 
			odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device)
		
		encoder_z0 = Encoder_z0_ODE_RNN(n_rec_dims, enc_input_dim, z0_diffeq_solver, 
			z0_dim = z0_dim, n_gru_units = args.gru_units, device = device).to(device)

	elif args.z0_encoder == "rnn":
		encoder_z0 = Encoder_z0_RNN(z0_dim, enc_input_dim,
			lstm_output_size = n_rec_dims, device = device).to(device)
	else:
		raise Exception("Unknown encoder for Latent ODE model: " + args.z0_encoder)

	decoder = Decoder(args.latents, gen_data_dim).to(device)

	diffeq_solver = DiffeqSolver(gen_data_dim, gen_ode_func, 'dopri5', args.latents, 
		odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device)

	use_gmm = getattr(args, 'use_gmm', False) 
	use_gmm_v = getattr(args, 'use_gmm_v', False) 
	use_flow = getattr(args, 'use_flow', False) 
	if use_gmm:
		n_components = getattr(args, 'n_components', 5)
		print(f"Initializing LatentODEGMM with {n_components} clusters.")

		model = LatentODEGMM(
		    input_dim = gen_data_dim, 
		    latent_dim = args.latents, 
		    encoder_z0 = encoder_z0, 
		    decoder = decoder, 
		    diffeq_solver = diffeq_solver, 
		    z0_prior = z0_prior, 
		    device = device,
		    n_components = n_components,  # Pass the number of clusters
		    obsrv_std = obsrv_std,
		    use_poisson_proc = args.poisson, 
		    use_binary_classif = args.classif,
		    linear_classifier = args.linear_classif,
		    classif_per_tp = classif_per_tp,
		    n_labels = n_labels,
		    train_classif_w_reconstr = (args.dataset == "physionet")
		    ).to(device)
		
	elif use_gmm_v:
		n_components = getattr(args, 'n_components', 5)
		print(f"Initializing LatentODEGMM with {n_components} clusters.")

		model = LatentODEGMM_V(
		    input_dim = gen_data_dim, 
		    latent_dim = args.latents, 
		    encoder_z0 = encoder_z0, 
		    decoder = decoder, 
		    diffeq_solver = diffeq_solver, 
		    z0_prior = z0_prior, 
		    device = device,
		    n_components = n_components,  # Pass the number of clusters
		    obsrv_std = obsrv_std,
		    use_poisson_proc = args.poisson, 
		    use_binary_classif = args.classif,
		    linear_classifier = args.linear_classif,
		    classif_per_tp = classif_per_tp,
		    n_labels = n_labels,
		    train_classif_w_reconstr = (args.dataset == "physionet")
		    ).to(device)
	elif use_flow:
		print(f"Initializing LatentODEFlow")

		model = LatentODEFlow(
		    input_dim = gen_data_dim, 
		    latent_dim = args.latents, 
		    encoder_z0 = encoder_z0, 
		    decoder = decoder, 
		    diffeq_solver = diffeq_solver, 
		    z0_prior = z0_prior, 
		    device = device,
		    obsrv_std = obsrv_std,
		    use_poisson_proc = args.poisson, 
		    use_binary_classif = args.classif,
		    linear_classifier = args.linear_classif,
		    classif_per_tp = classif_per_tp,
		    n_labels = n_labels,
		    train_classif_w_reconstr = (args.dataset == "physionet")
		    ).to(device)
	else:
        # Standard Model
		model = LatentODE(
		    input_dim = gen_data_dim, 
		    latent_dim = args.latents, 
		    encoder_z0 = encoder_z0, 
		    decoder = decoder, 
		    diffeq_solver = diffeq_solver, 
		    z0_prior = z0_prior, 
		    device = device,
		    obsrv_std = obsrv_std,
		    use_poisson_proc = args.poisson, 
		    use_binary_classif = args.classif,
		    linear_classifier = args.linear_classif,
		    classif_per_tp = classif_per_tp,
			film = args.use_film,
			film_time = args.use_time,
		    n_labels = n_labels,
		    train_classif_w_reconstr = (args.dataset == "physionet")
		    ).to(device)

	return model

