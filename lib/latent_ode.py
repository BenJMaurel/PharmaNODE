###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import numpy as np
import sklearn as sk
import torch
import torch.nn as nn
from torch.nn.functional import relu, softmax

from . import utils
from .utils import get_device
from .encoder_decoder import *
from .likelihood_eval import *
from .base_models import VAE_Baseline, VAE_GMM, VAE_GMM_V, VAE_Flow
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence, Independent, Categorical

class LatentODE(VAE_Baseline):
	def __init__(self, input_dim, latent_dim, encoder_z0, decoder, diffeq_solver, 
		z0_prior, device, obsrv_std = None, 
		use_binary_classif = False, use_poisson_proc = False,
		linear_classifier = False,
		classif_per_tp = False,
		n_labels = 1,
		train_classif_w_reconstr = False,
		dose_encoding_net = None):

		super(LatentODE, self).__init__(
			input_dim = input_dim, latent_dim = latent_dim, 
			z0_prior = z0_prior, 
			device = device, obsrv_std = obsrv_std, 
			use_binary_classif = use_binary_classif,
			classif_per_tp = classif_per_tp, 
			linear_classifier = linear_classifier,
			use_poisson_proc = use_poisson_proc,
			n_labels = n_labels,
			train_classif_w_reconstr = train_classif_w_reconstr)

		self.encoder_z0 = encoder_z0
		self.diffeq_solver = diffeq_solver
		self.decoder = decoder
		self.use_poisson_proc = use_poisson_proc
		if dose_encoding_net:
			self.dose_encoding_net = dose_encoding_net

	def get_reconstruction(self, time_steps_to_predict, truth, truth_time_steps, 
		mask = None, n_traj_samples = 1, run_backwards = True, mode = None, dose = None, static = None):

		if isinstance(self.encoder_z0, Encoder_z0_ODE_RNN) or \
			isinstance(self.encoder_z0, Encoder_z0_RNN):
			truth_w_mask = truth
			if mask is not None:
				truth_w_mask = torch.cat((truth, mask), -1)
			elif dose is not None:
				try:
					truth_w_mask = torch.cat((truth, dose), -1)
				except:
					pass
			
			first_point_mu, first_point_std = self.encoder_z0(
				truth_w_mask, truth_time_steps, static = static, run_backwards = run_backwards)
			means_z0 = first_point_mu.repeat(n_traj_samples, 1, 1)
			sigma_z0 = first_point_std.repeat(n_traj_samples, 1, 1)
			first_point_enc = utils.sample_standard_gaussian(means_z0, sigma_z0)

		else:
			raise Exception("Unknown encoder type {}".format(type(self.encoder_z0).__name__))
		
		first_point_std = first_point_std.abs()
		assert(torch.sum(first_point_std < 0) == 0.)

		if self.use_poisson_proc:
			n_traj_samples, n_traj, n_dims = first_point_enc.size()
			# append a vector of zeros to compute the integral of lambda
			zeros = torch.zeros([n_traj_samples, n_traj,self.input_dim]).to(get_device(truth))
			first_point_enc_aug = torch.cat((first_point_enc, zeros), -1)
			means_z0_aug = torch.cat((means_z0, zeros), -1)
		else:
			first_point_enc_aug = first_point_enc
			means_z0_aug = means_z0
			
		assert(not torch.isnan(time_steps_to_predict).any())
		assert(not torch.isnan(first_point_enc).any())
		assert(not torch.isnan(first_point_enc_aug).any())

		# Shape of sol_y [n_traj_samples, n_samples, n_timepoints, n_latents]
		sol_y = self.diffeq_solver(first_point_enc_aug, time_steps_to_predict)

		if self.use_poisson_proc:
			sol_y, log_lambda_y, int_lambda, _ = self.diffeq_solver.ode_func.extract_poisson_rate(sol_y)

			assert(torch.sum(int_lambda[:,:,0,:]) == 0.)
			assert(torch.sum(int_lambda[0,0,-1,:] <= 0) == 0.)

		pred_x = self.decoder(sol_y)

		all_extra_info = {
			"first_point": (first_point_mu, first_point_std, first_point_enc),
			"latent_traj": sol_y.detach()
		}

		if self.use_poisson_proc:
			# intergral of lambda from the last step of ODE Solver
			all_extra_info["int_lambda"] = int_lambda[:,:,-1,:]
			all_extra_info["log_lambda_y"] = log_lambda_y

		if self.use_binary_classif:
			if self.classif_per_tp:
				all_extra_info["label_predictions"] = self.classifier(sol_y)
			else:
				all_extra_info["label_predictions"] = self.classifier(first_point_enc).squeeze(-1)

		return pred_x, all_extra_info


	def sample_traj_from_prior(self, time_steps_to_predict, n_traj_samples = 1):
		# Sample z0 from prior
		starting_point_enc = self.z0_prior.sample([n_traj_samples, 1, self.latent_dim]).squeeze(-1)

		starting_point_enc_aug = starting_point_enc
		if self.use_poisson_proc:
			n_traj_samples, n_traj, n_dims = starting_point_enc.size()
			# append a vector of zeros to compute the integral of lambda
			zeros = torch.zeros(n_traj_samples, n_traj,self.input_dim).to(self.device)
			starting_point_enc_aug = torch.cat((starting_point_enc, zeros), -1)

		sol_y = self.diffeq_solver.sample_traj_from_prior(starting_point_enc_aug, time_steps_to_predict, 
			n_traj_samples = 3)

		if self.use_poisson_proc:
			sol_y, log_lambda_y, int_lambda, _ = self.diffeq_solver.ode_func.extract_poisson_rate(sol_y)
		
		return self.decoder(sol_y)


class LatentODEGMM(VAE_GMM):
    def __init__(self, input_dim, latent_dim, encoder_z0, decoder, diffeq_solver, 
        z0_prior, device, n_components = 5, obsrv_std = None, 
        use_binary_classif = False, use_poisson_proc = False,
        linear_classifier = False,
        classif_per_tp = False,
        n_labels = 1,
        train_classif_w_reconstr = False,
        dose_encoding_net = None):

        # Initialize VAE_GMM (which initializes VAE_Baseline)
        # We pass n_components here
        super(LatentODEGMM, self).__init__(
            input_dim = input_dim, latent_dim = latent_dim, 
            z0_prior = z0_prior, 
            device = device, obsrv_std = obsrv_std, 
            n_components = n_components,
            use_binary_classif = use_binary_classif,
            classif_per_tp = classif_per_tp, 
            linear_classifier = linear_classifier,
            use_poisson_proc = use_poisson_proc,
            n_labels = n_labels,
            train_classif_w_reconstr = train_classif_w_reconstr)

        self.encoder_z0 = encoder_z0
        self.diffeq_solver = diffeq_solver
        self.decoder = decoder
        self.use_poisson_proc = use_poisson_proc
        if dose_encoding_net:
            self.dose_encoding_net = dose_encoding_net

    def get_reconstruction(self, time_steps_to_predict, truth, truth_time_steps, 
        mask = None, n_traj_samples = 1, run_backwards = True, mode = None, dose = None, static = None):
        
        # This method is largely identical to LatentODE, as the encoder mechanism 
        # (producing q(z|x)) is structurally the same. The GMM logic is applied
        # during the loss calculation (in VAE_GMM.compute_all_losses) which 
        # calls this method.

        if isinstance(self.encoder_z0, Encoder_z0_ODE_RNN) or \
            isinstance(self.encoder_z0, Encoder_z0_RNN):
            truth_w_mask = truth
            if mask is not None:
                truth_w_mask = torch.cat((truth, mask), -1)
            elif dose is not None:
                try:
                    truth_w_mask = torch.cat((truth, dose), -1)
                except:
                    pass
            
            # 1. Run Encoder
            first_point_mu, first_point_std = self.encoder_z0(
                truth_w_mask, truth_time_steps, static = static, run_backwards = run_backwards)
            
            # 2. Sample z0 from the approximate posterior q(z|x) (Standard Normal Reparam)
            means_z0 = first_point_mu.repeat(n_traj_samples, 1, 1)
            sigma_z0 = first_point_std.repeat(n_traj_samples, 1, 1)
            first_point_enc = utils.sample_standard_gaussian(means_z0, sigma_z0)

        else:
            raise Exception("Unknown encoder type {}".format(type(self.encoder_z0).__name__))
        
        first_point_std = first_point_std.abs()
        assert(torch.sum(first_point_std < 0) == 0.)

        # 3. Handle Poisson Augmentation
        if self.use_poisson_proc:
            n_traj_samples, n_traj, n_dims = first_point_enc.size()
            zeros = torch.zeros([n_traj_samples, n_traj,self.input_dim]).to(get_device(truth))
            first_point_enc_aug = torch.cat((first_point_enc, zeros), -1)
        else:
            first_point_enc_aug = first_point_enc
            
        assert(not torch.isnan(time_steps_to_predict).any())
        assert(not torch.isnan(first_point_enc).any())

        # 4. Run ODE Solver
        sol_y = self.diffeq_solver(first_point_enc_aug, time_steps_to_predict)

        if self.use_poisson_proc:
            sol_y, log_lambda_y, int_lambda, _ = self.diffeq_solver.ode_func.extract_poisson_rate(sol_y)

            assert(torch.sum(int_lambda[:,:,0,:]) == 0.)
            assert(torch.sum(int_lambda[0,0,-1,:] <= 0) == 0.)

        # 5. Decode
        pred_x = self.decoder(sol_y)

        all_extra_info = {
            "first_point": (first_point_mu, first_point_std, first_point_enc),
            "latent_traj": sol_y.detach()
        }

        if self.use_poisson_proc:
            all_extra_info["int_lambda"] = int_lambda[:,:,-1,:]
            all_extra_info["log_lambda_y"] = log_lambda_y

        if self.use_binary_classif:
            if self.classif_per_tp:
                all_extra_info["label_predictions"] = self.classifier(sol_y)
            else:
                all_extra_info["label_predictions"] = self.classifier(first_point_enc).squeeze(-1)

        return pred_x, all_extra_info


    def sample_traj_from_prior(self, time_steps_to_predict, n_traj_samples = 1):
        """
        Modified for GMM: Samples z0 from the learned Gaussian Mixture Model prior.
        """
        
        # 1. Sample Cluster Assignments (c ~ Categorical(pi))
        # weights logits are [n_components]
        probs = softmax(self.prior_weights_logits, dim=0)
        dist_c = Categorical(probs)
        
        # Sample indices: [n_traj_samples]
        cluster_indices = dist_c.sample((n_traj_samples,))
        
        # 2. Gather parameters for the chosen clusters
        # prior_means: [n_components, latent_dim] -> [n_traj_samples, latent_dim]
        # prior_logvars: [n_components, latent_dim] -> [n_traj_samples, latent_dim]
        means = self.prior_means[cluster_indices]
        logvars = self.prior_logvars[cluster_indices]
        stds = torch.exp(0.5 * logvars)
        
        # 3. Sample z0 (z ~ N(mu_c, std_c))
        # Shape: [n_traj_samples, latent_dim]
        eps = torch.randn_like(stds)
        starting_point_enc = means + stds * eps
        
        # Reshape for ODE solver: [n_traj_samples, 1, latent_dim]
        # The '1' represents the batch dimension (n_traj), here 1 because we are just sampling general trajectories
        starting_point_enc = starting_point_enc.unsqueeze(1)

        starting_point_enc_aug = starting_point_enc
        if self.use_poisson_proc:
            n_samples, n_traj, n_dims = starting_point_enc.size()
            zeros = torch.zeros(n_samples, n_traj, self.input_dim).to(self.device)
            starting_point_enc_aug = torch.cat((starting_point_enc, zeros), -1)

        # 4. Solve ODE
        sol_y = self.diffeq_solver.sample_traj_from_prior(
            starting_point_enc_aug, 
            time_steps_to_predict, 
            n_traj_samples = n_traj_samples 
        )

        if self.use_poisson_proc:
            sol_y, log_lambda_y, int_lambda, _ = self.diffeq_solver.ode_func.extract_poisson_rate(sol_y)
        
        return self.decoder(sol_y)

class LatentODEGMM_V(VAE_GMM_V):
    def __init__(self, input_dim, latent_dim, encoder_z0, decoder, diffeq_solver, 
        z0_prior, device, n_components = 4, obsrv_std = None, 
        use_binary_classif = False, use_poisson_proc = False,
        linear_classifier = False,
        classif_per_tp = False,
        n_labels = 1,
        train_classif_w_reconstr = False,
        dose_encoding_net = None):

        super(LatentODEGMM_V, self).__init__(
            input_dim = input_dim, latent_dim = latent_dim, 
            z0_prior = z0_prior, 
            device = device, obsrv_std = obsrv_std, 
            n_components = n_components,
            use_binary_classif = use_binary_classif,
            classif_per_tp = classif_per_tp, 
            linear_classifier = linear_classifier,
            use_poisson_proc = use_poisson_proc,
            n_labels = n_labels,
            train_classif_w_reconstr = train_classif_w_reconstr)

        self.encoder_z0 = encoder_z0
        self.diffeq_solver = diffeq_solver
        self.decoder = decoder
        self.use_poisson_proc = use_poisson_proc
        if dose_encoding_net:
            self.dose_encoding_net = dose_encoding_net

    def get_reconstruction(self, time_steps_to_predict, truth, truth_time_steps, 
        mask = None, n_traj_samples = 1, run_backwards = True, mode = None, dose = None, static = None):
        
        # --- NO CHANGES NEEDED ---
        # Encoder -> Z -> ODE -> Decoder
        # The rotation is implicit in the loss function, not the reconstruction path.
        
        if isinstance(self.encoder_z0, Encoder_z0_ODE_RNN) or \
            isinstance(self.encoder_z0, Encoder_z0_RNN):
            truth_w_mask = truth
            if mask is not None:
                truth_w_mask = torch.cat((truth, mask), -1)
            elif dose is not None:
                try:
                    truth_w_mask = torch.cat((truth, dose), -1)
                except:
                    pass
            
            first_point_mu, first_point_std = self.encoder_z0(
                truth_w_mask, truth_time_steps, static = static, run_backwards = run_backwards)
            
            means_z0 = first_point_mu.repeat(n_traj_samples, 1, 1)
            sigma_z0 = first_point_std.repeat(n_traj_samples, 1, 1)
            first_point_enc = utils.sample_standard_gaussian(means_z0, sigma_z0)

        else:
            raise Exception("Unknown encoder type {}".format(type(self.encoder_z0).__name__))
        
        first_point_std = first_point_std.abs()
        assert(torch.sum(first_point_std < 0) == 0.)

        if self.use_poisson_proc:
            n_traj_samples, n_traj, n_dims = first_point_enc.size()
            zeros = torch.zeros([n_traj_samples, n_traj,self.input_dim]).to(utils.get_device(truth))
            first_point_enc_aug = torch.cat((first_point_enc, zeros), -1)
        else:
            first_point_enc_aug = first_point_enc

        sol_y = self.diffeq_solver(first_point_enc_aug, time_steps_to_predict)

        if self.use_poisson_proc:
            sol_y, log_lambda_y, int_lambda, _ = self.diffeq_solver.ode_func.extract_poisson_rate(sol_y)

        pred_x = self.decoder(sol_y)

        all_extra_info = {
            "first_point": (first_point_mu, first_point_std, first_point_enc),
            "latent_traj": sol_y.detach()
        }

        if self.use_poisson_proc:
            all_extra_info["int_lambda"] = int_lambda[:,:,-1,:]
            all_extra_info["log_lambda_y"] = log_lambda_y

        if self.use_binary_classif:
            if self.classif_per_tp:
                all_extra_info["label_predictions"] = self.classifier(sol_y)
            else:
                all_extra_info["label_predictions"] = self.classifier(first_point_enc).squeeze(-1)

        return pred_x, all_extra_info

    def sample_traj_from_prior(self, time_steps_to_predict, n_traj_samples = 1):
        """
        UPDATED: Samples u from Diagonal GMM, then inverse rotates to get z.
        """
        
        # 1. Sample Cluster Assignments
        probs = F.softmax(self.prior_weights_logits, dim=0)
        dist_c = Categorical(probs)
        cluster_indices = dist_c.sample((n_traj_samples,))
        
        # 2. Gather Diagonal Parameters for u
        means = self.prior_means[cluster_indices]
        logvars = self.prior_logvars[cluster_indices]
        stds = torch.exp(0.5 * logvars)
        
        # 3. Sample u ~ N(mu_c, std_c)
        eps = torch.randn_like(stds)
        u_samples = means + stds * eps # [n_traj_samples, latent_dim]
        
        # 4. Inverse Rotate to get z (z = R^-1 * u)
        W = self.latent_rotation.weight # [Out, In] -> [Dim, Dim]
        
        W_inv = torch.inverse(W)
        starting_point_enc = torch.matmul(u_samples, W_inv.t())
        
        # Reshape for ODE solver: [n_traj_samples, 1, latent_dim]
        starting_point_enc = starting_point_enc.unsqueeze(1)

        starting_point_enc_aug = starting_point_enc
        if self.use_poisson_proc:
            n_samples, n_traj, n_dims = starting_point_enc.size()
            zeros = torch.zeros(n_samples, n_traj, self.input_dim).to(self.device)
            starting_point_enc_aug = torch.cat((starting_point_enc, zeros), -1)

        # 5. Solve ODE
        sol_y = self.diffeq_solver.sample_traj_from_prior(
            starting_point_enc_aug, 
            time_steps_to_predict, 
            n_traj_samples = n_traj_samples 
        )

        if self.use_poisson_proc:
            sol_y, log_lambda_y, int_lambda, _ = self.diffeq_solver.ode_func.extract_poisson_rate(sol_y)
        
        return self.decoder(sol_y)
    
class LatentODEGMM_V2(VAE_GMM_V):
    def __init__(self, input_dim, latent_dim, encoder_z0, decoder, diffeq_solver, 
        z0_prior, device, n_components = 5, obsrv_std = None, 
        use_binary_classif = False, use_poisson_proc = False,
        linear_classifier = False,
        classif_per_tp = False,
        n_labels = 1,
        train_classif_w_reconstr = False,
        dose_encoding_net = None):

        # Initialize VAE_GMM (which initializes VAE_Baseline)
        super(LatentODEGMM_V, self).__init__(
            input_dim = input_dim, latent_dim = latent_dim, 
            z0_prior = z0_prior, 
            device = device, obsrv_std = obsrv_std, 
            n_components = n_components,
            use_binary_classif = use_binary_classif,
            classif_per_tp = classif_per_tp, 
            linear_classifier = linear_classifier,
            use_poisson_proc = use_poisson_proc,
            n_labels = n_labels,
            train_classif_w_reconstr = train_classif_w_reconstr)

        self.encoder_z0 = encoder_z0
        self.diffeq_solver = diffeq_solver
        self.decoder = decoder
        self.use_poisson_proc = use_poisson_proc
        if dose_encoding_net:
            self.dose_encoding_net = dose_encoding_net

    def get_reconstruction(self, time_steps_to_predict, truth, truth_time_steps, 
        mask = None, n_traj_samples = 1, run_backwards = True, mode = None, dose = None, static = None):
        
        # --- NO CHANGES NEEDED HERE ---
        # The encoder still produces a Diagonal Posterior q(z|x).
        # This is standard practice even if the Prior p(z) is Full Covariance.
        
        if isinstance(self.encoder_z0, Encoder_z0_ODE_RNN) or \
            isinstance(self.encoder_z0, Encoder_z0_RNN):
            truth_w_mask = truth
            if mask is not None:
                truth_w_mask = torch.cat((truth, mask), -1)
            elif dose is not None:
                try:
                    truth_w_mask = torch.cat((truth, dose), -1)
                except:
                    pass 
            
            # 1. Run Encoder
            first_point_mu, first_point_std = self.encoder_z0(
                truth_w_mask, truth_time_steps, static = static, run_backwards = run_backwards)
            
            # 2. Sample z0 from the approximate posterior q(z|x)
            means_z0 = first_point_mu.repeat(n_traj_samples, 1, 1)
            sigma_z0 = first_point_std.repeat(n_traj_samples, 1, 1)
            first_point_enc = utils.sample_standard_gaussian(means_z0, sigma_z0)

        else:
            raise Exception("Unknown encoder type {}".format(type(self.encoder_z0).__name__))
        
        first_point_std = first_point_std.abs()
        assert(torch.sum(first_point_std < 0) == 0.)

        # 3. Handle Poisson Augmentation
        if self.use_poisson_proc:
            n_traj_samples, n_traj, n_dims = first_point_enc.size()
            zeros = torch.zeros([n_traj_samples, n_traj,self.input_dim]).to(utils.get_device(truth))
            first_point_enc_aug = torch.cat((first_point_enc, zeros), -1)
        else:
            first_point_enc_aug = first_point_enc
            
        assert(not torch.isnan(time_steps_to_predict).any())
        assert(not torch.isnan(first_point_enc).any())

        # 4. Run ODE Solver
        sol_y = self.diffeq_solver(first_point_enc_aug, time_steps_to_predict)

        if self.use_poisson_proc:
            sol_y, log_lambda_y, int_lambda, _ = self.diffeq_solver.ode_func.extract_poisson_rate(sol_y)

            assert(torch.sum(int_lambda[:,:,0,:]) == 0.)
            assert(torch.sum(int_lambda[0,0,-1,:] <= 0) == 0.)

        # 5. Decode
        pred_x = self.decoder(sol_y)

        all_extra_info = {
            "first_point": (first_point_mu, first_point_std, first_point_enc),
            "latent_traj": sol_y.detach()
        }

        if self.use_poisson_proc:
            all_extra_info["int_lambda"] = int_lambda[:,:,-1,:]
            all_extra_info["log_lambda_y"] = log_lambda_y

        if self.use_binary_classif:
            if self.classif_per_tp:
                all_extra_info["label_predictions"] = self.classifier(sol_y)
            else:
                all_extra_info["label_predictions"] = self.classifier(first_point_enc).squeeze(-1)

        return pred_x, all_extra_info


    def sample_traj_from_prior(self, time_steps_to_predict, n_traj_samples = 1):
            """
            UPDATED: Samples z0 from the STABILIZED Full Covariance GMM Prior.
            """
            
            # 1. Sample Cluster Assignments (c ~ Categorical(pi))
            probs = F.softmax(self.prior_weights_logits, dim=0)
            dist_c = Categorical(probs)
            
            # Sample indices: [n_traj_samples]
            cluster_indices = dist_c.sample((n_traj_samples,))
            
            # 2. Reconstruct Full Cholesky Matrices for ALL components
            # MUST MATCH get_gmm_log_density EXACTLY
            L = torch.tril(self.prior_cov_tril)
            diag_mask = torch.eye(self.latent_dim, device=self.device).unsqueeze(0)
            
            # --- STABILIZATION FIX START ---
            # A. Clamp raw parameter to prevent explosion
            L_diag_raw = (L * diag_mask).clamp(min=-5, max=3)
            
            # B. Add Epsilon Jitter
            epsilon = 1e-4
            L = L * (1 - diag_mask) + (torch.exp(L_diag_raw) + epsilon) * diag_mask
            # --- STABILIZATION FIX END ---

            # 3. Gather Params for the chosen clusters
            # Select the specific Means and Covariance Matrices based on the sampled indices
            # means: [n_traj_samples, latent_dim]
            means = self.prior_means[cluster_indices]
            # scale_tril: [n_traj_samples, latent_dim, latent_dim]
            scale_tril = L[cluster_indices]
            
            # 4. Sample z0 using Multivariate Normal
            # z ~ N(mu_c, Sigma_c)
            mvn = MultivariateNormal(loc=means, scale_tril=scale_tril)
            starting_point_enc = mvn.rsample() # Shape: [n_traj_samples, latent_dim]
            
            # Reshape for ODE solver: [n_traj_samples, 1, latent_dim]
            starting_point_enc = starting_point_enc.unsqueeze(1)

            starting_point_enc_aug = starting_point_enc
            if self.use_poisson_proc:
                n_samples, n_traj, n_dims = starting_point_enc.size()
                zeros = torch.zeros(n_samples, n_traj, self.input_dim).to(self.device)
                starting_point_enc_aug = torch.cat((starting_point_enc, zeros), -1)

            # 5. Solve ODE
            sol_y = self.diffeq_solver.sample_traj_from_prior(
                starting_point_enc_aug, 
                time_steps_to_predict, 
                n_traj_samples = n_traj_samples 
            )

            if self.use_poisson_proc:
                sol_y, log_lambda_y, int_lambda, _ = self.diffeq_solver.ode_func.extract_poisson_rate(sol_y)
            
            return self.decoder(sol_y)
    
class LatentODEFlow(VAE_Flow):
    def __init__(self, input_dim, latent_dim, encoder_z0, decoder, diffeq_solver, 
        z0_prior, device, 
        num_flow_layers = 4, # New flow param
        obsrv_std = None, 
        use_binary_classif = False, use_poisson_proc = False,
        linear_classifier = False,
        classif_per_tp = False,
        n_labels = 1,
        train_classif_w_reconstr = False,
        dose_encoding_net = None):

        # Initialize VAE_Flow
        super(LatentODEFlow, self).__init__(
            input_dim = input_dim, latent_dim = latent_dim, 
            z0_prior = z0_prior, 
            device = device, obsrv_std = obsrv_std, 
            num_flow_layers = num_flow_layers, # Pass to parent
            use_binary_classif = use_binary_classif,
            classif_per_tp = classif_per_tp, 
            linear_classifier = linear_classifier,
            use_poisson_proc = use_poisson_proc,
            n_labels = n_labels,
            train_classif_w_reconstr = train_classif_w_reconstr)

        self.encoder_z0 = encoder_z0
        self.diffeq_solver = diffeq_solver
        self.decoder = decoder
        self.use_poisson_proc = use_poisson_proc
        if dose_encoding_net:
            self.dose_encoding_net = dose_encoding_net

    def get_reconstruction(self, time_steps_to_predict, truth, truth_time_steps, 
        mask = None, n_traj_samples = 1, run_backwards = True, mode = None, dose = None, static = None):
        
        # Standard reconstruction logic (same as original Latent ODE)
        if isinstance(self.encoder_z0, Encoder_z0_ODE_RNN) or \
            isinstance(self.encoder_z0, Encoder_z0_RNN):
            truth_w_mask = truth
            if mask is not None:
                truth_w_mask = torch.cat((truth, mask), -1)
            elif dose is not None:
                try:
                    truth_w_mask = torch.cat((truth, dose), -1)
                except:
                    pass 
            
            # 1. Run Encoder
            first_point_mu, first_point_std = self.encoder_z0(
                truth_w_mask, truth_time_steps, static = static, run_backwards = run_backwards)
            
            # 2. Sample z0 from q(z|x)
            means_z0 = first_point_mu.repeat(n_traj_samples, 1, 1)
            sigma_z0 = first_point_std.repeat(n_traj_samples, 1, 1)
            first_point_enc = utils.sample_standard_gaussian(means_z0, sigma_z0)

        else:
            raise Exception("Unknown encoder type {}".format(type(self.encoder_z0).__name__))
        
        first_point_std = first_point_std.abs()
        
        # 3. Handle Poisson Augmentation
        if self.use_poisson_proc:
            n_traj_samples, n_traj, n_dims = first_point_enc.size()
            zeros = torch.zeros([n_traj_samples, n_traj,self.input_dim]).to(self.device)
            first_point_enc_aug = torch.cat((first_point_enc, zeros), -1)
        else:
            first_point_enc_aug = first_point_enc
            
        # 4. Run ODE Solver
        sol_y = self.diffeq_solver(first_point_enc_aug, time_steps_to_predict)

        if self.use_poisson_proc:
            sol_y, log_lambda_y, int_lambda, _ = self.diffeq_solver.ode_func.extract_poisson_rate(sol_y)

        # 5. Decode
        pred_x = self.decoder(sol_y)

        all_extra_info = {
            "first_point": (first_point_mu, first_point_std, first_point_enc),
            "latent_traj": sol_y.detach()
        }

        if self.use_poisson_proc:
            all_extra_info["int_lambda"] = int_lambda[:,:,-1,:]
            all_extra_info["log_lambda_y"] = log_lambda_y

        if self.use_binary_classif:
            if self.classif_per_tp:
                all_extra_info["label_predictions"] = self.classifier(sol_y)
            else:
                all_extra_info["label_predictions"] = self.classifier(first_point_enc).squeeze(-1)

        return pred_x, all_extra_info

    def sample_traj_from_prior(self, time_steps_to_predict, n_traj_samples = 1):
        """
        Samples z0 from the Learned Flow Prior.
        """
        
        # 1. Sample from Base Distribution u ~ N(0, I)
        # shape: [n_traj_samples, latent_dim]
        u = torch.randn(n_traj_samples, self.latent_dim).to(self.device)
        
        # 2. Transform u -> z via Flow
        # z = Flow(u)
        z0, _ = self.flow(u)
        
        # Reshape for ODE solver: [n_traj_samples, 1, latent_dim]
        starting_point_enc = z0.unsqueeze(1)

        starting_point_enc_aug = starting_point_enc
        if self.use_poisson_proc:
            n_samples, n_traj, n_dims = starting_point_enc.size()
            zeros = torch.zeros(n_samples, n_traj, self.input_dim).to(self.device)
            starting_point_enc_aug = torch.cat((starting_point_enc, zeros), -1)

        # 3. Solve ODE
        sol_y = self.diffeq_solver.sample_traj_from_prior(
            starting_point_enc_aug, 
            time_steps_to_predict, 
            n_traj_samples = n_traj_samples 
        )

        if self.use_poisson_proc:
            sol_y, log_lambda_y, int_lambda, _ = self.diffeq_solver.ode_func.extract_poisson_rate(sol_y)
        
        return self.decoder(sol_y)