###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import numpy as np
import sklearn as sk
#import gc
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
from scipy.special import inv_boxcox
class LatentODE(VAE_Baseline):
	def __init__(self, input_dim, latent_dim, encoder_z0, decoder, diffeq_solver, 
		z0_prior, device, obsrv_std = None, 
		use_binary_classif = False, use_poisson_proc = False,
		linear_classifier = False,
		classif_per_tp = False,
		n_labels = 1,
		train_classif_w_reconstr = False,
		dose_encoding_net = None, film = False, film_time = False, film_occ = 0):

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
		self.film_time = film_time
		if dose_encoding_net:
			self.dose_encoding_net = dose_encoding_net
		if film:
			# n_film_occ = 3 adds [delta_days, Ht1, Ht2] to the context, giving
			# c = [d1, d2, ddays, Ht1, Ht2, z0]. Default 0 reproduces the published
			# dims exactly, so existing checkpoints still load.
			self.n_film_occ = int(film_occ)
			film_input_dim = 2 + self.n_film_occ + self.latent_dim
			film_hidden = 32
            
			self.film_gamma = nn.Sequential(
                nn.Linear(film_input_dim, film_hidden),
                nn.ReLU(),
                nn.Linear(film_hidden, self.latent_dim)
            )
			self.film_beta = nn.Sequential(
                nn.Linear(film_input_dim, film_hidden),
                nn.ReLU(),
                nn.Linear(film_hidden, self.latent_dim)
            )
            # Initialize final layer of gamma to output 1s (Identity scaling initially)
			nn.init.constant_(self.film_gamma[-1].weight, 0.0)
			nn.init.constant_(self.film_gamma[-1].bias, 1.0)
			if film_time:
				self.film_gamma_t = nn.Sequential(
                nn.Linear(film_input_dim, film_hidden),
                nn.ReLU(),
                nn.Linear(film_hidden, self.latent_dim)
            )
				self.film_beta_t = nn.Sequential(
                nn.Linear(film_input_dim, film_hidden),
                nn.ReLU(),
                nn.Linear(film_hidden, self.latent_dim)
            )
            # Initialize final layer of gamma to output 1s (Identity scaling initially)
				nn.init.constant_(self.film_gamma_t[-1].weight, 0.0)
				nn.init.constant_(self.film_gamma_t[-1].bias, 1.0)

	def _encoder_input(self, data, dose, static):
		"""Build the encoder input [y, dose] and the static vector.

		encoder_dose_zero blanks BOTH places the dose reaches the encoder: the
		concatenated dose channel and static[:,0] (which read_tacro sets to the
		dose). Without this, z0 is entangled with d1 by construction and no
		projection in latent space can fully remove it.
		"""
		seq = data.size(1)
		dose_ch = dose.view(-1, 1, 1).expand(-1, seq, 1)
		if getattr(self, 'encoder_dose_zero', False):
			dose_ch = torch.zeros_like(dose_ch)
			if static is not None:
				static = static.clone(); static[:, 0] = 0.0
		if getattr(self, 'blank_formulation', False) and static is not None:
			# static[:,1] is the Prograf/Advagraf indicator. When a model is trained on
			# one formulation it is constant, so its weights are unconstrained and
			# flipping it at transfer time would inject an arbitrary bias.
			static = static.clone(); static[:, 1] = 0.0
		return torch.cat((data, dose_ch), dim=-1), static

	def _film_maps(self, c, dose_from, dose_to, n_traj_samples):
		"""gamma and beta for a dose change.

		Default: gamma = film_gamma(c), beta = film_beta(c) -- unconstrained, so
		zero dose change does NOT give the identity (measured offset ~1-6 in
		||z0'-z0||), which is what breaks interpolation under sparse dosing.

		film_residual_dose: reparameterises as
		    gamma = 1 + (d2-d1) * u(c),   beta = (d2-d1) * v(c)
		so d2 == d1 yields exactly gamma=1, beta=0 by construction. The identity
		anchor becomes structural instead of a penalty term.
		"""
		g_raw, b_raw = self.film_gamma(c), self.film_beta(c)
		if not getattr(self, 'film_residual_dose', False):
			return g_raw, b_raw
		delta = (dose_to - dose_from).view(1, -1, 1).repeat(n_traj_samples, 1, 1)
		return 1.0 + delta * g_raw, delta * b_raw

	def _occ_context(self, occ_a, occ_b, n_traj_samples):
		"""[delta_days, Ht_a, Ht_b] from the two per-occasion vectors [days/100, Ht/30].

		The elapsed time enters as a DIFFERENCE (the transport is a step between two
		occasions) while both hematocrits enter in levels, because clearance depends
		on the absolute value at each occasion, not on its change.
		"""
		if occ_a is None or occ_b is None:
			raise RuntimeError("film_occ is on but the batch carries no occ_v1/occ_v2")
		blk = torch.stack([occ_b[:, 0] - occ_a[:, 0], occ_a[:, 1], occ_b[:, 1]], dim=1)
		return blk.unsqueeze(0).repeat(n_traj_samples, 1, 1).to(self.device)

	def _film_context(self, c_doses, z0, occ_ctx=None):
		"""Conditioning vector for the FiLM maps.

		By default this is [d1, d2, z0], so gamma and beta depend on the patient's
		latent state -- which makes the transport NOT affine in z0, despite the
		paper's description, and lets the maps identify which training dose pair
		they are in from z0 alone (z0 decodes d1 at R^2 = 0.87). With
		film_no_z0_cond the z0 block is zeroed, leaving gamma,beta = f(d1,d2):
		a genuinely affine transport whose dose dependence must be learned as a
		function of dose alone.
		"""
		if getattr(self, 'film_no_z0_cond', False):
			z0 = torch.zeros_like(z0)
		if getattr(self, 'n_film_occ', 0):
			if occ_ctx is None:
				raise RuntimeError("model built with film_occ but _film_context got occ_ctx=None")
			return torch.cat([c_doses, occ_ctx, z0], dim=-1)
		return torch.cat([c_doses, z0], dim=-1)

	def get_reconstruction(self, time_steps_to_predict, truth, truth_time_steps, 
		mask = None, n_traj_samples = 1, run_backwards = True, mode = None, dose = None, static = None):

		if isinstance(self.encoder_z0, Encoder_z0_ODE_RNN) or \
			isinstance(self.encoder_z0, Encoder_z0_RNN):
			truth_w_mask = truth
			if mask is not None:
				truth_w_mask = torch.cat((truth, mask), -1)
				# truth_w_mask = torch.cat((truth_w_mask, dose), -1)
			elif dose is not None:
				try:
					truth_w_mask = torch.cat((truth, dose), -1)
				except:
					dose_expanded = dose.view(-1, 1, 1).expand(-1, truth.shape[1], 1)
					truth_w_mask = torch.cat((truth, dose_expanded), -1)  
				# static = torch.cat((dose, static), -1)
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

	def get_reconstruction_extrapolation(self, data_v1, time_steps_v1, time_steps_v2, dose_v1, dose_v2, delta_t, t_v1 = 0.0, time_steps_to_predict_v1=None, static_v1=None, n_traj_samples=1, occ_v1=None, occ_v2=None):
		"""
        Extrapolates a new trajectory for Visit 2 given Visit 1 data and the Dose change via FiLM.
        Also returns the base prediction for Visit 1 using its full prediction time steps.
        """
        # Fallback in case we don't pass full V1 time steps
		if time_steps_to_predict_v1 is None:
			time_steps_to_predict_v1 = time_steps_v1

		truth_w_mask, static_v1 = self._encoder_input(data_v1, dose_v1, static_v1)

        # 1. Encode Visit 1 (Old Dose)
		first_point_mu, first_point_std = self.encoder_z0(truth_w_mask, time_steps_v1, static=static_v1, run_backwards=True)
		z0_old = utils.sample_standard_gaussian(
            first_point_mu.repeat(n_traj_samples, 1, 1), 
            first_point_std.repeat(n_traj_samples, 1, 1).abs()
        )

        # --- Reconstruct full Visit 1 trajectory for visualization ---
		if self.use_poisson_proc:
			zeros_v1 = torch.zeros(n_traj_samples, z0_old.size(1), self.input_dim).to(self.device)
			z0_old_aug = torch.cat((z0_old, zeros_v1), -1)
		else:
			z0_old_aug = z0_old
        
        # FIX: Solve over the dense tp_to_predict_v1 instead of the sparse time_steps_v1    
		sol_y_v1 = self.diffeq_solver(z0_old_aug, time_steps_to_predict_v1)
		pred_x_v1 = self.decoder(sol_y_v1)

        # 2. FiLM Modulation
		# 2. Context-Aware FiLM Modulation
		c_doses = torch.stack([dose_v1, dose_v2], dim=1).to(self.device)
		c_doses = c_doses.unsqueeze(0).repeat(n_traj_samples, 1, 1)
		occ_ctx = (self._occ_context(occ_v1, occ_v2, n_traj_samples)
		           if getattr(self, 'n_film_occ', 0) else None)
		c = self._film_context(c_doses, z0_old_aug.detach(), occ_ctx)
          		
		if self.film_time:
			delta = torch.stack([delta_t.squeeze(-1), t_v1.squeeze(-1)], dim=1).to(self.device)
			delta = delta.unsqueeze(0).repeat(n_traj_samples, 1, 1)
			c_t = torch.cat([delta, z0_old_aug.detach()], dim=-1)  
			gamma_t = self.film_gamma_t(c_t) 
			beta_t = self.film_beta_t(c_t)
			z0_old = (z0_old * gamma_t) + beta_t
		
		gamma, beta = self._film_maps(c, dose_v1, dose_v2, n_traj_samples)
		z0_new = (z0_old * gamma) + beta

        # 3. Decode for Visit 2 (New Dose)
		if self.use_poisson_proc:
			zeros_v2 = torch.zeros(n_traj_samples, z0_new.size(1), self.input_dim).to(self.device)
			z0_new_aug = torch.cat((z0_new, zeros_v2), -1)
		else:
			z0_new_aug = z0_new

		sol_y_v2 = self.diffeq_solver(z0_new_aug, time_steps_v2)
		pred_x_v2 = self.decoder(sol_y_v2)

		return pred_x_v2, {"latent_traj": sol_y_v2, "pred_x_v1": pred_x_v1}
     
	def compute_film_losses(self, batch_dict, n_traj_samples=1, kl_coef=1.0, ot_coef=1.0, max_out=None):
		"""
        Bidirectional End-to-End FiLM Loss with IWAE, Gaussian Log-Likelihood, 
        and Explicit Bures-Wasserstein Optimal Transport Penalty.
        """

		def get_encoded_distribution(data_enc, tp_enc, static_enc, dose_enc):
			truth_w_mask, static_enc = self._encoder_input(data_enc, dose_enc, static_enc)
			mu, std = self.encoder_z0(truth_w_mask, tp_enc, static=static_enc, run_backwards=True)
			return mu, std.abs().clamp(min=1e-5)

		# Pre-compute the true distributions for V1 and V2 to act as OT targets
		true_mu_v1, true_std_v1 = get_encoded_distribution(
			batch_dict["observed_data_v1"], batch_dict["observed_tp_v1"], 
			batch_dict.get("static_v1", None), batch_dict["dose_v1"]
		)
		true_mu_v2, true_std_v2 = get_encoded_distribution(
			batch_dict["observed_data_v2"], batch_dict["observed_tp_v2"], 
			batch_dict.get("static_v2", None), batch_dict["dose_v2"]
		)
		
		def compute_directional_loss(data_enc, tp_enc, static_enc, dose_enc, 
                                     tp_target_base, data_target_base,
                                     dose_extrap, tp_target_extrap, data_target_extrap, 
                                     delta_t, t_v1, target_mu, target_std,
                                     occ_base=None, occ_extrap=None): # <--- Added target params
            # 1. Encode Base
			truth_w_mask, static_enc = self._encoder_input(data_enc, dose_enc, static_enc)
			first_point_mu, first_point_std = self.encoder_z0(truth_w_mask, tp_enc, static=static_enc, run_backwards=True)
			# floor as in LatentODEGMM.get_reconstruction below: the encoder can drive
			# the posterior scale to exactly 0 late in training, which makes Normal()
			# raise. Inactive for healthy runs.
			first_point_std = first_point_std.abs().clamp(min=1e-5)
			means_z0 = first_point_mu.repeat(n_traj_samples, 1, 1)
			sigma_z0 = first_point_std.repeat(n_traj_samples, 1, 1)
			z0 = utils.sample_standard_gaussian(means_z0, sigma_z0)
			# 2. Exact KL Divergence
			fp_distr = Normal(first_point_mu, first_point_std)
			kldiv_z0 = kl_divergence(fp_distr, self.z0_prior)
			kldiv_z0_mean = torch.mean(kldiv_z0,(1,2))
			# kldiv_z0 = torch.mean(kldiv_z0, dim=-1) 
			# kldiv_z0_mean = torch.mean(kldiv_z0)    

			# 3. Base Reconstruction
			z0_aug = torch.cat((z0, torch.zeros(n_traj_samples, z0.size(1), self.input_dim).to(self.device)), -1) if self.use_poisson_proc else z0
			sol_y_base = self.diffeq_solver(z0_aug, tp_target_base)
			pred_base = self.decoder(sol_y_base)
			valid_times_mask = torch.ones_like(data_target_base.permute(2, 0, 1),  dtype=torch.bool)
			rec_lik_base = self.get_gaussian_likelihood(data_target_base, pred_base, mask=valid_times_mask)

            # 4. Context-Aware FiLM Modulation
			c_doses = torch.stack([dose_enc, dose_extrap], dim=1).to(self.device)
			c_doses = c_doses.unsqueeze(0).repeat(n_traj_samples, 1, 1)
			occ_ctx = (self._occ_context(occ_base, occ_extrap, n_traj_samples)
			           if getattr(self, 'n_film_occ', 0) else None)
			c = self._film_context(c_doses, z0_aug.detach(), occ_ctx)
          		
			if self.film_time:
				delta = torch.stack([delta_t.squeeze(-1), t_v1.squeeze(-1)], dim=1).to(self.device)
				delta = delta.unsqueeze(0).repeat(n_traj_samples, 1, 1)
				c_t = torch.cat([delta, z0_aug.detach()], dim=-1)  
				gamma_t = self.film_gamma_t(c_t) 
				beta_t = self.film_beta_t(c_t)
				z0_aug = (z0_aug * gamma_t) + beta_t
			
			gamma, beta = self._film_maps(c, dose_enc, dose_extrap, n_traj_samples)
			z0_extrap = (z0_aug * gamma) + beta		

			# --- self-consistency: transporting to the SAME dose must be the identity ---
			# The generator never produces d2 == d1 (it resamples until they differ), so
			# that region of the FiLM input space is entirely untrained. With sparse dose
			# levels the transport then fits the few (d1,d2) pairs it saw instead of a
			# dose map, and degrades badly at every other target dose -- including the one
			# case whose answer is known exactly. This term supplies that constraint.
			sc_coef = getattr(self, 'self_consistency_coef', 0.0)
			if sc_coef > 0:
				occ_id = (self._occ_context(occ_base, occ_base, n_traj_samples)
				          if getattr(self, 'n_film_occ', 0) else None)
				c_id = self._film_context(
					torch.stack([dose_enc, dose_enc], dim=1).unsqueeze(0).repeat(n_traj_samples, 1, 1),
					z0_aug.detach(), occ_id)
				# detach z0 on BOTH sides: the constraint is on the FiLM maps
				# (gamma(d,d)->1, beta(d,d)->0), not on the encoder. Leaving z0
				# attached lets the term be satisfied by shrinking ||z0|| instead.
				z0_det = z0_aug.detach()
				g_id, b_id = self._film_maps(c_id, dose_enc, dose_enc, n_traj_samples)
				z0_identity = (z0_det * g_id) + b_id
				sc_loss = torch.mean((z0_identity - z0_det) ** 2)
			else:
				sc_loss = torch.zeros((), device=z0_extrap.device)

            # W2 between the transported posterior and the independently encoded target.
			latent_dim = target_mu.size(-1)
			z0_extrap_base = z0_extrap[:, :, :latent_dim] # Ensure we only compare the base latent dims

			if getattr(self, 'w2_analytic', False):
				# The transported posterior is available in closed form. When gamma and
				# beta depend only on (d1, d2) the map z -> gamma*z + beta is a fixed
				# diagonal affine transform, so the pushforward of N(mu, sigma^2) is
				# exactly N(gamma*mu + beta, (|gamma|*sigma)^2). Estimating those two
				# moments from n_traj_samples draws instead costs real accuracy: the
				# standard deviation is off by 38% on average at 3 samples, 19% at 10
				# and 10% at 30, which is the regime training runs in. The guard below
				# refuses the shortcut whenever the map is not actually affine.
				if self.film_time or not getattr(self, 'film_no_z0_cond', False):
					raise RuntimeError(
						"w2_analytic requires an affine transport: pass --film-no-z0-cond "
						"and leave --film-time off. With gamma or beta a function of z0 the "
						"pushforward is not Gaussian and the closed form does not hold.")
				g = gamma[0, :, :latent_dim]
				b = beta[0, :, :latent_dim]
				emp_mu_extrap = g * first_point_mu.squeeze(0) + b
				emp_std_extrap = (g.abs() * first_point_std.squeeze(0)) + 1e-8
			else:
				emp_mu_extrap = torch.mean(z0_extrap_base, dim=0) # Mean across sample dimension [Batch, Latent]
				emp_std_extrap = torch.std(z0_extrap_base, dim=0) + 1e-8 # Std across sample dimension
            # W2^2 for diagonal Gaussians = ||mu_1 - mu_2||^2 + ||sigma_1 - sigma_2||^2
			w2_loss = torch.sum((emp_mu_extrap - target_mu)**2 + (emp_std_extrap - target_std)**2, dim=-1)
			w2_loss_mean = torch.mean(w2_loss)

			# --- explicit transported-width ridge -----------------------------------
			# Estimating the transported moments from S draws does not just add noise:
			# it adds a bias.  With g = gamma*sigma1 the true transported width,
			#     E[W2_hat] = W2_exact + (1/S + 1 - c4(S)^2) * sum_d g_d^2,
			# c4(3) = 0.8862, so the S=3 estimator every earlier run used carried a
			# hidden ridge of weight 0.548 on ||gamma*sigma1||^2.  That term shrank
			# both the posterior width (sigma 0.28 vs 0.80) and the transport gain
			# (|gamma| 0.79 vs 0.99), and removing it cost 3-5 pp.  Stating it here
			# makes it a declared hyperparameter instead of a side effect of S:
			# width_ridge_coef = 0.548 reproduces the old objective in expectation,
			# but with an exact gradient rather than a 3-sample estimate.
			# --- dose-orthogonality of the transport residual ---------------------
			# W2 is a DISTRIBUTIONAL constraint: it matches the first two moments of
			# the transported cloud to the target's.  Two clouds can coincide exactly
			# while each patient sits in the wrong place, and measurably they do --
			# a linear probe recovers the administered first dose from the residual
			# at R^2 = 0.40 (classic) to 0.84 (w2ana) on the control cohort, where
			# the target's own d1 content is 0.015.  No choice of W2 weight touches
			# that, because W2 never sees the pairing.  This term does: it penalises
			# the same linear R^2 the diagnostic reports, so training drives exactly
			# the quantity we measure.
			#
			# The target is DETACHED.  Left attached, the cheapest way to clean the
			# residual is to push the same d1 structure into the target encoder so
			# the difference cancels -- which corrupts the reference rather than
			# fixing the transport.
			orth_coef = getattr(self, 'dose_orth_coef', 0.0)
			if orth_coef > 0:
				if self.film_time or not getattr(self, 'film_no_z0_cond', False):
					raise RuntimeError(
						"dose_orth requires an affine transport: pass --film-no-z0-cond "
						"and leave --film-time off, so gamma/beta are shared across "
						"posterior samples and the transported mean is well defined.")
				g_o = gamma[0, :, :latent_dim]
				b_o = beta[0, :, :latent_dim]
				resid = (g_o * first_point_mu.squeeze(0) + b_o) - target_mu.squeeze(0).detach()
				dose_col = dose_enc.reshape(-1, 1).to(resid.dtype)
				R = resid - resid.mean(0, keepdim=True)
				y = dose_col - dose_col.mean(0, keepdim=True)
				# Closed-form ridge probe -- no adversary to train, no inner loop, and
				# differentiable through the solve.  The ridge is scaled to the data so
				# it stays meaningful if the residual collapses.
				RtR = R.t() @ R
				alpha = getattr(self, 'dose_orth_probe_ridge', 1e-3) * (
					torch.diagonal(RtR).mean().detach() + 1e-8)
				w_o = torch.linalg.solve(
					RtR + alpha * torch.eye(latent_dim, device=R.device, dtype=R.dtype),
					R.t() @ y)
				ss_res = (y - R @ w_o).pow(2).sum()
				ss_tot = y.pow(2).sum() + 1e-8
				# R^2 is scale-free in both R and y, so the term cannot be satisfied by
				# shrinking the residual -- only by removing d1 structure from it.
				dose_orth_loss = (1.0 - ss_res / ss_tot).clamp(min=0.0)
			else:
				dose_orth_loss = torch.zeros((), device=z0_extrap.device)

			ridge_coef = getattr(self, 'width_ridge_coef', 0.0)
			if ridge_coef > 0:
				g_ridge = gamma[0, :, :latent_dim].abs() * first_point_std.squeeze(0)
				width_ridge = torch.mean(torch.sum(g_ridge ** 2, dim=-1))
			else:
				width_ridge = torch.zeros((), device=z0_extrap.device)
            # ---------------------------------------------------------------

            # 5. Extrapolation Reconstruction
			z0_extrap_aug = torch.cat((z0_extrap, torch.zeros(n_traj_samples, z0_extrap.size(1), self.input_dim).to(self.device)), -1) if self.use_poisson_proc else z0_extrap
			sol_y_extrap = self.diffeq_solver(z0_extrap_aug, tp_target_extrap)
			pred_extrap = self.decoder(sol_y_extrap)
            
			rec_lik_extrap = self.get_gaussian_likelihood(data_target_extrap, pred_extrap, mask=None)
			mse_extrap = self.get_mse(data_target_extrap, pred_extrap, mask=None)

			return rec_lik_base, rec_lik_extrap, mse_extrap, kldiv_z0_mean, pred_extrap, first_point_std, w2_loss_mean, sc_loss, width_ridge, dose_orth_loss

        # --- Forward Direction: Encode V1 -> Extrapolate V2 ---
		rec_lik_base_v1, rec_lik_extrap_v2, mse_extrap_v2, kl_v1, pred_extrap_v2, fp_std_v1, w2_loss_v1_to_v2, sc_v1, ridge_v1, orth_v1 = compute_directional_loss(
            data_enc=batch_dict["observed_data_v1"], tp_enc=batch_dict["observed_tp_v1"], static_enc=batch_dict.get("static_v1", None), dose_enc=batch_dict["dose_v1"],
            tp_target_base=batch_dict["tp_to_predict_v1"], data_target_base=batch_dict["data_to_predict_v1"],
            dose_extrap=batch_dict["dose_v2"], t_v1 = batch_dict['t_v1'], delta_t = batch_dict['delta_t'], tp_target_extrap=batch_dict["tp_to_predict_v2"], data_target_extrap=batch_dict["data_to_predict_v2"],
            target_mu=true_mu_v2, target_std=true_std_v2,
            occ_base=batch_dict.get("occ_v1"), occ_extrap=batch_dict.get("occ_v2")
        )
        
		t_v2 = batch_dict['t_v1'] + batch_dict['delta_t']
		reverse_delta_t = -batch_dict['delta_t']
		
        # --- Reverse Direction: Encode V2 -> Extrapolate V1 ---
		rec_lik_base_v2, rec_lik_extrap_v1, mse_extrap_v1, kl_v2, pred_extrap_v1, fp_std_v2, w2_loss_v2_to_v1, sc_v2, ridge_v2, orth_v2 = compute_directional_loss(
            data_enc=batch_dict["observed_data_v2"], tp_enc=batch_dict["observed_tp_v2"], static_enc=batch_dict.get("static_v2", None), dose_enc=batch_dict["dose_v2"],
            tp_target_base=batch_dict["tp_to_predict_v2"], data_target_base=batch_dict["data_to_predict_v2"],
            dose_extrap=batch_dict["dose_v1"], t_v1 = t_v2, delta_t = reverse_delta_t, tp_target_extrap=batch_dict["tp_to_predict_v1"], data_target_extrap=batch_dict["data_to_predict_v1"],
            target_mu=true_mu_v1, target_std=true_std_v1,
            occ_base=batch_dict.get("occ_v2"), occ_extrap=batch_dict.get("occ_v1")
        )

        # --- Aggregate Losses using IWAE bounds ---
		lik_v1_to_v2 = (rec_lik_base_v1 + rec_lik_extrap_v2)
		lik_v2_to_v1 = (rec_lik_base_v2 + rec_lik_extrap_v1) 
		factual_only = getattr(self, 'film_factual_only', False)
		if factual_only:
			# opt-in ablation (--film-factual-only): keep only each visit's reconstruction from its own
			# 3 points; the transported-visit likelihood, W2 and self-consistency are dropped below.
			lik_v1_to_v2 = rec_lik_base_v1
			lik_v2_to_v1 = rec_lik_base_v2
		# lik_v1_to_v2 = (rec_lik_base_v1)
		# lik_v2_to_v1 = (rec_lik_base_v2)
		loss_forward = -torch.logsumexp(lik_v1_to_v2 - kl_coef * kl_v1, 0)
		
		if torch.isnan(loss_forward): loss_forward = -torch.mean(lik_v1_to_v2 - kl_coef * kl_v1, 0)

		loss_reverse = -torch.logsumexp(lik_v2_to_v1 - kl_coef * kl_v2, 0)
		if torch.isnan(loss_reverse): loss_reverse = -torch.mean(lik_v2_to_v1 - kl_coef * kl_v2, 0)
		total_w2_loss = (w2_loss_v1_to_v2 + w2_loss_v2_to_v1) / 2.0
		total_sc_loss = (sc_v1 + sc_v2) / 2.0
		if factual_only:
			total_w2_loss = total_w2_loss * 0.0
			total_sc_loss = total_sc_loss * 0.0
		total_ridge = (ridge_v1 + ridge_v2) / 2.0
		total_dose_orth = (orth_v1 + orth_v2) / 2.0
		total_loss = ((loss_forward + loss_reverse) / 2.0) + ot_coef*total_w2_loss \
					 + getattr(self, 'self_consistency_coef', 0.0) * total_sc_loss \
					 + getattr(self, 'width_ridge_coef', 0.0) * total_ridge \
					 + getattr(self, 'dose_orth_coef', 0.0) * total_dose_orth
		# total_loss = loss_forward
        # Base Metrics for logging
		final_kl = torch.mean((kl_v1 + kl_v2) / 2.0)
		final_lik = torch.mean((rec_lik_extrap_v2 + rec_lik_extrap_v1) / 2.0)
		final_mse = (mse_extrap_v2 + mse_extrap_v1) / 2.0
		final_fp_std = torch.mean((fp_std_v1 + fp_std_v2) / 2.0)
		
        # --- Specialized Clinical Metrics (RMSE AUC & Conditional MSE) ---
		def compute_rmse_auc(pred_extrap, auc_target, tp_target, scaler):
			
			reconstructions = pred_extrap
			scaling_factor = scaler['max_out']
            
			if 'best_lambda' in max_out.keys():
				reconstructions = inv_boxcox(reconstructions.detach().cpu().numpy(), scaler['best_lambda'])
                
			reconstructions = reconstructions * scaling_factor
            
			times = batch_dict.get('y_true_times', tp_target.cpu().numpy())
			if isinstance(times, torch.Tensor): times = times.cpu().numpy()
            
			reference = auc_target * scaling_factor
			predicted = np.mean(reconstructions, axis=0)
			predicted = np.trapezoid(predicted.squeeze(-1), tp_target.cpu().numpy(), axis=1)
            
			return np.sqrt(np.mean((((reference - predicted) / (reference + 1e-8))**2).numpy()))

		rmse_auc_v2 = compute_rmse_auc(pred_extrap_v2, batch_dict["auc_red_v2"], batch_dict["tp_to_predict_v2"], scaler = max_out)
		rmse_auc_v1 = compute_rmse_auc(pred_extrap_v1, batch_dict["auc_red_v1"], batch_dict["tp_to_predict_v1"], scaler = max_out)
		rmse_overall = (rmse_auc_v2 + rmse_auc_v1) / 2.0

        # Conditional MSE (Subgrouped by clinical covariates)
		desired_values = torch.tensor([0, 1, 2, 3], device=self.device)
		mse_cond = []
		static_data = batch_dict.get('static_v1', None)
		if static_data is not None and static_data.size(-1) > 1:
			for cond in desired_values:
				condition_mask = torch.isin(static_data[:,1], cond)
				if condition_mask.sum() > 0:
					mse_c = self.get_mse(
                        batch_dict["data_to_predict_v2"][condition_mask], 
                        pred_extrap_v2[:, condition_mask, :, :],
                        mask=None
                    ).detach()
					mse_cond.append(mse_c)
				else:
					mse_cond.append(torch.tensor(0.0).to(self.device))
		else:
			mse_cond = [torch.tensor(0.0).to(self.device)] * 4

		return {
            "loss": total_loss, 
            "rec_loss_v1": -torch.mean(rec_lik_base_v1).detach(), # Tracked as Neg Log-Likelihood now
            "rec_loss_v2": -torch.mean(rec_lik_extrap_v2).detach(),
            "kl_loss": final_kl.detach(),
            "mse": final_mse.detach(),  
            "likelihood": final_lik.detach(),
            "kl_first_p": final_kl.detach(),
            "std_first_p": final_fp_std.detach(),
            "ce_loss": torch.tensor(0.0).to(self.device), 
            "pois_likelihood": torch.tensor(0.0).to(self.device),
            "mse_cond": torch.stack(mse_cond) if isinstance(mse_cond, list) else mse_cond,
            "mse_v2": mse_extrap_v2.detach(),   # forward-direction counterfactual MSE only
            "rmse_auc": torch.tensor(rmse_overall).to(self.device),
            "sc_loss": total_sc_loss.detach()
        }
	
	def sample_traj_from_prior(self, time_steps_to_predict, n_traj_samples = 1):
		# input_dim = starting_point.size()[-1]
		# starting_point = starting_point.view(1,1,input_dim)

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
        
        first_point_std = first_point_std.abs().clamp(min=1e-5)
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
        # We use a linear solve for stability instead of explicit inverse
        # u = z @ W.T  =>  z = u @ (W.T)^-1
        # In PyTorch linear layers: y = x @ W.T
        
        # We need z such that u = Linear(z)
        # z = u @ W_inverse.T
        W = self.latent_rotation.weight # [Out, In] -> [Dim, Dim]
        
        # Solve W * z_T = u_T
        # z_T = torch.linalg.solve(W, u_samples.T)
        # z = z_T.T
        
        # Alternatively, simpler: z = u @ inverse(W).T
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