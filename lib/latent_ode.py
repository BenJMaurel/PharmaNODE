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
		dose_encoding_net = None, film = False, film_time = False):

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
			film_input_dim = 2 + self.latent_dim  # Doses + t-V1 + delta_t +  Patient Context
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

	def get_reconstruction_extrapolation(self, data_v1, time_steps_v1, time_steps_v2, dose_v1, dose_v2, delta_t, t_v1 = 0.0, time_steps_to_predict_v1=None, static_v1=None, n_traj_samples=1):
		"""
        Extrapolates a new trajectory for Visit 2 given Visit 1 data and the Dose change via FiLM.
        Also returns the base prediction for Visit 1 using its full prediction time steps.
        """
        # Fallback in case we don't pass full V1 time steps
		if time_steps_to_predict_v1 is None:
			time_steps_to_predict_v1 = time_steps_v1

		seq_len = data_v1.size(1)
		dose_expanded = dose_v1.view(-1, 1, 1).expand(-1, seq_len, 1)
		truth_w_mask = torch.cat((data_v1, dose_expanded), dim=-1)

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
		c = torch.cat([c_doses, z0_old_aug.detach()], dim=-1)
          		
		if self.film_time:
			delta = torch.stack([delta_t.squeeze(-1), t_v1.squeeze(-1)], dim=1).to(self.device)
			delta = delta.unsqueeze(0).repeat(n_traj_samples, 1, 1)
			c_t = torch.cat([delta, z0_old_aug.detach()], dim=-1)  
			gamma_t = self.film_gamma_t(c_t) 
			beta_t = self.film_beta_t(c_t)
			z0_old = (z0_old * gamma_t) + beta_t
		
		gamma = self.film_gamma(c) 
		beta = self.film_beta(c)
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

		# --- NEW: Helper to pre-calculate target distributions for the exact OT Loss ---
		def get_encoded_distribution(data_enc, tp_enc, static_enc, dose_enc):
			seq_len = data_enc.size(1)
			dose_expanded = dose_enc.view(-1, 1, 1).expand(-1, seq_len, 1)
			truth_w_mask = torch.cat((data_enc, dose_expanded), dim=-1)
			mu, std = self.encoder_z0(truth_w_mask, tp_enc, static=static_enc, run_backwards=True)
			return mu, std.abs()

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
                                     delta_t, t_v1, target_mu, target_std): # <--- Added target params
            # 1. Encode Base
			seq_len = data_enc.size(1)
			dose_expanded = dose_enc.view(-1, 1, 1).expand(-1, seq_len, 1)
			truth_w_mask = torch.cat((data_enc, dose_expanded), dim=-1)

			first_point_mu, first_point_std = self.encoder_z0(truth_w_mask, tp_enc, static=static_enc, run_backwards=True)
			first_point_std = first_point_std.abs()
			means_z0 = first_point_mu.repeat(n_traj_samples, 1, 1)
			sigma_z0 = first_point_std.repeat(n_traj_samples, 1, 1)
			z0 = utils.sample_standard_gaussian(means_z0, sigma_z0)

			# 2. Exact KL Divergence
			fp_distr = Normal(first_point_mu, first_point_std)
			kldiv_z0 = kl_divergence(fp_distr, self.z0_prior)
			kldiv_z0 = torch.mean(kldiv_z0, dim=-1) 
			kldiv_z0_mean = torch.mean(kldiv_z0)    

			# 3. Base Reconstruction
			z0_aug = torch.cat((z0, torch.zeros(n_traj_samples, z0.size(1), self.input_dim).to(self.device)), -1) if self.use_poisson_proc else z0
			sol_y_base = self.diffeq_solver(z0_aug, tp_target_base)
			pred_base = self.decoder(sol_y_base)
            
			rec_lik_base = self.get_gaussian_likelihood(data_target_base, pred_base, mask=None)

            # 4. Context-Aware FiLM Modulation
			c_doses = torch.stack([dose_enc, dose_extrap], dim=1).to(self.device)
			c_doses = c_doses.unsqueeze(0).repeat(n_traj_samples, 1, 1)
			c = torch.cat([c_doses, z0_aug.detach()], dim=-1)
          		
			if self.film_time:
				delta = torch.stack([delta_t.squeeze(-1), t_v1.squeeze(-1)], dim=1).to(self.device)
				delta = delta.unsqueeze(0).repeat(n_traj_samples, 1, 1)
				c_t = torch.cat([delta, z0_aug.detach()], dim=-1)  
				gamma_t = self.film_gamma_t(c_t) 
				beta_t = self.film_beta_t(c_t)
				z0_aug = (z0_aug * gamma_t) + beta_t
		
			gamma = self.film_gamma(c) 
			beta = self.film_beta(c)
			z0_extrap = (z0_aug * gamma) + beta		

            # --- NEW: Explicit Bures-Wasserstein Optimal Transport Penalty ---
            # We calculate the empirical distribution of the transported samples
            # and penalize the exact W2 distance to the true target distribution.
			latent_dim = target_mu.size(-1)
			z0_extrap_base = z0_extrap[:, :, :latent_dim] # Ensure we only compare the base latent dims
            
			emp_mu_extrap = torch.mean(z0_extrap_base, dim=0) # Mean across sample dimension [Batch, Latent]
			emp_std_extrap = torch.std(z0_extrap_base, dim=0) + 1e-8 # Std across sample dimension
            
            # W2^2 for diagonal Gaussians = ||mu_1 - mu_2||^2 + ||sigma_1 - sigma_2||^2
			w2_loss = torch.sum((emp_mu_extrap - target_mu)**2 + (emp_std_extrap - target_std)**2, dim=-1)
			w2_loss_mean = torch.mean(w2_loss)
            # ---------------------------------------------------------------

            # 5. Extrapolation Reconstruction
			z0_extrap_aug = torch.cat((z0_extrap, torch.zeros(n_traj_samples, z0_extrap.size(1), self.input_dim).to(self.device)), -1) if self.use_poisson_proc else z0_extrap
			sol_y_extrap = self.diffeq_solver(z0_extrap_aug, tp_target_extrap)
			pred_extrap = self.decoder(sol_y_extrap)
            
			rec_lik_extrap = self.get_gaussian_likelihood(data_target_extrap, pred_extrap, mask=None)
			mse_extrap = self.get_mse(data_target_extrap, pred_extrap, mask=None)

			return rec_lik_base, rec_lik_extrap, mse_extrap, kldiv_z0_mean, pred_extrap, first_point_std, w2_loss_mean

        # --- Forward Direction: Encode V1 -> Extrapolate V2 ---
		rec_lik_base_v1, rec_lik_extrap_v2, mse_extrap_v2, kl_v1, pred_extrap_v2, fp_std_v1, w2_loss_v1_to_v2 = compute_directional_loss(
            data_enc=batch_dict["observed_data_v1"], tp_enc=batch_dict["observed_tp_v1"], static_enc=batch_dict.get("static_v1", None), dose_enc=batch_dict["dose_v1"],
            tp_target_base=batch_dict["tp_to_predict_v1"], data_target_base=batch_dict["data_to_predict_v1"],
            dose_extrap=batch_dict["dose_v2"], t_v1 = batch_dict['t_v1'], delta_t = batch_dict['delta_t'], tp_target_extrap=batch_dict["tp_to_predict_v2"], data_target_extrap=batch_dict["data_to_predict_v2"],
            target_mu=true_mu_v2, target_std=true_std_v2 # <--- Passing target V2 distribution
        )
        
		t_v2 = batch_dict['t_v1'] + batch_dict['delta_t']
		reverse_delta_t = -batch_dict['delta_t']
		
        # --- Reverse Direction: Encode V2 -> Extrapolate V1 ---
		rec_lik_base_v2, rec_lik_extrap_v1, mse_extrap_v1, kl_v2, pred_extrap_v1, fp_std_v2, w2_loss_v2_to_v1 = compute_directional_loss(
            data_enc=batch_dict["observed_data_v2"], tp_enc=batch_dict["observed_tp_v2"], static_enc=batch_dict.get("static_v2", None), dose_enc=batch_dict["dose_v2"],
            tp_target_base=batch_dict["tp_to_predict_v2"], data_target_base=batch_dict["data_to_predict_v2"],
            dose_extrap=batch_dict["dose_v1"], t_v1 = t_v2, delta_t = reverse_delta_t, tp_target_extrap=batch_dict["tp_to_predict_v1"], data_target_extrap=batch_dict["data_to_predict_v1"],
            target_mu=true_mu_v1, target_std=true_std_v1 # <--- Passing target V1 distribution
        )

        # --- Aggregate Losses using IWAE bounds ---
		lik_v1_to_v2 = (rec_lik_base_v1 + rec_lik_extrap_v2)
		lik_v2_to_v1 = (rec_lik_base_v2 + rec_lik_extrap_v1) 
        
		loss_forward = -torch.logsumexp(lik_v1_to_v2 - kl_coef * kl_v1, 0)
		if torch.isnan(loss_forward): loss_forward = -torch.mean(lik_v1_to_v2 - kl_coef * kl_v1, 0)

		loss_reverse = -torch.logsumexp(lik_v2_to_v1 - kl_coef * kl_v2, 0)
		if torch.isnan(loss_reverse): loss_reverse = -torch.mean(lik_v2_to_v1 - kl_coef * kl_v2, 0)

		total_w2_loss = (w2_loss_v1_to_v2 + w2_loss_v2_to_v1) / 2.0

        # --- NEW: Inject Optimal Transport into final objective ---
		total_loss = ((loss_forward + loss_reverse) / 2.0) + ot_coef*total_w2_loss

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
			predicted = np.trapz(predicted.squeeze(-1), tp_target.cpu().numpy(), axis=1)
            
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
            "rmse_auc": torch.tensor(rmse_overall).to(self.device)
        }
	# def compute_film_losses(self, batch_dict, n_traj_samples=1, kl_coef=1.0, max_out=None):
	# 	"""
    #     Bidirectional End-to-End FiLM Loss with IWAE and Gaussian Log-Likelihood.
    #     Computes (V1 -> V2) AND (V2 -> V1) simultaneously.
    #     """
		

	# 	def compute_directional_loss(data_enc, tp_enc, static_enc, dose_enc, 
    #                                  tp_target_base, data_target_base,
    #                                  dose_extrap, tp_target_extrap, data_target_extrap, delta_t, t_v1):
    #         # 1. Encode Base
	# 		seq_len = data_enc.size(1)
	# 		dose_expanded = dose_enc.view(-1, 1, 1).expand(-1, seq_len, 1)
	# 		truth_w_mask = torch.cat((data_enc, dose_expanded), dim=-1)

	# 		first_point_mu, first_point_std = self.encoder_z0(truth_w_mask, tp_enc, static=static_enc, run_backwards=True)
	# 		first_point_std = first_point_std.abs()
	# 		means_z0 = first_point_mu.repeat(n_traj_samples, 1, 1)
	# 		sigma_z0 = first_point_std.repeat(n_traj_samples, 1, 1)
	# 		z0 = utils.sample_standard_gaussian(means_z0, sigma_z0)

	# 		# 2. Exact KL Divergence
	# 		fp_distr = Normal(first_point_mu, first_point_std)
	# 		kldiv_z0 = kl_divergence(fp_distr, self.z0_prior)
	# 		kldiv_z0 = torch.mean(kldiv_z0, dim=-1) # Mean over latent dimensions
	# 		kldiv_z0_mean = torch.mean(kldiv_z0)    # Mean over batch
	# 		# kldiv_z0_sample_wise = torch.mean(kldiv_z0, dim=-1)

	# 		# 3. Base Reconstruction
	# 		z0_aug = torch.cat((z0, torch.zeros(n_traj_samples, z0.size(1), self.input_dim).to(self.device)), -1) if self.use_poisson_proc else z0
	# 		sol_y_base = self.diffeq_solver(z0_aug, tp_target_base)
	# 		pred_base = self.decoder(sol_y_base)
            
    #         # Use Gaussian Log-Likelihood instead of MSE
	# 		rec_lik_base = self.get_gaussian_likelihood(data_target_base, pred_base, mask=None)

    #         # 4. Context-Aware FiLM Modulation

	# 		c_doses = torch.stack([dose_enc, dose_extrap], dim=1).to(self.device)
	# 		c_doses = c_doses.unsqueeze(0).repeat(n_traj_samples, 1, 1)
	# 		c = torch.cat([c_doses, z0_aug.detach()], dim=-1)
          		
	# 		if self.film_time:
	# 			delta = torch.stack([delta_t.squeeze(-1), t_v1.squeeze(-1)], dim=1).to(self.device)
	# 			delta = delta.unsqueeze(0).repeat(n_traj_samples, 1, 1)
	# 			c_t = torch.cat([delta, z0_aug.detach()], dim=-1)  
	# 			gamma_t = self.film_gamma_t(c_t) 
	# 			beta_t = self.film_beta_t(c_t)
	# 			z0_aug = (z0_aug * gamma_t) + beta_t
		
	# 		gamma = self.film_gamma(c) 
	# 		beta = self.film_beta(c)
	# 		z0_extrap = (z0_aug * gamma) + beta		

    #         # 5. Extrapolation Reconstruction
	# 		z0_extrap_aug = torch.cat((z0_extrap, torch.zeros(n_traj_samples, z0_extrap.size(1), self.input_dim).to(self.device)), -1) if self.use_poisson_proc else z0_extrap
	# 		sol_y_extrap = self.diffeq_solver(z0_extrap_aug, tp_target_extrap)
	# 		pred_extrap = self.decoder(sol_y_extrap)
            
    #         # Likelihood and tracking MSE for the extrapolation
	# 		rec_lik_extrap = self.get_gaussian_likelihood(data_target_extrap, pred_extrap, mask=None)
	# 		mse_extrap = self.get_mse(data_target_extrap, pred_extrap, mask=None)

	# 		return rec_lik_base, rec_lik_extrap, mse_extrap, kldiv_z0_mean, pred_extrap, first_point_std

    #     # --- Forward Direction: Encode V1 -> Extrapolate V2 ---
	# 	rec_lik_base_v1, rec_lik_extrap_v2, mse_extrap_v2, kl_v1, pred_extrap_v2, fp_std_v1 = compute_directional_loss(
    #         data_enc=batch_dict["observed_data_v1"], tp_enc=batch_dict["observed_tp_v1"], static_enc=batch_dict.get("static_v1", None), dose_enc=batch_dict["dose_v1"],
    #         tp_target_base=batch_dict["tp_to_predict_v1"], data_target_base=batch_dict["data_to_predict_v1"],
    #         dose_extrap=batch_dict["dose_v2"], t_v1 = batch_dict['t_v1'], delta_t = batch_dict['delta_t'], tp_target_extrap=batch_dict["tp_to_predict_v2"], data_target_extrap=batch_dict["data_to_predict_v2"]
    #     )
	# 	t_v2 = batch_dict['t_v1'] + batch_dict['delta_t']
	# 	reverse_delta_t = -batch_dict['delta_t']
    #     # --- Reverse Direction: Encode V2 -> Extrapolate V1 ---
	# 	rec_lik_base_v2, rec_lik_extrap_v1, mse_extrap_v1, kl_v2, pred_extrap_v1, fp_std_v2 = compute_directional_loss(
    #         data_enc=batch_dict["observed_data_v2"], tp_enc=batch_dict["observed_tp_v2"], static_enc=batch_dict.get("static_v2", None), dose_enc=batch_dict["dose_v2"],
    #         tp_target_base=batch_dict["tp_to_predict_v2"], data_target_base=batch_dict["data_to_predict_v2"],
    #         dose_extrap=batch_dict["dose_v1"], t_v1 = t_v2, delta_t = reverse_delta_t, tp_target_extrap=batch_dict["tp_to_predict_v1"], data_target_extrap=batch_dict["data_to_predict_v1"]
    #     )

    #     # --- Aggregate Losses using IWAE bounds ---
		
	# 	lik_v1_to_v2 = (rec_lik_base_v1 + rec_lik_extrap_v2)
	# 	lik_v2_to_v1 = (rec_lik_base_v2 + rec_lik_extrap_v1) 
        
	# 	loss_forward = -torch.logsumexp(lik_v1_to_v2 - kl_coef * kl_v1, 0)
	# 	if torch.isnan(loss_forward): loss_forward = -torch.mean(lik_v1_to_v2 - kl_coef * kl_v1, 0)

	# 	loss_reverse = -torch.logsumexp(lik_v2_to_v1 - kl_coef * kl_v2, 0)
	# 	if torch.isnan(loss_reverse): loss_reverse = -torch.mean(lik_v2_to_v1 - kl_coef * kl_v2, 0)

	# 	total_loss = (loss_forward + loss_reverse) / 2.0

    #     # Base Metrics for logging
	# 	final_kl = torch.mean((kl_v1 + kl_v2) / 2.0)
	# 	final_lik = torch.mean((rec_lik_extrap_v2 + rec_lik_extrap_v1) / 2.0)
	# 	final_mse = (mse_extrap_v2 + mse_extrap_v1) / 2.0
	# 	final_fp_std = torch.mean((fp_std_v1 + fp_std_v2) / 2.0)
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