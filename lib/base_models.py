###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import relu

from . import utils
from .encoder_decoder import *
from .likelihood_eval import *
from scipy.special import inv_boxcox
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.nn.modules.rnn import GRUCell, LSTMCell, RNNCellBase

from torch.distributions.normal import Normal
from torch.distributions import Independent
from torch.nn.parameter import Parameter


def create_classifier(z0_dim, n_labels):
	return nn.Sequential(
			nn.Linear(z0_dim, 300),
			nn.ReLU(),
			nn.Linear(300, 300),
			nn.ReLU(),
			nn.Linear(300, n_labels),)


class Baseline(nn.Module):
	def __init__(self, input_dim, latent_dim, device, 
		obsrv_std = 0.01, use_binary_classif = False,
		classif_per_tp = False,
		use_poisson_proc = False,
		linear_classifier = False,
		n_labels = 1,
		train_classif_w_reconstr = False):
		super(Baseline, self).__init__()

		self.input_dim = input_dim
		self.latent_dim = latent_dim
		self.n_labels = n_labels

		self.obsrv_std = torch.Tensor([obsrv_std]).to(device)
		self.device = device

		self.use_binary_classif = use_binary_classif
		self.classif_per_tp = classif_per_tp
		self.use_poisson_proc = use_poisson_proc
		self.linear_classifier = linear_classifier
		self.train_classif_w_reconstr = train_classif_w_reconstr

		z0_dim = latent_dim
		if use_poisson_proc:
			z0_dim += latent_dim

		if use_binary_classif: 
			if linear_classifier:
				self.classifier = nn.Sequential(
					nn.Linear(z0_dim, n_labels))
			else:
				self.classifier = create_classifier(z0_dim, n_labels)
			utils.init_network_weights(self.classifier)


	def get_gaussian_likelihood(self, truth, pred_y, mask = None):
		# pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
		# truth shape  [n_traj, n_tp, n_dim]
		if mask is not None:
			mask = mask.repeat(pred_y.size(0), 1, 1, 1)

		# Compute likelihood of the data under the predictions
		log_density_data = masked_gaussian_log_density(pred_y, truth, 
			obsrv_std = self.obsrv_std, mask = mask)
		log_density_data = log_density_data.permute(1,0)

		# Compute the total density
		# Take mean over n_traj_samples
		log_density = torch.mean(log_density_data, 0)

		# shape: [n_traj]
		return log_density


	def get_mse(self, truth, pred_y, mask = None):
		# pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
		# truth shape  [n_traj, n_tp, n_dim]
		if mask is not None:
			mask = mask.repeat(pred_y.size(0), 1, 1, 1)

		# Compute likelihood of the data under the predictions
		log_density_data = compute_mse(pred_y, truth, mask = mask)
		# shape: [1]
		return torch.mean(log_density_data)


	def compute_all_losses(self, batch_dict,
		n_tp_to_sample = None, n_traj_samples = 1, kl_coef = 1.):
		
		# Condition on subsampled points
		# Make predictions for all the points
		pred_x, info = self.get_reconstruction(batch_dict["tp_to_predict"], 
			batch_dict["observed_data"], batch_dict["observed_tp"], 
			mask = batch_dict["observed_mask"], dose = batch_dict['dose'], static = batch_dict['static'], n_traj_samples = n_traj_samples,
			mode = batch_dict["mode"])

		# Compute likelihood of all the points
		likelihood = self.get_gaussian_likelihood(batch_dict["data_to_predict"], pred_x,
			mask = batch_dict["mask_predicted_data"])

		mse = self.get_mse(batch_dict["data_to_predict"], pred_x,
			mask = batch_dict["mask_predicted_data"])
		
		################################
		# Compute CE loss for binary classification on Physionet
		# Use only last attribute -- mortatility in the hospital 
		device = get_device(batch_dict["data_to_predict"])
		ce_loss = torch.Tensor([0.]).to(device)
		
		if (batch_dict["labels"] is not None) and self.use_binary_classif:
			if (batch_dict["labels"].size(-1) == 1) or (len(batch_dict["labels"].size()) == 1):
				ce_loss = compute_binary_CE_loss(
					info["label_predictions"], 
					batch_dict["labels"])
			else:
				ce_loss = compute_multiclass_CE_loss(
					info["label_predictions"], 
					batch_dict["labels"],
					mask = batch_dict["mask_predicted_data"])

			if torch.isnan(ce_loss):
				print("label pred")
				print(info["label_predictions"])
				print("labels")
				print( batch_dict["labels"])
				raise Exception("CE loss is Nan!")

		pois_log_likelihood = torch.Tensor([0.]).to(get_device(batch_dict["data_to_predict"]))
		if self.use_poisson_proc:
			pois_log_likelihood = compute_poisson_proc_likelihood(
				batch_dict["data_to_predict"], pred_x, 
				info, mask = batch_dict["mask_predicted_data"])
			# Take mean over n_traj
			pois_log_likelihood = torch.mean(pois_log_likelihood, 1)

		loss = - torch.mean(likelihood)

		if self.use_poisson_proc:
			loss = loss - 0.1 * pois_log_likelihood 

		if self.use_binary_classif:
			if self.train_classif_w_reconstr:
				loss = loss +  ce_loss * 100
			else:
				loss =  ce_loss

		# Take mean over the number of samples in a batch
		results = {}
		results["loss"] = torch.mean(loss)
		results["likelihood"] = torch.mean(likelihood).detach()
		results["mse"] = torch.mean(mse).detach()
		results["pois_likelihood"] = torch.mean(pois_log_likelihood).detach()
		results["ce_loss"] = torch.mean(ce_loss).detach()
		results["kl"] = 0.
		results["kl_first_p"] =  0.
		results["std_first_p"] = 0.

		if batch_dict["labels"] is not None and self.use_binary_classif:
			results["label_predictions"] = info["label_predictions"].detach()
		return results



class VAE_Baseline(nn.Module):
	def __init__(self, input_dim, latent_dim, 
		z0_prior, device,
		obsrv_std = 0.01, 
		use_binary_classif = False,
		classif_per_tp = False,
		use_poisson_proc = False,
		linear_classifier = False,
		n_labels = 1,
		train_classif_w_reconstr = False):

		super(VAE_Baseline, self).__init__()
		
		self.input_dim = input_dim
		self.latent_dim = latent_dim
		self.device = device
		self.n_labels = n_labels

		self.obsrv_std = torch.Tensor([obsrv_std]).to(device)

		self.z0_prior = z0_prior
		self.use_binary_classif = use_binary_classif
		self.classif_per_tp = classif_per_tp
		self.use_poisson_proc = use_poisson_proc
		self.linear_classifier = linear_classifier
		self.train_classif_w_reconstr = train_classif_w_reconstr

		z0_dim = latent_dim
		if use_poisson_proc:
			z0_dim += latent_dim

		if use_binary_classif: 
			if linear_classifier:
				self.classifier = nn.Sequential(
					nn.Linear(z0_dim, n_labels))
			else:
				self.classifier = create_classifier(z0_dim, n_labels)
			utils.init_network_weights(self.classifier)


	def get_gaussian_likelihood(self, truth, pred_y, mask = None):
		# pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
		# truth shape  [n_traj, n_tp, n_dim]
		n_traj, n_tp, n_dim = truth.size()

		# Compute likelihood of the data under the predictions
		truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)
		
		if mask is not None:
			mask = mask.repeat(pred_y.size(0), 1, 1, 1)
		log_density_data = masked_gaussian_log_density(pred_y, truth_repeated, 
			obsrv_std = self.obsrv_std, mask = mask)
		log_density_data = log_density_data.permute(1,0)
		log_density = torch.mean(log_density_data, 1)

		# shape: [n_traj_samples]
		return log_density


	def get_mse(self, truth, pred_y, mask = None):
		# pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
		# truth shape  [n_traj, n_tp, n_dim]
		n_traj, n_tp, n_dim = truth.size()

		# Compute likelihood of the data under the predictions
		truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)
		
		if mask is not None:
			mask = mask.repeat(pred_y.size(0), 1, 1, 1)

		# Compute likelihood of the data under the predictions
		log_density_data = compute_mse(pred_y, truth_repeated, mask = mask)
		# shape: [1]
		return torch.mean(log_density_data)


	def compute_all_losses(self, batch_dict, n_traj_samples = 1, kl_coef = 1.):
		# Condition on subsampled points
		# Make predictions for all the points
		pred_y, info = self.get_reconstruction(
			batch_dict["tp_to_predict"],
			batch_dict["observed_data"],
			batch_dict["observed_tp"],
			mask=batch_dict["observed_mask"],
			dose=batch_dict['dose'],
			static=batch_dict['static'],
			n_traj_samples=n_traj_samples,
			mode=batch_dict["mode"]
		)

		# Compute likelihood of all the points
		likelihood = self.get_gaussian_likelihood(
			batch_dict["data_to_predict"], pred_y,
			mask=batch_dict["mask_predicted_data"]
		)

		mse = self.get_mse(
			batch_dict["data_to_predict"], pred_y,
			mask=batch_dict["mask_predicted_data"]
		)

		#Latent space
		fp_mu, fp_std, fp_enc = info["first_point"]
		fp_std = fp_std.abs()
		fp_distr = Normal(fp_mu, fp_std)
		kldiv_z0 = kl_divergence(fp_distr, self.z0_prior)
		kldiv_z0 = torch.mean(kldiv_z0, (1, 2))

		# Loss
		loss = -torch.logsumexp(likelihood - kl_coef * kldiv_z0, 0)
		if torch.isnan(loss):
			loss = -torch.mean(likelihood - kl_coef * kldiv_z0, 0)

		# The AUC-based RMSE calculation should be performed in the main evaluation loop.
		# The following is an example of how it could be implemented:
		#
		# if max_out:
		# 	reconstructions = pred_y
		# 	dataset = batch_dict['dataset_number']
		# 	original_indices = dataset.cpu().numpy().astype(int)
		# 	scaling_factor = max_out['max_out'][original_indices]
		# 	if 'best_lambda' in max_out.keys():
		# 		data = inv_boxcox(batch_dict["data_to_predict"], max_out['best_lambda'][0])
		# 		reconstructions = inv_boxcox(reconstructions, max_out['best_lambda'][0])
		# 	reconstructions = reconstructions.detach().numpy()*scaling_factor[:, np.newaxis, np.newaxis]
		# 	data = data.detach().numpy()*scaling_factor[:, np.newaxis, np.newaxis]
		# 	times = batch_dict['y_true_times']
		# 	reference = np.trapz(data.squeeze(-1), times, axis=1)
		# 	predicted = np.mean(np.array(reconstructions), axis=0)
		# 	predicted = np.trapz(predicted.squeeze(-1), batch_dict["tp_to_predict"], axis=1)
		# 	rmse_overall = np.sqrt(np.mean(np.array((np.array(reference) - predicted) / reference)**2))
		# else:
		# 	rmse_overall = 0.0
		# results['rmse_auc'] = torch.tensor(rmse_overall)

		# The AUC-based RMSE calculation should be performed in the main evaluation loop.
		# The following is an example of how it could be implemented:
		#
		# if max_out:
		# 	reconstructions = pred_y
		# 	dataset = batch_dict['dataset_number']
		# 	original_indices = dataset.cpu().numpy().astype(int)
		# 	scaling_factor = max_out['max_out'][original_indices]
		# 	if 'best_lambda' in max_out.keys():
		# 		data = inv_boxcox(batch_dict["data_to_predict"], max_out['best_lambda'][0])
		# 		reconstructions = inv_boxcox(reconstructions, max_out['best_lambda'][0])
		# 	reconstructions = reconstructions.detach().numpy()*scaling_factor[:, np.newaxis, np.newaxis]
		# 	data = data.detach().numpy()*scaling_factor[:, np.newaxis, np.newaxis]
		# 	times = batch_dict['y_true_times']
		# 	reference = np.trapz(data.squeeze(-1), times, axis=1)
		# 	predicted = np.mean(np.array(reconstructions), axis=0)
		# 	predicted = np.trapz(predicted.squeeze(-1), batch_dict["tp_to_predict"], axis=1)
		# 	rmse_overall = np.sqrt(np.mean(np.array((np.array(reference) - predicted) / reference)**2))
		# else:
		# 	rmse_overall = 0.0
		# results['rmse_auc'] = torch.tensor(rmse_overall)

		results = {
			"loss": torch.mean(loss),
			"likelihood": torch.mean(likelihood).detach(),
			"mse": torch.mean(mse).detach(),
			"kl_first_p": torch.mean(kldiv_z0).detach(),
			"std_first_p": torch.mean(fp_std).detach(),
			"pred_y": pred_y.detach(),
		}
		if self.use_binary_classif:
			results["ce_loss"] = torch.mean(ce_loss).detach()
		return results



