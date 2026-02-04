###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Modified version of Yulia Rubanova
###########################

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import relu, softmax, log_softmax
import torch.nn.functional as F
from . import utils
from .encoder_decoder import *
from .likelihood_eval import *
from scipy.special import inv_boxcox
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.nn.modules.rnn import GRUCell, LSTMCell, RNNCellBase
from torch.distributions import Independent
from torch.nn.parameter import Parameter
import torch.distributions as D
import torch.nn.utils.parametrizations as parametrizations

def create_classifier(z0_dim, n_labels):
	return nn.Sequential(
			nn.Linear(z0_dim, 300),
			nn.ReLU(),
			nn.Linear(300, 300),
			nn.ReLU(),
			nn.Linear(300, n_labels),)

class CouplingLayer(nn.Module):
    """
    Affine Coupling Layer for RealNVP.
    Splits the latent vector in two, transforms one half based on the other.
    """
    def __init__(self, input_dim, hidden_dim, mask):
        super(CouplingLayer, self).__init__()
        self.mask = mask
        # Scale and translation networks (s and t)
        # Maps input_dim -> input_dim (but masked parts will be zeroed effectively)
        self.s_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.t_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        # Forward mapping: u -> z (Base to Latent)
        # Used for Sampling
        mx = x * self.mask
        s = self.s_net(mx) * (1 - self.mask)
        t = self.t_net(mx) * (1 - self.mask)
        
        # Formula: y = x * exp(s) + t
        # (Only modifies the unmasked part)
        z = x * torch.exp(s) + t
        
        # Log Determinant of Jacobian
        # sum(s) over the dimensions
        log_det_jacobian = torch.sum(s, dim=-1) 
        return z, log_det_jacobian

    def inverse(self, z):
        # Inverse mapping: z -> u (Latent to Base)
        # Used for Density Estimation (Training)
        mz = z * self.mask
        s = self.s_net(mz) * (1 - self.mask)
        t = self.t_net(mz) * (1 - self.mask)
        
        # Formula: x = (y - t) * exp(-s)
        u = (z - t) * torch.exp(-s)
        
        # Inverse Log Det = - Forward Log Det
        log_det_jacobian = -torch.sum(s, dim=-1)
        return u, log_det_jacobian

class RealNVP(nn.Module):
    """
    A sequence of Coupling Layers.
    """
    def __init__(self, latent_dim, num_layers=4, hidden_dim=32, device='cpu'):
        super(RealNVP, self).__init__()
        self.layers = nn.ModuleList()
        self.device = device
        
        # Create alternating masks
        # [1, 0, 1, 0...] vs [0, 1, 0, 1...]
        for i in range(num_layers):
            mask = torch.zeros(latent_dim).to(device)
            mask[i % 2::2] = 1 
            self.layers.append(CouplingLayer(latent_dim, hidden_dim, mask))

    def forward(self, u):
        # Base (Gaussian) -> Latent (Complex)
        log_det_sum = 0
        z = u
        for layer in self.layers:
            z, log_det = layer(z)
            log_det_sum += log_det
        return z, log_det_sum

    def inverse(self, z):
        # Latent (Complex) -> Base (Gaussian)
        # We process layers in REVERSE order for inverse
        log_det_sum = 0
        u = z
        for layer in reversed(self.layers):
            u, log_det = layer.inverse(u)
            log_det_sum += log_det
        return u, log_det_sum


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


	def compute_all_losses(self, batch_dict, n_traj_samples = 1, kl_coef = 1., max_out = None):
		# Condition on subsampled points
		# Make predictions for all the points
		
		# pred_y, info = self.get_reconstruction(batch_dict["tp_to_predict"], 
		# 	batch_dict["observed_data"], batch_dict["observed_tp"], 
		# 	mask = batch_dict["observed_mask"], dose = batch_dict['dose'], n_traj_samples = n_traj_samples,
		# 	mode = batch_dict["mode"])
		pred_y, info = self.get_reconstruction(batch_dict["tp_to_predict"], 
			batch_dict["observed_data"], batch_dict["observed_tp"], 
			mask = batch_dict["observed_mask"], dose = batch_dict['dose'], static = batch_dict['static'], n_traj_samples = n_traj_samples,
			mode = batch_dict["mode"])
		if max_out:
			reconstructions = pred_y
			dataset = batch_dict['dataset_number']
			original_indices = dataset.cpu().numpy().astype(int)
			scaling_factor = max_out['max_out'][original_indices]
			if 'best_lambda' in max_out.keys():
				data = inv_boxcox(batch_dict["data_to_predict"], max_out['best_lambda'][0])
				reconstructions = inv_boxcox(reconstructions, max_out['best_lambda'][0])
			reconstructions = reconstructions.detach().numpy()*scaling_factor[:, np.newaxis, np.newaxis]
			data = data.detach().numpy()*scaling_factor[:, np.newaxis, np.newaxis]
			# reconstructions[:, batch_dict['static'][:,1].bool(), 50:, :] = 0
			times = batch_dict['y_true_times']
			reference = np.trapz(data.squeeze(-1), times, axis=1)
			# auc_red = batch_dict['auc_red']*scaling_factor
			predicted = np.mean(np.array(reconstructions), axis=0)
			predicted = np.trapz(predicted.squeeze(-1), batch_dict["tp_to_predict"], axis=1)
			# for i in range(len(predicted)):
			# 	import pdb; pdb.set_trace()
			# 	rmse_patient = np.array((np.array(reference[i]) - predicted[i]) / reference[i])**2
			rmse_overall = np.sqrt(np.mean(np.array((np.array(reference) - predicted) / reference)**2))
		
			
		else:
			rmse_overall = 0.0
		indices = torch.searchsorted(batch_dict["tp_to_predict"], batch_dict['y_true_times'])
		mask = torch.zeros(len(pred_y[1]), len(batch_dict["tp_to_predict"]), dtype=torch.float)
		mask.scatter_(1, indices, 1.0)
		mask = mask.unsqueeze(0).unsqueeze(-1).expand(pred_y.shape[0],-1,-1,-1)
		try:
			pred_y = pred_y[mask.bool()].view(mask.size()[0], mask.size()[1], 12, mask.size()[3])
		except:
			try:
				pred_y = pred_y[mask.bool()].view(mask.size()[0], mask.size()[1], 11, mask.size()[3])
			except:
				pred_y = pred_y[mask.bool()].view(mask.size()[0], mask.size()[1], 10, mask.size()[3])
		# print("get_reconstruction done -- computing likelihood")
		fp_mu, fp_std, fp_enc = info["first_point"]
		fp_std = fp_std.abs()
		fp_distr = Normal(fp_mu, fp_std)

		assert(torch.sum(fp_std < 0) == 0.)

		kldiv_z0 = kl_divergence(fp_distr, self.z0_prior)

		if torch.isnan(kldiv_z0).any():
			print(fp_mu)
			print(fp_std)
			raise Exception("kldiv_z0 is Nan!")

		# Mean over number of latent dimensions
		# kldiv_z0 shape: [n_traj_samples, n_traj, n_latent_dims] if prior is a mixture of gaussians (KL is estimated)
		# kldiv_z0 shape: [1, n_traj, n_latent_dims] if prior is a standard gaussian (KL is computed exactly)
		# shape after: [n_traj_samples]
		valid_times_mask = batch_dict['y_true_times'] != 28.1
		
		kldiv_z0 = torch.mean(kldiv_z0,(1,2))
		# Compute likelihood of all the points
		# rec_likelihood = self.get_gaussian_likelihood(
		# 	batch_dict["data_to_predict"], pred_y,
		# 	mask = batch_dict["mask_predicted_data"])
		rec_likelihood = self.get_gaussian_likelihood(
			batch_dict["data_to_predict"], pred_y,
			mask = valid_times_mask)
		mse = self.get_mse(
			batch_dict["data_to_predict"], pred_y,
			mask = valid_times_mask)
		desired_values = torch.tensor([0, 1, 2, 3], device=batch_dict['static'].device)
		mse_cond = []
		for cond in desired_values:
			condition_mask = torch.isin(batch_dict['static'][:,1], cond)
			# The rest of the code remains the same
			mse_cond.append(self.get_mse(
			batch_dict["data_to_predict"][condition_mask], pred_y[:,condition_mask,:,:],
			mask = batch_dict["mask_predicted_data"]).detach())
		pois_log_likelihood = torch.Tensor([0.]).to(get_device(batch_dict["data_to_predict"]))
		if self.use_poisson_proc:
			pois_log_likelihood = compute_poisson_proc_likelihood(
				batch_dict["data_to_predict"], pred_y, 
				info, mask = batch_dict["mask_predicted_data"])
			# Take mean over n_traj
			pois_log_likelihood = torch.mean(pois_log_likelihood, 1)

		################################
		# Compute CE loss for binary classification on Physionet
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

		# IWAE loss
		loss = - torch.logsumexp(rec_likelihood -  kl_coef * kldiv_z0,0)
		if torch.isnan(loss):
			loss = - torch.mean(rec_likelihood - kl_coef * kldiv_z0,0)
			
		if self.use_poisson_proc:
			loss = loss - 0.1 * pois_log_likelihood 

		if self.use_binary_classif:
			if self.train_classif_w_reconstr:
				loss = loss +  ce_loss * 100
			else:
				loss =  ce_loss
		results = {}
		results['rmse_auc'] = torch.tensor(rmse_overall)
		results["loss"] = torch.mean(loss)
		results["likelihood"] = torch.mean(rec_likelihood).detach()
		results["mse"] = torch.mean(mse).detach()
		results['mse_cond'] = torch.tensor(mse_cond)
		results["pois_likelihood"] = torch.mean(pois_log_likelihood).detach()
		results["ce_loss"] = torch.mean(ce_loss).detach()
		results["kl_first_p"] =  torch.mean(kldiv_z0).detach()
		results["std_first_p"] = torch.mean(fp_std).detach()
		if batch_dict["labels"] is not None and self.use_binary_classif:
			results["label_predictions"] = info["label_predictions"].detach()

		return results


class VAE_GMM(VAE_Baseline):
    def __init__(self, input_dim, latent_dim, 
        z0_prior, device,
        n_components=4, # New parameter: Number of clusters
        obsrv_std = 0.01, 
        use_binary_classif = False,
        classif_per_tp = False,
        use_poisson_proc = False,
        linear_classifier = False,
        n_labels = 1,
        train_classif_w_reconstr = False):

        # Initialize Baseline
        super(VAE_GMM, self).__init__(
            input_dim=input_dim, 
            latent_dim=latent_dim, 
            z0_prior=z0_prior, 
            device=device,
            obsrv_std=obsrv_std, 
            use_binary_classif=use_binary_classif,
            classif_per_tp=classif_per_tp,
            use_poisson_proc=use_poisson_proc,
            linear_classifier=linear_classifier,
            n_labels=n_labels,
            train_classif_w_reconstr=train_classif_w_reconstr
        )

        self.n_components = n_components
        
        # -------------------------------------------------------
        # Define Learnable GMM Prior Parameters
        # -------------------------------------------------------
        # Means: [n_components, latent_dim]
        self.prior_means = nn.Parameter(torch.zeros(n_components, latent_dim).to(device))
        self.prior_means.data.uniform_(-2, 2)
        # LogVars: [n_components, latent_dim]
        # Initialize close to 0 (std=1)
        self.prior_logvars = nn.Parameter(torch.zeros(n_components, latent_dim).to(device))
        self.diversity_weight = 5.0
        # Mixture Weights (logits): [n_components]
        # Initialize uniformly
        self.prior_weights_logits = nn.Parameter(torch.zeros(n_components).to(device))

    def get_gmm_log_density(self, z, hard_assignment= True):
        """
        Computes log p(z) where p is the GMM prior.
        z: [batch_size, latent_dim]
        """
        # 1. Compute log probabilities of z under each component
        # z: [batch, 1, dim] vs means: [1, components, dim]
        z = z.unsqueeze(-2) 
        means = self.prior_means.unsqueeze(0)
        logvars = self.prior_logvars.unsqueeze(0)
        
        # Log N(z | mu, sigma) = -0.5 * (log(2pi) + logvar + (z-mu)^2/var)
        # We sum over the latent dimension to get prob of the vector
        log_prob_components = -0.5 * (
            torch.log(torch.tensor(2.0 * np.pi).to(self.device)) + 
            logvars + 
            (z - means)**2 / torch.exp(logvars)
        ).sum(dim=-1) # shape: [batch, n_components]

        # 2. Add mixture weights
        # log(w * N(..)) = log(w) + log(N(..))
        weights_log_softmax = log_softmax(self.prior_weights_logits, dim=0)
        log_prob_weighted = log_prob_components + weights_log_softmax.unsqueeze(0)

        if hard_assignment:
            # Softmax to get probabilities
            probs = F.softmax(log_prob_weighted, dim=-1)
            
            # Hard Argmax (One-Hot)
            index = torch.argmax(probs, dim=-1)
            probs_hard = F.one_hot(index, num_classes=self.n_components).float()
            
            # Straight-Through Estimator:
            # Forward pass: uses probs_hard (1.0 or 0.0)
            # Backward pass: gradients flow through probs (soft)
            probs_st = (probs_hard - probs).detach() + probs
            
            # Calculate log density using ONLY the chosen cluster
            # This prevents the Gaussian from trying to stretch to cover other clusters
            log_density = (probs_st * log_prob_weighted).sum(dim=-1)
            
            # Keep soft responsibilities for diversity loss calculation later
            responsibilities = probs
        else:
            # Standard Soft Mixture (causes the overlap issue)
            log_density = torch.logsumexp(log_prob_weighted, dim=-1)
            responsibilities = F.softmax(log_prob_weighted, dim=-1)
        
        return log_density, responsibilities
        
        return log_density, responsibilities

    def compute_all_losses(self, batch_dict, n_traj_samples = 1, kl_coef = 1., max_out = None):
        # ===========================================================
        # 1. Reconstruction Phase (Inherited logic)
        # ===========================================================
        pred_y, info = self.get_reconstruction(batch_dict["tp_to_predict"], 
            batch_dict["observed_data"], batch_dict["observed_tp"], 
            mask = batch_dict["observed_mask"], dose = batch_dict['dose'], static = batch_dict['static'], n_traj_samples = n_traj_samples,
            mode = batch_dict["mode"])

        # --- Handling Scaling/Inverse BoxCox (From your Baseline) ---
        if max_out:
            reconstructions = pred_y
            dataset = batch_dict['dataset_number']
            original_indices = dataset.cpu().numpy().astype(int)
            scaling_factor = max_out['max_out'][original_indices]
            if 'best_lambda' in max_out.keys():
                data = inv_boxcox(batch_dict["data_to_predict"], max_out['best_lambda'][0])
                reconstructions = inv_boxcox(reconstructions, max_out['best_lambda'][0])
            reconstructions = reconstructions.detach().numpy()*scaling_factor[:, np.newaxis, np.newaxis]
            data = data.detach().numpy()*scaling_factor[:, np.newaxis, np.newaxis]
            times = batch_dict['y_true_times']
            reference = np.trapz(data.squeeze(-1), times, axis=1)
            predicted = np.mean(np.array(reconstructions), axis=0)
            predicted = np.trapz(predicted.squeeze(-1), batch_dict["tp_to_predict"], axis=1)
            rmse_overall = np.sqrt(np.mean(np.array((np.array(reference) - predicted) / reference)**2))
        else:
            rmse_overall = 0.0

        indices = torch.searchsorted(batch_dict["tp_to_predict"], batch_dict['y_true_times'])
        mask = torch.zeros(len(pred_y[1]), len(batch_dict["tp_to_predict"]), dtype=torch.float)
        mask.scatter_(1, indices, 1.0)
        mask = mask.unsqueeze(0).unsqueeze(-1).expand(pred_y.shape[0],-1,-1,-1)
        
        # Robust reshaping
        try:
            pred_y = pred_y[mask.bool()].view(mask.size()[0], mask.size()[1], 12, mask.size()[3])
        except:
            try:
                pred_y = pred_y[mask.bool()].view(mask.size()[0], mask.size()[1], 11, mask.size()[3])
            except:
                pred_y = pred_y[mask.bool()].view(mask.size()[0], mask.size()[1], 10, mask.size()[3])

        valid_times_mask = batch_dict['y_true_times'] != 28.1
        # ===========================================================
        # 2. Compute Likelihoods (MSE / Gaussian)
        # ===========================================================
        rec_likelihood = self.get_gaussian_likelihood(
            batch_dict["data_to_predict"], pred_y,
            mask = valid_times_mask)
        
        mse = self.get_mse(
            batch_dict["data_to_predict"], pred_y,
            mask = valid_times_mask)

        desired_values = torch.tensor([0, 1, 2, 3], device=batch_dict['static'].device)
        mse_cond = []
        for cond in desired_values:
            condition_mask = torch.isin(batch_dict['static'][:,1], cond)
            if condition_mask.sum() > 0:
                mse_cond.append(self.get_mse(
                    batch_dict["data_to_predict"][condition_mask], pred_y[:,condition_mask,:,:],
                    mask = batch_dict["mask_predicted_data"]).detach())
            else:
                mse_cond.append(torch.tensor(0.0))

        # ===========================================================
        # 3. KL Divergence Calculation for GMM Prior
        # ===========================================================
        fp_mu, fp_std, fp_enc = info["first_point"]
        fp_std = fp_std.abs()
        
        # Check numerics
        assert(torch.sum(fp_std < 0) == 0.)
        if torch.isnan(fp_mu).any() or torch.isnan(fp_std).any():
            raise Exception("fp_mu or fp_std is Nan!")

        # A. Sample z0 from the posterior q(z|x)
        # fp_mu shape: [n_traj, latent_dim] (assuming n_traj_samples handled in get_reconstruction loop or here)
        # We use the reparameterization trick
        eps = torch.randn_like(fp_mu)
        z0_samples = fp_mu + fp_std * eps

        # B. Log Density of q(z|x) (Approx Posterior)
        # log N(z | mu, std)
        log_q_z_x = -0.5 * (
            torch.log(torch.tensor(2.0 * np.pi).to(self.device)) + 
            2 * torch.log(fp_std) + 
            (z0_samples - fp_mu)**2 / (fp_std**2)
        ).sum(dim=-1)

        # C. Log Density of p(z) (GMM Prior)
        # This calls our helper function defined above
        log_p_z, cluster_responsibilities = self.get_gmm_log_density(z0_samples)

        # D. KL Divergence approximation
        # KL(q || p) = E_q [ log q - log p ]
        kldiv_z0 = log_q_z_x - log_p_z
        # Mean over batch
        kldiv_z0_mean = torch.mean(kldiv_z0)

		# ===========================================================
        # STRATEGY 2: Diversity Loss (Entropy of mean usage)
        # ===========================================================
        # 1. Average usage of clusters across the batch (for each sample)
        # shape: [n_traj_samples, n_components]
        mean_usage = cluster_responsibilities.mean(dim=1) 
        
        # 2. Calculate Entropy of this usage: -sum(p * log p)
        # We add 1e-8 for numerical stability
        usage_entropy = -torch.sum(mean_usage * torch.log(mean_usage + 1e-8), dim=-1)
        
        # 3. We want to MAXIMIZE entropy (make usage uniform), so we subtract it from loss.
        # Averaging over samples if necessary
        diversity_loss = torch.mean(usage_entropy)
		
        # ===========================================================
        # 4. Auxiliary Losses (Poisson, Classification)
        # ===========================================================
        pois_log_likelihood = torch.Tensor([0.]).to(self.device)
        if self.use_poisson_proc:
            pois_log_likelihood = compute_poisson_proc_likelihood(
                batch_dict["data_to_predict"], pred_y, 
                info, mask = batch_dict["mask_predicted_data"])
            pois_log_likelihood = torch.mean(pois_log_likelihood, 1)

        ce_loss = torch.Tensor([0.]).to(self.device)
        if (batch_dict["labels"] is not None) and self.use_binary_classif:
            if (batch_dict["labels"].size(-1) == 1) or (len(batch_dict["labels"].size()) == 1):
                ce_loss = compute_binary_CE_loss(info["label_predictions"], batch_dict["labels"])
            else:
                ce_loss = compute_multiclass_CE_loss(
                    info["label_predictions"], 
                    batch_dict["labels"],
                    mask = batch_dict["mask_predicted_data"])

        # ===========================================================
        # 5. Final Loss Aggregation
        # ===========================================================
        # IWAE loss style aggregation
        # shape of rec_likelihood: [n_traj_samples]
        # shape of kldiv_z0: [n_traj] (assuming 1 sample for KL calc per batch item usually)
        
        # Align shapes if necessary. If n_traj_samples > 1, we might need to repeat KL
        if rec_likelihood.shape[0] != kldiv_z0.shape[0]:
            # usually rec_likelihood is reduced to [n_traj_samples] in get_gaussian_likelihood
            # but kldiv is per trajectory.
            # For safety, let's take mean of KL across batch
            pass 

        # Standard ELBO: Reconstruction - beta * KL
        # Note: The original code used logsumexp for IWAE. 
        # Here we approximate ELBO simply and substract entropy:
        loss = - torch.mean(rec_likelihood - kl_coef * kldiv_z0_mean) - (self.diversity_weight * usage_entropy)

        if torch.isnan(loss):
            loss = - torch.mean(rec_likelihood - kl_coef * kldiv_z0_mean)
            
        if self.use_poisson_proc:
            loss = loss - 0.1 * pois_log_likelihood 

        if self.use_binary_classif:
            if self.train_classif_w_reconstr:
                loss = loss + ce_loss * 100
            else:
                loss = ce_loss

        # ===========================================================
        # 6. Compile Results
        # ===========================================================
        results = {}
        results['rmse_auc'] = torch.tensor(rmse_overall)
        results["loss"] = loss
        results["likelihood"] = torch.mean(rec_likelihood).detach()
        results["mse"] = torch.mean(mse).detach()
        results['mse_cond'] = torch.tensor(mse_cond) # Can be issue converting list of tensors
        results["pois_likelihood"] = torch.mean(pois_log_likelihood).detach()
        results["ce_loss"] = torch.mean(ce_loss).detach()
        results["kl_first_p"] = kldiv_z0_mean.detach()
        results["std_first_p"] = torch.mean(fp_std).detach()
        
        # Save cluster assignment probabilities (useful for analysis!)
        results["cluster_probs"] = cluster_responsibilities.detach()
        # print(results["cluster_probs"])
        if batch_dict["labels"] is not None and self.use_binary_classif:
            results["label_predictions"] = info["label_predictions"].detach()

        return results

class VAE_GMM_V(VAE_Baseline):
    def __init__(self, input_dim, latent_dim, 
        z0_prior, device,
        n_components=4, 
        obsrv_std = 0.01, 
        use_binary_classif = False,
        classif_per_tp = False,
        use_poisson_proc = False,
        linear_classifier = False,
        n_labels = 1,
        train_classif_w_reconstr = False):

        super(VAE_GMM_V, self).__init__(
            input_dim=input_dim, latent_dim=latent_dim, z0_prior=z0_prior, 
            device=device, obsrv_std=obsrv_std, use_binary_classif=use_binary_classif,
            classif_per_tp=classif_per_tp, use_poisson_proc=use_poisson_proc,
            linear_classifier=linear_classifier, n_labels=n_labels,
            train_classif_w_reconstr=train_classif_w_reconstr
        )

        self.n_components = n_components
        self.latent_dim = latent_dim
        
        # -------------------------------------------------------
        # 1. NEW: Learnable Rotation Layer (Linear Flow)
        # -------------------------------------------------------
        # This rotates z -> u so that the diagonal GMM fits better.
        # Bias=False because rotation happens around the origin relative to the means.
        self.latent_rotation = nn.Linear(latent_dim, latent_dim, bias=False).to(device)
        
        # Initialize as Identity
        self.latent_rotation.weight.data = torch.eye(latent_dim).to(device)
        
        # Apply Orthogonal Constraint
        # This wraps the layer so 'weight' is always computed as an orthogonal matrix
        # parametrizations.orthogonal(self.latent_rotation, "weight")
        self.latent_rotation.weight.requires_grad = False
        # -------------------------------------------------------
        # 2. Diagonal GMM Parameters (Stable)
        # -------------------------------------------------------
        # Means: [n_components, latent_dim]
        self.prior_means = nn.Parameter(torch.zeros(n_components, latent_dim).to(device))
        self.prior_means.data.uniform_(-1, 1)
        
        # LogVars: [n_components, latent_dim] - Simple Diagonal!
        self.prior_logvars = nn.Parameter(torch.zeros(n_components, latent_dim).to(device))
        
        self.diversity_weight = 5.0
        self.prior_weights_logits = nn.Parameter(torch.zeros(n_components).to(device))

    def get_gmm_log_density(self, z, temp=1.0):
        """
        Computes log p(z) = log p(u) + log |det(R)|
        where u = Rotation(z)
        """
        # 1. Rotate z -> u
        # Because of the parametrization, this is a PURE rotation.
        u = self.latent_rotation(z)
        
        # Calculate Log Determinant of the Jacobian (for Linear layer, it's just det(Weight))
        # We use slogdet for numerical stability
        sign, log_abs_det = torch.slogdet(self.latent_rotation.weight)
        
        # If sign is negative (reflection), it's fine, we just need log_abs_det
        # However, for a simple rotation, we generally expect positive det. 
        # We assume the layer is invertible.
        
        # -------------------------------------------------------
        # 2. Compute Log Probabilities of u under Diagonal GMM
        # -------------------------------------------------------
        # u: [batch, latent_dim]
        u = u.unsqueeze(-2) # [batch, 1, dim]
        means = self.prior_means.unsqueeze(0)
        logvars = self.prior_logvars.unsqueeze(0)
        
        # Diagonal Gaussian Log Density
        log_prob_components = -0.5 * (
            torch.log(torch.tensor(2.0 * np.pi).to(self.device)) + 
            logvars + 
            (u - means)**2 / torch.exp(logvars)
        ).sum(dim=-1) # [batch, n_components]

        # 3. Add mixture weights
        weights_log_softmax = F.log_softmax(self.prior_weights_logits, dim=0)
        log_prob_weighted = log_prob_components + weights_log_softmax.unsqueeze(0)

        # -------------------------------------------------------
        # 4. Temperature Annealing (Soft -> Hard)
        # -------------------------------------------------------
        if temp < 1.0:
            responsibilities = F.softmax(log_prob_weighted / temp, dim=-1)
            log_p_u = (responsibilities * log_prob_weighted).sum(dim=-1)
        else:
            log_p_u = torch.logsumexp(log_prob_weighted, dim=-1)
            responsibilities = F.softmax(log_prob_weighted, dim=-1)
            
        # -------------------------------------------------------
        # 5. Add Jacobian Determinant
        # -------------------------------------------------------
        # log p(z) = log p(u) + log |det(dz/du)|^-1 = log p(u) + log |det(R)|
        # Note: If u = Wz, then z = W^-1 u. 
        # The Change of Variable formula: p(z) = p(u) |det du/dz|
        # du/dz = W. So we ADD log|det(W)|.
        log_p_z = log_p_u + log_abs_det

        return log_p_z, responsibilities

    def compute_all_losses(self, batch_dict, n_traj_samples = 1, kl_coef = 1., max_out = None, temp=1.0):
        # ... (Same initial reconstruction logic as before) ...
        pred_y, info = self.get_reconstruction(batch_dict["tp_to_predict"], 
            batch_dict["observed_data"], batch_dict["observed_tp"], 
            mask = batch_dict["observed_mask"], dose = batch_dict['dose'], static = batch_dict['static'], n_traj_samples = n_traj_samples,
            mode = batch_dict["mode"])

        # ... (Scaling logic omitted for brevity, keep your existing code) ...
        
        if max_out:
            reconstructions = pred_y
            dataset = batch_dict['dataset_number']
            original_indices = dataset.cpu().numpy().astype(int)
            scaling_factor = max_out['max_out'][original_indices]
            if 'best_lambda' in max_out.keys():
                data = inv_boxcox(batch_dict["data_to_predict"], max_out['best_lambda'][0])
                reconstructions = inv_boxcox(reconstructions, max_out['best_lambda'][0])
            reconstructions = reconstructions.detach().numpy()*scaling_factor[:, np.newaxis, np.newaxis]
            data = data.detach().numpy()*scaling_factor[:, np.newaxis, np.newaxis]
            times = batch_dict['y_true_times']
            reference = np.trapz(data.squeeze(-1), times, axis=1)
            predicted = np.mean(np.array(reconstructions), axis=0)
            predicted = np.trapz(predicted.squeeze(-1), batch_dict["tp_to_predict"], axis=1)
            rmse_overall = np.sqrt(np.mean(np.array((np.array(reference) - predicted) / reference)**2))
        else:
            rmse_overall = 0.0
            
        # ... (Masking logic omitted, keep your existing code) ...
        indices = torch.searchsorted(batch_dict["tp_to_predict"], batch_dict['y_true_times'])
        mask = torch.zeros(len(pred_y[1]), len(batch_dict["tp_to_predict"]), dtype=torch.float)
        mask.scatter_(1, indices, 1.0)
        mask = mask.unsqueeze(0).unsqueeze(-1).expand(pred_y.shape[0],-1,-1,-1)
        try:
            pred_y = pred_y[mask.bool()].view(mask.size()[0], mask.size()[1], 12, mask.size()[3])
        except:
            try:
                pred_y = pred_y[mask.bool()].view(mask.size()[0], mask.size()[1], 11, mask.size()[3])
            except:
                pred_y = pred_y[mask.bool()].view(mask.size()[0], mask.size()[1], 10, mask.size()[3])
        valid_times_mask = batch_dict['y_true_times'] != 28.1

        rec_likelihood = self.get_gaussian_likelihood(
            batch_dict["data_to_predict"], pred_y,
            mask = valid_times_mask)
        
        mse = self.get_mse(
            batch_dict["data_to_predict"], pred_y,
            mask = valid_times_mask)
            
        desired_values = torch.tensor([0, 1, 2, 3], device=batch_dict['static'].device)
        mse_cond = []
        for cond in desired_values:
            condition_mask = torch.isin(batch_dict['static'][:,1], cond)
            if condition_mask.sum() > 0:
                mse_cond.append(self.get_mse(
                    batch_dict["data_to_predict"][condition_mask], pred_y[:,condition_mask,:,:],
                    mask = batch_dict["mask_predicted_data"]).detach())
            else:
                mse_cond.append(torch.tensor(0.0))

        # ===========================================================
        # 3. KL Divergence Calculation (Updated)
        # ===========================================================
        fp_mu, fp_std, fp_enc = info["first_point"]
        fp_std = fp_std.abs()
        
        eps = torch.randn_like(fp_mu)
        z0_samples = fp_mu + fp_std * eps

        log_q_z_x = -0.5 * (
            torch.log(torch.tensor(2.0 * np.pi).to(self.device)) + 
            2 * torch.log(fp_std) + 
            (z0_samples - fp_mu)**2 / (fp_std**2)
        ).sum(dim=-1)

        # CALL NEW DENSITY FUNCTION WITH TEMP
        log_p_z, cluster_responsibilities = self.get_gmm_log_density(z0_samples, temp=temp)

        kldiv_z0 = log_q_z_x - log_p_z
        kldiv_z0_mean = torch.mean(kldiv_z0)

        # ===========================================================
        # Diversity Loss
        # ===========================================================
        mean_usage = cluster_responsibilities.mean(dim=0) # Mean across batch 
        usage_entropy = -torch.sum(mean_usage * torch.log(mean_usage + 1e-8))
        diversity_loss = self.diversity_weight * usage_entropy
        
        # ... (Auxiliary losses same as before) ...
        pois_log_likelihood = torch.Tensor([0.]).to(self.device)
        if self.use_poisson_proc:
            pois_log_likelihood = utils.compute_poisson_proc_likelihood(
                batch_dict["data_to_predict"], pred_y, 
                info, mask = batch_dict["mask_predicted_data"])
            pois_log_likelihood = torch.mean(pois_log_likelihood, 1)

        ce_loss = torch.Tensor([0.]).to(self.device)
        if (batch_dict["labels"] is not None) and self.use_binary_classif:
            if (batch_dict["labels"].size(-1) == 1) or (len(batch_dict["labels"].size()) == 1):
                ce_loss = utils.compute_binary_CE_loss(info["label_predictions"], batch_dict["labels"])
            else:
                ce_loss = utils.compute_multiclass_CE_loss(
                    info["label_predictions"], 
                    batch_dict["labels"],
                    mask = batch_dict["mask_predicted_data"])

        # Final Loss
        loss = - torch.mean(rec_likelihood - kl_coef * kldiv_z0_mean) - diversity_loss

        if torch.isnan(loss):
            loss = - torch.mean(rec_likelihood - kl_coef * kldiv_z0_mean)
            
        if self.use_poisson_proc:
            loss = loss - 0.1 * pois_log_likelihood 

        if self.use_binary_classif:
            if self.train_classif_w_reconstr:
                loss = loss + ce_loss * 100
            else:
                loss = ce_loss

        results = {}
        results['rmse_auc'] = torch.tensor(rmse_overall)
        results["loss"] = loss
        results["likelihood"] = torch.mean(rec_likelihood).detach()
        results["mse"] = torch.mean(mse).detach()
        results['mse_cond'] = torch.tensor(mse_cond)
        results["pois_likelihood"] = torch.mean(pois_log_likelihood).detach()
        results["ce_loss"] = torch.mean(ce_loss).detach()
        results["kl_first_p"] = kldiv_z0_mean.detach()
        results["std_first_p"] = torch.mean(fp_std).detach()
        results["cluster_probs"] = cluster_responsibilities.detach()
        
        if batch_dict["labels"] is not None and self.use_binary_classif:
            results["label_predictions"] = info["label_predictions"].detach()

        return results
    
class VAE_GMM_V2(VAE_Baseline):
    def __init__(self, input_dim, latent_dim, 
        z0_prior, device,
        n_components=4, # New parameter: Number of clusters
        obsrv_std = 0.01, 
        use_binary_classif = False,
        classif_per_tp = False,
        use_poisson_proc = False,
        linear_classifier = False,
        n_labels = 1,
        train_classif_w_reconstr = False):

        # Initialize Baseline
        super(VAE_GMM_V, self).__init__(
            input_dim=input_dim, 
            latent_dim=latent_dim, 
            z0_prior=z0_prior, 
            device=device,
            obsrv_std=obsrv_std, 
            use_binary_classif=use_binary_classif,
            classif_per_tp=classif_per_tp,
            use_poisson_proc=use_poisson_proc,
            linear_classifier=linear_classifier,
            n_labels=n_labels,
            train_classif_w_reconstr=train_classif_w_reconstr
        )

        self.n_components = n_components
        
        # -------------------------------------------------------
        # Define Learnable GMM Prior Parameters
        # -------------------------------------------------------
        # Means: [n_components, latent_dim]
        
        self.prior_means = nn.Parameter(torch.zeros(n_components, latent_dim).to(device))
        self.prior_means.data.uniform_(-2, 2)
        # LogVars: [n_components, latent_dim]
        # Initialize close to 0 (std=1)
        self.prior_cov_tril = nn.Parameter(
            torch.eye(latent_dim).unsqueeze(0).repeat(n_components, 1, 1).to(device)
        )
        self.diversity_weight = 5.0
        # Mixture Weights (logits): [n_components]
        # Initialize uniformly
        self.prior_weights_logits = nn.Parameter(torch.zeros(n_components).to(device))
		
    def get_gmm_log_density(self, z, hard_assignment = True):
        """
        Computes log p(z) where p is the GMM prior with FULL COVARIANCE.
        z: [batch_size, latent_dim]
        """
        L = torch.tril(self.prior_cov_tril)
        
        # B. Handle Diagonal: Exp() for positivity, but CLAMPED for stability
        diag_mask = torch.eye(self.latent_dim, device=self.device).unsqueeze(0)
        
        # Clamp the raw parameter before exp to prevent overflow
        # min=-5 (std ~0.006), max=3 (std ~20.0)
        L_diag_raw = (L * diag_mask).clamp(min=-5, max=3) 
        
        # C. Reconstruct L with epsilon jitter
        # We add 1e-4 to the diagonal. This acts as a "minimum variance" floor.
        epsilon = 1e-4
        L = L * (1 - diag_mask) + (torch.exp(L_diag_raw) + epsilon) * diag_mask
        
        try:
            gmm_components = D.MultivariateNormal(loc=self.prior_means, scale_tril=L)
            
            # z_expanded: [batch, 1, latent_dim]
            z_expanded = z.unsqueeze(-2)
            log_prob_components = gmm_components.log_prob(z_expanded)
            
        except ValueError as e:
            # Fallback if Cholesky still fails (prevents training crash)
            print(f"Warning: Cholesky failed, using diagonal fallback. Error: {e}")
            # Fallback to diagonal only for this batch
            L_diag_only = L * diag_mask
            gmm_components = D.MultivariateNormal(loc=self.prior_means, scale_tril=L_diag_only)
            log_prob_components = gmm_components.log_prob(z.unsqueeze(-2))

        # 4. Add mixture weights
        weights_log_softmax = log_softmax(self.prior_weights_logits, dim=0)
        log_prob_weighted = log_prob_components + weights_log_softmax.unsqueeze(0)

        if hard_assignment:
            probs = F.softmax(log_prob_weighted, dim=-1)
            index = torch.argmax(probs, dim=-1)
            probs_hard = F.one_hot(index, num_classes=self.n_components).float()
            probs_st = (probs_hard - probs).detach() + probs
            
            log_density = (probs_st * log_prob_weighted).sum(dim=-1)
            responsibilities = probs
        else:
            log_density = torch.logsumexp(log_prob_weighted, dim=-1)
            responsibilities = F.softmax(log_prob_weighted, dim=-1)
        
        return log_density, responsibilities

    def compute_all_losses(self, batch_dict, n_traj_samples = 1, kl_coef = 1., max_out = None):
        # ===========================================================
        # 1. Reconstruction Phase (Inherited logic)
        # ===========================================================
        pred_y, info = self.get_reconstruction(batch_dict["tp_to_predict"], 
            batch_dict["observed_data"], batch_dict["observed_tp"], 
            mask = batch_dict["observed_mask"], dose = batch_dict['dose'], static = batch_dict['static'], n_traj_samples = n_traj_samples,
            mode = batch_dict["mode"])

        # --- Handling Scaling/Inverse BoxCox (From your Baseline) ---
        if max_out:
            reconstructions = pred_y
            dataset = batch_dict['dataset_number']
            original_indices = dataset.cpu().numpy().astype(int)
            scaling_factor = max_out['max_out'][original_indices]
            if 'best_lambda' in max_out.keys():
                data = inv_boxcox(batch_dict["data_to_predict"], max_out['best_lambda'][0])
                reconstructions = inv_boxcox(reconstructions, max_out['best_lambda'][0])
            reconstructions = reconstructions.detach().numpy()*scaling_factor[:, np.newaxis, np.newaxis]
            data = data.detach().numpy()*scaling_factor[:, np.newaxis, np.newaxis]
            times = batch_dict['y_true_times']
            reference = np.trapz(data.squeeze(-1), times, axis=1)
            predicted = np.mean(np.array(reconstructions), axis=0)
            predicted = np.trapz(predicted.squeeze(-1), batch_dict["tp_to_predict"], axis=1)
            rmse_overall = np.sqrt(np.mean(np.array((np.array(reference) - predicted) / reference)**2))
        else:
            rmse_overall = 0.0

        indices = torch.searchsorted(batch_dict["tp_to_predict"], batch_dict['y_true_times'])
        mask = torch.zeros(len(pred_y[1]), len(batch_dict["tp_to_predict"]), dtype=torch.float)
        mask.scatter_(1, indices, 1.0)
        mask = mask.unsqueeze(0).unsqueeze(-1).expand(pred_y.shape[0],-1,-1,-1)
        
        # Robust reshaping
        try:
            pred_y = pred_y[mask.bool()].view(mask.size()[0], mask.size()[1], 12, mask.size()[3])
        except:
            try:
                pred_y = pred_y[mask.bool()].view(mask.size()[0], mask.size()[1], 11, mask.size()[3])
            except:
                pred_y = pred_y[mask.bool()].view(mask.size()[0], mask.size()[1], 10, mask.size()[3])

        valid_times_mask = batch_dict['y_true_times'] != 28.1
        # ===========================================================
        # 2. Compute Likelihoods (MSE / Gaussian)
        # ===========================================================
        rec_likelihood = self.get_gaussian_likelihood(
            batch_dict["data_to_predict"], pred_y,
            mask = valid_times_mask)
        
        mse = self.get_mse(
            batch_dict["data_to_predict"], pred_y,
            mask = valid_times_mask)

        desired_values = torch.tensor([0, 1, 2, 3], device=batch_dict['static'].device)
        mse_cond = []
        for cond in desired_values:
            condition_mask = torch.isin(batch_dict['static'][:,1], cond)
            if condition_mask.sum() > 0:
                mse_cond.append(self.get_mse(
                    batch_dict["data_to_predict"][condition_mask], pred_y[:,condition_mask,:,:],
                    mask = batch_dict["mask_predicted_data"]).detach())
            else:
                mse_cond.append(torch.tensor(0.0))

        # ===========================================================
        # 3. KL Divergence Calculation for GMM Prior
        # ===========================================================
        fp_mu, fp_std, fp_enc = info["first_point"]
        fp_std = fp_std.abs()
        
        # Check numerics
        assert(torch.sum(fp_std < 0) == 0.)
        if torch.isnan(fp_mu).any() or torch.isnan(fp_std).any():
            raise Exception("fp_mu or fp_std is Nan!")

        # A. Sample z0 from the posterior q(z|x)
        # fp_mu shape: [n_traj, latent_dim] (assuming n_traj_samples handled in get_reconstruction loop or here)
        # We use the reparameterization trick
        eps = torch.randn_like(fp_mu)
        z0_samples = fp_mu + fp_std * eps

        # B. Log Density of q(z|x) (Approx Posterior)
        # log N(z | mu, std)
        log_q_z_x = -0.5 * (
            torch.log(torch.tensor(2.0 * np.pi).to(self.device)) + 
            2 * torch.log(fp_std) + 
            (z0_samples - fp_mu)**2 / (fp_std**2)
        ).sum(dim=-1)

        # C. Log Density of p(z) (GMM Prior)
        # This calls our helper function defined above
        log_p_z, cluster_responsibilities = self.get_gmm_log_density(z0_samples)

        # D. KL Divergence approximation
        # KL(q || p) = E_q [ log q - log p ]
        kldiv_z0 = log_q_z_x - log_p_z
        # Mean over batch
        kldiv_z0_mean = torch.mean(kldiv_z0)

		# ===========================================================
        # STRATEGY 2: Diversity Loss (Entropy of mean usage)
        # ===========================================================
        # 1. Average usage of clusters across the batch (for each sample)
        # shape: [n_traj_samples, n_components]
        mean_usage = cluster_responsibilities.mean(dim=1) 
        
        # 2. Calculate Entropy of this usage: -sum(p * log p)
        # We add 1e-8 for numerical stability
        usage_entropy = -torch.sum(mean_usage * torch.log(mean_usage + 1e-8), dim=-1)
        
        # 3. We want to MAXIMIZE entropy (make usage uniform), so we subtract it from loss.
        # Averaging over samples if necessary
        diversity_loss = torch.mean(usage_entropy)
		
        # ===========================================================
        # 4. Auxiliary Losses (Poisson, Classification)
        # ===========================================================
        pois_log_likelihood = torch.Tensor([0.]).to(self.device)
        if self.use_poisson_proc:
            pois_log_likelihood = compute_poisson_proc_likelihood(
                batch_dict["data_to_predict"], pred_y, 
                info, mask = batch_dict["mask_predicted_data"])
            pois_log_likelihood = torch.mean(pois_log_likelihood, 1)

        ce_loss = torch.Tensor([0.]).to(self.device)
        if (batch_dict["labels"] is not None) and self.use_binary_classif:
            if (batch_dict["labels"].size(-1) == 1) or (len(batch_dict["labels"].size()) == 1):
                ce_loss = compute_binary_CE_loss(info["label_predictions"], batch_dict["labels"])
            else:
                ce_loss = compute_multiclass_CE_loss(
                    info["label_predictions"], 
                    batch_dict["labels"],
                    mask = batch_dict["mask_predicted_data"])

        # ===========================================================
        # 5. Final Loss Aggregation
        # ===========================================================
        # IWAE loss style aggregation
        # shape of rec_likelihood: [n_traj_samples]
        # shape of kldiv_z0: [n_traj] (assuming 1 sample for KL calc per batch item usually)
        
        # Align shapes if necessary. If n_traj_samples > 1, we might need to repeat KL
        if rec_likelihood.shape[0] != kldiv_z0.shape[0]:
            # usually rec_likelihood is reduced to [n_traj_samples] in get_gaussian_likelihood
            # but kldiv is per trajectory.
            # For safety, let's take mean of KL across batch
            pass 

        # Standard ELBO: Reconstruction - beta * KL
        # Note: The original code used logsumexp for IWAE. 
        # Here we approximate ELBO simply and substract entropy:
        loss = - torch.mean(rec_likelihood - kl_coef * kldiv_z0_mean) - (self.diversity_weight * usage_entropy)

        if torch.isnan(loss):
            loss = - torch.mean(rec_likelihood - kl_coef * kldiv_z0_mean)
            
        if self.use_poisson_proc:
            loss = loss - 0.1 * pois_log_likelihood 

        if self.use_binary_classif:
            if self.train_classif_w_reconstr:
                loss = loss + ce_loss * 100
            else:
                loss = ce_loss

        # ===========================================================
        # 6. Compile Results
        # ===========================================================
        results = {}
        results['rmse_auc'] = torch.tensor(rmse_overall)
        results["loss"] = loss
        results["likelihood"] = torch.mean(rec_likelihood).detach()
        results["mse"] = torch.mean(mse).detach()
        results['mse_cond'] = torch.tensor(mse_cond) # Can be issue converting list of tensors
        results["pois_likelihood"] = torch.mean(pois_log_likelihood).detach()
        results["ce_loss"] = torch.mean(ce_loss).detach()
        results["kl_first_p"] = kldiv_z0_mean.detach()
        results["std_first_p"] = torch.mean(fp_std).detach()
        
        # Save cluster assignment probabilities (useful for analysis!)
        results["cluster_probs"] = cluster_responsibilities.detach()
        # print(results["cluster_probs"])
        if batch_dict["labels"] is not None and self.use_binary_classif:
            results["label_predictions"] = info["label_predictions"].detach()

        return results
    
class VAE_Flow(VAE_Baseline):
    def __init__(self, input_dim, latent_dim, 
        z0_prior, device,
        num_flow_layers=4, # New param: complexity of the flow
        flow_hidden_dim=32,
        obsrv_std = 0.01, 
        use_binary_classif = False,
        classif_per_tp = False,
        use_poisson_proc = False,
        linear_classifier = False,
        n_labels = 1,
        train_classif_w_reconstr = False):

        # Initialize Baseline
        super(VAE_Flow, self).__init__(
            input_dim=input_dim, 
            latent_dim=latent_dim, 
            z0_prior=z0_prior, 
            device=device,
            obsrv_std=obsrv_std, 
            use_binary_classif=use_binary_classif,
            classif_per_tp=classif_per_tp,
            use_poisson_proc=use_poisson_proc,
            linear_classifier=linear_classifier,
            n_labels=n_labels,
            train_classif_w_reconstr=train_classif_w_reconstr
        )
        
        # -------------------------------------------------------
        # Define Normalizing Flow Prior
        # -------------------------------------------------------
        # We assume the base distribution is Standard Normal N(0, I)
        # The Flow transforms N(0, I) -> Complex p(z)
        self.flow = RealNVP(
            latent_dim=latent_dim, 
            num_layers=num_flow_layers, 
            hidden_dim=flow_hidden_dim, 
            device=device
        ).to(device)

    def get_flow_prior_log_density(self, z):
        """
        Computes log p(z) where p is the Flow-based prior.
        We use the Change of Variables formula:
        log p(z) = log p(u) + log |det(du/dz)|
        """
        # 1. Map z (Complex) back to u (Base Gaussian)
        u, log_det_inv = self.flow.inverse(z) 
        
        # 2. Compute log p(u) where u ~ N(0, I)
        # log N(u|0,I) = -0.5 * (log(2pi) + u^2)
        log_p_u = -0.5 * (
            torch.log(torch.tensor(2.0 * np.pi).to(self.device)) + 
            u**2
        ).sum(dim=-1)

        # 3. Apply change of variables
        # log p(z) = log p(u) + log |det J_{inv}|
        log_p_z = log_p_u + log_det_inv
        
        return log_p_z

    def compute_all_losses(self, batch_dict, n_traj_samples = 1, kl_coef = 1., max_out = None):
        # ===========================================================
        # 1. Reconstruction Phase (Standard)
        # ===========================================================
        pred_y, info = self.get_reconstruction(batch_dict["tp_to_predict"], 
            batch_dict["observed_data"], batch_dict["observed_tp"], 
            mask = batch_dict["observed_mask"], dose = batch_dict['dose'], 
            static = batch_dict['static'], n_traj_samples = n_traj_samples,
            mode = batch_dict["mode"])

        # --- Handling Scaling/Inverse BoxCox ---
        if max_out:
            reconstructions = pred_y
            dataset = batch_dict['dataset_number']
            original_indices = dataset.cpu().numpy().astype(int)
            scaling_factor = max_out['max_out'][original_indices]
            if 'best_lambda' in max_out.keys():
                data = inv_boxcox(batch_dict["data_to_predict"], max_out['best_lambda'][0])
                reconstructions = inv_boxcox(reconstructions, max_out['best_lambda'][0])
            reconstructions = reconstructions.detach().numpy()*scaling_factor[:, np.newaxis, np.newaxis]
            data = data.detach().numpy()*scaling_factor[:, np.newaxis, np.newaxis]
            times = batch_dict['y_true_times']
            reference = np.trapz(data.squeeze(-1), times, axis=1)
            predicted = np.mean(np.array(reconstructions), axis=0)
            predicted = np.trapz(predicted.squeeze(-1), batch_dict["tp_to_predict"], axis=1)
            rmse_overall = np.sqrt(np.mean(np.array((np.array(reference) - predicted) / reference)**2))
        else:
            rmse_overall = 0.0

        valid_times_mask = batch_dict['y_true_times'] != 28.1
        rec_likelihood = self.get_gaussian_likelihood(
            batch_dict["data_to_predict"], pred_y, mask = valid_times_mask)
        
        mse = self.get_mse(batch_dict["data_to_predict"], pred_y, mask = valid_times_mask)

        # ===========================================================
        # 2. KL Divergence with Flow Prior
        # ===========================================================
        fp_mu, fp_std, fp_enc = info["first_point"]
        fp_std = fp_std.abs()
        
        # A. Sample z0 from the posterior q(z|x)
        # We use the reparameterization trick
        eps = torch.randn_like(fp_mu)
        z0_samples = fp_mu + fp_std * eps

        # B. Log Density of q(z|x) (Approx Posterior is diagonal Gaussian)
        log_q_z_x = -0.5 * (
            torch.log(torch.tensor(2.0 * np.pi).to(self.device)) + 
            2 * torch.log(fp_std) + 
            (z0_samples - fp_mu)**2 / (fp_std**2)
        ).sum(dim=-1)

        # C. Log Density of p(z) (Flow Prior)
        # We evaluate the likelihood of our sampled z0 under the learned Flow
        log_p_z = self.get_flow_prior_log_density(z0_samples)

        # D. KL Divergence
        # KL(q || p) = E_q [ log q(z|x) - log p(z) ]
        kldiv_z0 = log_q_z_x - log_p_z
        kldiv_z0_mean = torch.mean(kldiv_z0)

        # ===========================================================
        # 3. Auxiliary Losses
        # ===========================================================
        desired_values = torch.tensor([0, 1, 2, 3], device=batch_dict['static'].device)
        mse_cond = []
        for cond in desired_values:
            condition_mask = torch.isin(batch_dict['static'][:,1], cond)
            if condition_mask.sum() > 0:
                mse_cond.append(self.get_mse(
                    batch_dict["data_to_predict"][condition_mask], pred_y[:,condition_mask,:,:],
                    mask = batch_dict["mask_predicted_data"]).detach())
            else:
                mse_cond.append(torch.tensor(0.0))

        pois_log_likelihood = torch.Tensor([0.]).to(self.device)
        if self.use_poisson_proc:
            pois_log_likelihood = compute_poisson_proc_likelihood(
                batch_dict["data_to_predict"], pred_y, 
                info, mask = batch_dict["mask_predicted_data"])
            pois_log_likelihood = torch.mean(pois_log_likelihood, 1)

        ce_loss = torch.Tensor([0.]).to(self.device)
        if (batch_dict["labels"] is not None) and self.use_binary_classif:
            if (batch_dict["labels"].size(-1) == 1) or (len(batch_dict["labels"].size()) == 1):
                ce_loss = compute_binary_CE_loss(info["label_predictions"], batch_dict["labels"])
            else:
                ce_loss = compute_multiclass_CE_loss(
                    info["label_predictions"], 
                    batch_dict["labels"],
                    mask = batch_dict["mask_predicted_data"])

        # ===========================================================
        # 4. Final Loss
        # ===========================================================
        # ELBO = Likelihood - KL
        # Loss = -ELBO
        loss = - torch.mean(rec_likelihood - kl_coef * kldiv_z0_mean)

        if torch.isnan(loss):
            print("Loss is NaN. Clipping.")
            loss = - torch.mean(rec_likelihood - kl_coef * kldiv_z0_mean)
            
        if self.use_poisson_proc:
            loss = loss - 0.1 * pois_log_likelihood 

        if self.use_binary_classif:
            loss = loss + ce_loss * (100 if self.train_classif_w_reconstr else 1.0)

        results = {}
        results['rmse_auc'] = torch.tensor(rmse_overall)
        results["loss"] = loss
        results["likelihood"] = torch.mean(rec_likelihood).detach()
        results["mse"] = torch.mean(mse).detach()
        results["mse_cond"] = torch.tensor(mse_cond).detach()
        results["ce_loss"] = torch.mean(ce_loss).detach()
        results["kl_first_p"] = kldiv_z0_mean.detach()
        results["std_first_p"] = torch.mean(fp_std).detach()
        
        if batch_dict["labels"] is not None and self.use_binary_classif:
            results["label_predictions"] = info["label_predictions"].detach()

        return results