###########################
# Dose-conditioned latent ODE baseline  (paper revision, Priority 1)
#
# This module is PURELY ADDITIVE: it is not imported by any of the original
# scripts (run_models.py / train_film.py / test_film.py / lib/latent_ode.py),
# so it cannot change the behaviour of the experiments already reported.
#
# It implements the "cheap" competing design suggested by the reviewer:
#   * the SAME ODE-RNN encoder produces q(z0 | O_v, s_i) as in the main model;
#   * the dose is NOT used to transport z0. Instead it is a standing
#     conditioning argument of the generative vector field f_phi(z, tau, d)
#     and/or of the decoder g_theta(z, d);
#   * the SAME z0 must reconstruct both visits, each decoded at its own dose,
#     which is what pushes z0 towards dose invariance.
#
# No FiLM MLPs, no Bures-Wasserstein penalty and (by default) no IWAE bound.
###########################

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
from scipy.special import inv_boxcox

from . import utils
from .base_models import VAE_Baseline
from .encoder_decoder import Encoder_z0_ODE_RNN, Encoder_z0_RNN
from .diffeq_solver import DiffeqSolver
from .ode_func import ODEFunc


#####################################################################################################
#  Vector field  f_phi(z, tau, d)
#####################################################################################################

class ODEFuncDoseCond(nn.Module):
	"""Neural ODE vector field that takes the dose as a standing argument.

	The dose is a per-trajectory constant over the whole integration, so it is
	stashed on the module by `set_dose()` right before the solve instead of
	being carried as an extra (constant) state dimension. This keeps the state
	dimensionality -- and therefore the KL term -- identical to the main model.
	"""

	def __init__(self, latent_dim, ode_func_net, use_time=False, device=torch.device("cpu")):
		super(ODEFuncDoseCond, self).__init__()
		self.latent_dim = latent_dim
		self.use_time = use_time
		self.device = device
		utils.init_network_weights(ode_func_net)
		self.gradient_net = ode_func_net
		self._dose = None
		self._occ = None
		self.n_occ = 0
		# number of function evaluations of the last solve (cheap stiffness proxy)
		self.nfe = 0

	def set_dose(self, dose):
		"""dose: [n_traj_samples, n_traj, 1] (or broadcastable to it)."""
		self._dose = dose

	def set_occ(self, occ):
		"""occ: [n_traj_samples, n_traj, n_occ] occasion covariates [days/100, Ht/30],
		constant over the integration like the dose. None disables the term."""
		self._occ = occ

	def forward(self, t_local, y, backwards=False):
		if self._dose is None:
			raise RuntimeError("ODEFuncDoseCond.set_dose() must be called before integrating.")
		self.nfe += 1
		dose = self._dose.expand(*y.shape[:-1], self._dose.size(-1))
		net_input = torch.cat((y, dose), -1)
		if self.n_occ and self._occ is not None:
			occ = self._occ.expand(*y.shape[:-1], self._occ.size(-1))
			net_input = torch.cat((net_input, occ), -1)
		if self.use_time:
			t_col = torch.full_like(y[..., :1], float(t_local))
			net_input = torch.cat((net_input, t_col), -1)
		grad = self.gradient_net(net_input)
		if backwards:
			grad = -grad
		return grad

	# kept for API symmetry with lib.ode_func.ODEFunc
	def get_ode_gradient_nn(self, t_local, y):
		return self.forward(t_local, y)

	def sample_next_point_from_prior(self, t_local, y):
		return self.forward(t_local, y)


#####################################################################################################
#  Decoder  g_theta(z, d)
#####################################################################################################

class DecoderDoseCond(nn.Module):
	"""Decoder conditioned on the dose.

	With `hidden == 0` this is exactly the original `lib.encoder_decoder.Decoder`
	(a single Linear layer) with one extra input channel for the dose, so the
	comparison against the main model changes one thing only. `hidden > 0`
	gives a one-hidden-layer MLP instead.
	"""

	def __init__(self, latent_dim, input_dim, use_dose=True, hidden=0, n_occ=0, residual=False):
		super(DecoderDoseCond, self).__init__()
		self.use_dose = use_dose
		self.n_occ = n_occ
		n_in = latent_dim + (1 if use_dose else 0) + n_occ
		# residual mode (opt-in, --decoder-residual), identical in spirit to lib.encoder_decoder.Decoder:
		# linear map on [z, dose] + an MLP branch whose last layer starts at zero, so training starts at
		# the linear decoder. The default paths below are unchanged, so existing checkpoints still load.
		self.residual = bool(residual) and bool(hidden) and hidden > 0
		if self.residual:
			self.lin = nn.Linear(n_in, input_dim)
			mlp = nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(), nn.Linear(hidden, input_dim))
			utils.init_network_weights(mlp); utils.init_network_weights(self.lin)
			nn.init.zeros_(mlp[-1].weight); nn.init.zeros_(mlp[-1].bias)
			self.mlp = mlp
			self.decoder = None
			return
		if hidden and hidden > 0:
			decoder = nn.Sequential(
				nn.Linear(n_in, hidden),
				nn.Tanh(),
				nn.Linear(hidden, input_dim))
		else:
			decoder = nn.Sequential(nn.Linear(n_in, input_dim))
		utils.init_network_weights(decoder)
		self.decoder = decoder

	def forward(self, data, dose=None, occ=None):
		"""data: [..., latent_dim]; dose: [S, B, 1]; occ: [S, B, n_occ]."""
		if self.use_dose:
			if dose is None:
				raise RuntimeError("DecoderDoseCond was built with use_dose=True but got dose=None.")
			# data is [S, B, T, L]; dose is [S, B, 1] -> [S, B, T, 1]
			d = dose.unsqueeze(-2).expand(*data.shape[:-1], dose.size(-1))
			data = torch.cat((data, d), -1)
		if self.n_occ:
			if occ is None:
				raise RuntimeError("DecoderDoseCond was built with n_occ>0 but got occ=None.")
			o = occ.unsqueeze(-2).expand(*data.shape[:-1], occ.size(-1))
			data = torch.cat((data, o), -1)
		if getattr(self, 'residual', False):
			return self.lin(data) + self.mlp(data)
		return self.decoder(data)


#####################################################################################################
#  The model
#####################################################################################################

class LatentODEDoseCond(VAE_Baseline):
	"""Latent ODE with dose as an explicit, decoupled conditioning input."""

	def __init__(self, input_dim, latent_dim, encoder_z0, decoder, diffeq_solver,
		z0_prior, device, obsrv_std=None,
		cond_mode="both",          # 'ode' | 'decoder' | 'both'
		n_occ=0,                   # 0 = original model; 2 = [days/100, Ht/30]
		encoder_dose="concat",     # 'concat' (encoder unchanged) | 'zero' (z0 dose-free by construction)
		use_binary_classif=False, use_poisson_proc=False,
		linear_classifier=False, classif_per_tp=False, n_labels=1,
		train_classif_w_reconstr=False):

		super(LatentODEDoseCond, self).__init__(
			input_dim=input_dim, latent_dim=latent_dim,
			z0_prior=z0_prior, device=device, obsrv_std=obsrv_std,
			use_binary_classif=use_binary_classif,
			classif_per_tp=classif_per_tp,
			linear_classifier=linear_classifier,
			use_poisson_proc=use_poisson_proc,
			n_labels=n_labels,
			train_classif_w_reconstr=train_classif_w_reconstr)

		self.encoder_z0 = encoder_z0
		self.diffeq_solver = diffeq_solver
		self.decoder = decoder
		self.cond_mode = cond_mode
		self.n_occ = n_occ
		self.encoder_dose = encoder_dose

	# ------------------------------------------------------------------ #
	#  Encoding
	# ------------------------------------------------------------------ #
	def encode(self, data, time_steps, dose, static=None, run_backwards=True):
		"""Returns (mu, std) of q(z0 | O, s), both [1, n_traj, latent_dim].

		The encoder architecture is byte-for-byte the one used by the main
		model. With `encoder_dose='zero'` the dose channel of the input and the
		dose entry of the static vector are zeroed, which makes z0 dose-free by
		construction rather than only through the loss.
		"""
		seq_len = data.size(1)
		dose_channel = dose.view(-1, 1, 1).expand(-1, seq_len, 1)
		if self.encoder_dose == "zero":
			dose_channel = torch.zeros_like(dose_channel)
			if static is not None:
				static = static.clone()
				static[:, 0] = 0.0
		if getattr(self, 'blank_formulation', False) and static is not None:
			# formulation indicator: constant when training on one formulation, so its
			# weights are unconstrained and flipping it at transfer time injects a bias
			static = static.clone()
			static[:, 1] = 0.0
		truth_w_dose = torch.cat((data, dose_channel), dim=-1)
		mu, std = self.encoder_z0(truth_w_dose, time_steps, static=static,
			run_backwards=run_backwards)
		# see the note in lib/latent_ode.py: the scale can reach exactly 0 late in
		# training and make Normal() raise. Inactive for healthy runs.
		return mu, std.abs().clamp(min=1e-5)

	# ------------------------------------------------------------------ #
	#  Decoding at an arbitrary dose
	# ------------------------------------------------------------------ #
	def decode(self, z0, time_steps_to_predict, dose, occ=None, count_nfe=False):
		"""Integrate f_phi(., ., dose) from z0 and decode at `dose`.

		z0:   [n_traj_samples, n_traj, latent_dim]
		dose: [n_traj] or [n_traj_samples, n_traj, 1]
		"""
		n_traj_samples, n_traj = z0.size(0), z0.size(1)
		if dose.dim() == 1:
			dose = dose.view(1, -1, 1).expand(n_traj_samples, n_traj, 1)
		elif dose.dim() == 2:
			dose = dose.unsqueeze(0).expand(n_traj_samples, n_traj, 1)
		dose = dose.to(z0.device)

		n_occ = getattr(self, 'n_occ', 0)
		if n_occ:
			if occ is None:
				raise RuntimeError("model built with occasion covariates but decode() got occ=None")
			if occ.dim() == 2:
				occ = occ.unsqueeze(0).expand(n_traj_samples, n_traj, occ.size(-1))
			occ = occ.to(z0.device)

		ode_func = self.diffeq_solver.ode_func
		if self.cond_mode in ("ode", "both"):
			ode_func.set_dose(dose)
		else:
			ode_func.set_dose(torch.zeros_like(dose))
		if n_occ:
			ode_func.set_occ(occ if self.cond_mode in ("ode", "both") else torch.zeros_like(occ))
		if count_nfe:
			ode_func.nfe = 0

		sol_y = self.diffeq_solver(z0, time_steps_to_predict)
		pred_x = self.decoder(sol_y,
			dose if self.cond_mode in ("decoder", "both") else None,
			occ=(occ if n_occ and self.cond_mode in ("decoder", "both") else
			     (torch.zeros_like(occ) if n_occ else None)))
		info = {"latent_traj": sol_y, "nfe": ode_func.nfe}
		return pred_x, info

	# ------------------------------------------------------------------ #
	#  Counterfactual prediction (the quantity the paper is about)
	# ------------------------------------------------------------------ #
	def get_reconstruction_counterfactual(self, data_v1, time_steps_v1, dose_v1, dose_v2,
		tp_to_predict_v1, tp_to_predict_v2, static_v1=None, n_traj_samples=1,
		use_posterior_mean=False, occ_v1=None, occ_v2=None):
		"""Encode Visit 1, then decode the SAME z0 at dose_v1 and at dose_v2."""
		mu, std = self.encode(data_v1, time_steps_v1, dose_v1, static=static_v1)
		means_z0 = mu.repeat(n_traj_samples, 1, 1)
		sigma_z0 = std.repeat(n_traj_samples, 1, 1)
		z0 = means_z0 if use_posterior_mean else utils.sample_standard_gaussian(means_z0, sigma_z0)

		pred_v1, info_v1 = self.decode(z0, tp_to_predict_v1, dose_v1, occ=occ_v1)
		pred_v2, info_v2 = self.decode(z0, tp_to_predict_v2, dose_v2, occ=occ_v2)
		return pred_v2, {
			"pred_x_v1": pred_v1,
			"first_point": (mu, std, z0),
			"latent_traj": info_v2["latent_traj"],
			"nfe_v1": info_v1["nfe"], "nfe_v2": info_v2["nfe"],
		}

	# ------------------------------------------------------------------ #
	#  Training loss
	# ------------------------------------------------------------------ #
	def compute_dose_cond_losses(self, batch_dict, n_traj_samples=1, kl_coef=1.0,
		max_out=None, direction="both", iwae=False):
		"""L = -E_q[ log p(y1|z0,d1) + log p(y2|z0,d2) ] + kl_coef * KL(q||p).

		direction:
		  'v1'   -- encode Visit 1 only (the literal design in the guide);
		  'both' -- also encode Visit 2 and decode both visits from it, then
		            average. This mirrors the bidirectional protocol the
		            OT-FiLM model is trained with, so the comparison is fair.
		"""

		def one_direction(data_enc, tp_enc, static_enc, dose_enc,
			tp_self, y_self, dose_other, tp_other, y_other,
			occ_self=None, occ_other=None):

			mu, std = self.encode(data_enc, tp_enc, dose_enc, static=static_enc)
			means_z0 = mu.repeat(n_traj_samples, 1, 1)
			sigma_z0 = std.repeat(n_traj_samples, 1, 1)
			z0 = utils.sample_standard_gaussian(means_z0, sigma_z0)

			kldiv_z0 = kl_divergence(Normal(mu, std), self.z0_prior)
			kl_mean = torch.mean(kldiv_z0, (1, 2))          # [1]

			pred_self, _ = self.decode(z0, tp_self, dose_enc, occ=occ_self)
			pred_other, _ = self.decode(z0, tp_other, dose_other, occ=occ_other)

			lik_self = self.get_gaussian_likelihood(y_self, pred_self, mask=None)
			lik_other = self.get_gaussian_likelihood(y_other, pred_other, mask=None)
			mse_other = self.get_mse(y_other, pred_other, mask=None)

			return dict(lik_self=lik_self, lik_other=lik_other, kl=kl_mean,
				mse_other=mse_other, pred_other=pred_other, std=std)

		fwd = one_direction(
			data_enc=batch_dict["observed_data_v1"], tp_enc=batch_dict["observed_tp_v1"],
			static_enc=batch_dict.get("static_v1", None), dose_enc=batch_dict["dose_v1"],
			tp_self=batch_dict["tp_to_predict_v1"], y_self=batch_dict["data_to_predict_v1"],
			dose_other=batch_dict["dose_v2"],
			tp_other=batch_dict["tp_to_predict_v2"], y_other=batch_dict["data_to_predict_v2"],
			occ_self=batch_dict.get("occ_v1"), occ_other=batch_dict.get("occ_v2"))

		runs = [fwd]
		if direction == "both":
			rev = one_direction(
				data_enc=batch_dict["observed_data_v2"], tp_enc=batch_dict["observed_tp_v2"],
				static_enc=batch_dict.get("static_v2", None), dose_enc=batch_dict["dose_v2"],
				tp_self=batch_dict["tp_to_predict_v2"], y_self=batch_dict["data_to_predict_v2"],
				dose_other=batch_dict["dose_v1"],
				tp_other=batch_dict["tp_to_predict_v1"], y_other=batch_dict["data_to_predict_v1"],
				occ_self=batch_dict.get("occ_v2"), occ_other=batch_dict.get("occ_v1"))
			runs.append(rev)

		losses = []
		for r in runs:
			elbo_terms = (r["lik_self"] + r["lik_other"]) - kl_coef * r["kl"]
			if iwae:
				l = -torch.logsumexp(elbo_terms, 0)
				if torch.isnan(l):
					l = -torch.mean(elbo_terms, 0)
			else:
				l = -torch.mean(elbo_terms, 0)
			losses.append(l)
		total_loss = torch.stack(losses).mean()

		final_kl = torch.mean(torch.stack([r["kl"].mean() for r in runs]))
		final_lik = torch.mean(torch.stack([r["lik_other"].mean() for r in runs]))
		final_mse = torch.mean(torch.stack([r["mse_other"] for r in runs]))
		final_std = torch.mean(torch.stack([r["std"].mean() for r in runs]))

		# --- AUC RMSE on the counterfactual visit, same definition as the FiLM model ---
		rmse_auc = _relative_auc_rmse(fwd["pred_other"], batch_dict["auc_red_v2"],
			batch_dict["tp_to_predict_v2"], max_out)
		if direction == "both":
			rmse_auc = 0.5 * (rmse_auc + _relative_auc_rmse(runs[1]["pred_other"],
				batch_dict["auc_red_v1"], batch_dict["tp_to_predict_v1"], max_out))

		# --- MSE per formulation/genotype subgroup, for logging parity ---
		mse_cond = []
		static_data = batch_dict.get("static_v1", None)
		if static_data is not None and static_data.size(-1) > 1:
			for cond in torch.tensor([0, 1, 2, 3], device=self.device):
				m = torch.isin(static_data[:, 1], cond)
				if m.sum() > 0:
					mse_cond.append(self.get_mse(
						batch_dict["data_to_predict_v2"][m],
						fwd["pred_other"][:, m, :, :], mask=None).detach())
				else:
					mse_cond.append(torch.tensor(0.0).to(self.device))
		else:
			mse_cond = [torch.tensor(0.0).to(self.device)] * 4

		return {
			"loss": total_loss,
			"rec_loss_v1": -torch.mean(fwd["lik_self"]).detach(),
			"rec_loss_v2": -torch.mean(fwd["lik_other"]).detach(),
			"kl_loss": final_kl.detach(),
			"mse": final_mse.detach(),
			"likelihood": final_lik.detach(),
			"kl_first_p": final_kl.detach(),
			"std_first_p": final_std.detach(),
			"ce_loss": torch.tensor(0.0).to(self.device),
			"pois_likelihood": torch.tensor(0.0).to(self.device),
			"mse_cond": torch.stack(mse_cond),
			"rmse_auc": torch.tensor(float(rmse_auc)).to(self.device),
		}


#####################################################################################################
#  Helpers
#####################################################################################################

def _relative_auc_rmse(pred, auc_target, tp_target, scaler):
	"""Relative RMSE on the AUC, computed on the un-scaled (physiological) scale."""
	if scaler is None:
		return float("nan")
	rec = pred.detach().cpu().numpy()
	if isinstance(scaler, dict) and "best_lambda" in scaler:
		rec = inv_boxcox(rec, scaler["best_lambda"])
		rec = np.nan_to_num(rec, nan=0.0)
	max_out = scaler["max_out"] if isinstance(scaler, dict) else scaler
	rec = rec * max_out
	tp = tp_target.detach().cpu().numpy()
	predicted = np.trapezoid(rec.mean(axis=0).squeeze(-1), tp, axis=1)
	reference = auc_target.detach().cpu().numpy() * max_out
	return float(np.sqrt(np.mean(((reference - predicted) / (reference + 1e-8)) ** 2)))


def create_dose_conditioned_model(args, input_dim, z0_prior, obsrv_std, device):
	"""Builds the dose-conditioned baseline.

	The encoder is constructed exactly as in `lib.create_latent_ode_model`; only
	the generative side (vector field / decoder) differs.
	"""
	latent_dim = args.latents
	n_rec_dims = args.rec_dims
	enc_input_dim = int(input_dim) + 1        # observations + dose channel
	gen_data_dim = input_dim

	cond_mode = getattr(args, "cond_mode", "both")
	ode_time = bool(getattr(args, "ode_time", False))

	# ---- generative vector field: f_phi(z, tau, d) ----
	# occasion covariates are OPT-IN: n_occ=0 reproduces the published dims exactly,
	# so every existing checkpoint still loads.
	n_occ = int(getattr(args, "n_occ", 0))
	n_field_inputs = latent_dim + 1 + n_occ + (1 if ode_time else 0)
	ode_func_net = utils.create_net(n_field_inputs, latent_dim,
		n_layers=args.gen_layers, n_units=args.units, nonlinear=torch.nn.Tanh)
	gen_ode_func = ODEFuncDoseCond(latent_dim, ode_func_net,
		use_time=ode_time, device=device).to(device)
	gen_ode_func.n_occ = n_occ
	diffeq_solver = DiffeqSolver(gen_data_dim, gen_ode_func, "dopri5", latent_dim,
		odeint_rtol=1e-3, odeint_atol=1e-4, device=device)

	# ---- encoder: unchanged ODE-RNN ----
	if args.z0_encoder == "odernn":
		rec_ode_func_net = utils.create_net(n_rec_dims, n_rec_dims,
			n_layers=args.rec_layers, n_units=args.units, nonlinear=torch.nn.Tanh)
		rec_ode_func = ODEFunc(input_dim=enc_input_dim, latent_dim=n_rec_dims,
			ode_func_net=rec_ode_func_net, device=device).to(device)
		z0_diffeq_solver = DiffeqSolver(enc_input_dim, rec_ode_func, "euler", latent_dim,
			odeint_rtol=1e-3, odeint_atol=1e-4, device=device)
		encoder_z0 = Encoder_z0_ODE_RNN(n_rec_dims, enc_input_dim, z0_diffeq_solver,
			z0_dim=latent_dim, n_gru_units=args.gru_units, device=device,
			static_dim=int(getattr(args, 'static_dim', 3))).to(device)
	elif args.z0_encoder == "rnn":
		encoder_z0 = Encoder_z0_RNN(latent_dim, enc_input_dim,
			lstm_output_size=n_rec_dims, device=device).to(device)
	else:
		raise Exception("Unknown encoder for Latent ODE model: " + args.z0_encoder)

	# ---- decoder: g_theta(z, d) ----
	decoder = DecoderDoseCond(latent_dim, gen_data_dim,
		use_dose=cond_mode in ("decoder", "both"),
		hidden=int(getattr(args, "decoder_hidden", 0)),
		n_occ=n_occ, residual=getattr(args, "decoder_residual", False)).to(device)

	model = LatentODEDoseCond(
		input_dim=gen_data_dim, latent_dim=latent_dim,
		encoder_z0=encoder_z0, decoder=decoder, diffeq_solver=diffeq_solver,
		z0_prior=z0_prior, device=device, obsrv_std=obsrv_std,
		cond_mode=cond_mode,
		n_occ=n_occ,
		encoder_dose=getattr(args, "encoder_dose", "concat"),
	).to(device)
	return model
