###########################
# Lu, Bender, Jin & Guan -- neural-PK submodule  (paper revision, Priority 4)
#
# A literal port of the PK half of their architecture, built as a standalone
# baseline rather than as a patch on our latent space, so the head-to-head is
# row-for-row. Implemented to the specification in experiments_guide.md:
#
#   * deterministic GRU encoder over the Visit 1 sparse observations
#     (time-after-dose, time, concentration, dosed amount) -> a static
#     per-patient parameter vector. No KL, no sampling, no GMM prior.
#   * a 2-dimensional recurrent PK state (one dosed coordinate, one
#     concentration coordinate), forward-Euler integrated, conditioned on that
#     parameter vector at every step.
#   * two-port dosing: the encoder reads the *history* dose, while the
#     simulation forcing reads whichever dose is being predicted for. The
#     counterfactual is obtained by freezing the encoder output computed from
#     Visit 1 and swapping only the forcing port to d2.
#   * plain L2 loss on reconstructed concentration. No ELBO.
#
# PURELY ADDITIVE: nothing here is imported by the original pipeline, and no
# existing module is modified.
###########################

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
#  Encoder: sparse observation history -> static per-patient parameters
# --------------------------------------------------------------------------- #

class LuPKEncoder(nn.Module):
	"""Deterministic GRU encoder producing a point estimate of the patient's
	static PK parameter vector.

	Input features per observed time point, as specified:
	   [ time-after-dose , time , concentration , dosed amount ]

	In this dataset each visit window contains exactly one dosing event, at the
	start of the window, so time-after-dose coincides with time; both columns
	are kept so the input layout matches the description rather than a
	dataset-specific simplification.
	"""

	def __init__(self, param_dim=8, hidden=64, n_layers=1, n_static=0, emit_x0=False):
		super(LuPKEncoder, self).__init__()
		self.param_dim = param_dim
		self.n_static = n_static
		self.emit_x0 = emit_x0
		self.n_features = 4
		self.gru = nn.GRU(self.n_features, hidden, num_layers=n_layers, batch_first=True)
		self.head = nn.Sequential(
			nn.Linear(hidden + n_static, hidden),
			nn.Tanh(),
			nn.Linear(hidden, param_dim + (2 if emit_x0 else 0)))

	def forward(self, conc, times, dose, static=None):
		"""conc: [B, T, 1]; times: [T] or [B, T]; dose: [B]; static: [B, S] or None."""
		B, T, _ = conc.size()
		if times.dim() == 1:
			times = times.unsqueeze(0).expand(B, T)
		t = times.unsqueeze(-1)                       # [B, T, 1]
		tad = t                                       # one dose per window, at t = 0
		d = dose.view(B, 1, 1).expand(B, T, 1)
		feats = torch.cat((tad, t, conc, d), dim=-1)  # [B, T, 4]

		_, h = self.gru(feats)
		h = h[-1]                                     # [B, hidden]
		if self.n_static > 0:
			if static is None:
				raise RuntimeError("Encoder was built with static inputs but got static=None.")
			h = torch.cat((h, static), dim=-1)
		out = self.head(h)
		if self.emit_x0:
			return out[:, :self.param_dim], out[:, self.param_dim:]
		return out, None                              # [B, param_dim], x0


# --------------------------------------------------------------------------- #
#  Dynamics: 2-dimensional recurrent PK state
# --------------------------------------------------------------------------- #

class LuPKDynamics(nn.Module):
	"""dx/dt = f(x, theta) on a 2-D state (dosed coordinate, concentration
	coordinate). The static parameter vector conditions the field at every step.
	"""

	def __init__(self, param_dim=8, hidden=64, n_layers=2):
		super(LuPKDynamics, self).__init__()
		layers = [nn.Linear(2 + param_dim, hidden), nn.Tanh()]
		for _ in range(max(0, n_layers - 1)):
			layers += [nn.Linear(hidden, hidden), nn.Tanh()]
		layers += [nn.Linear(hidden, 2)]
		self.net = nn.Sequential(*layers)

	def forward(self, x, theta):
		return self.net(torch.cat((x, theta), dim=-1))


# --------------------------------------------------------------------------- #
#  The model
# --------------------------------------------------------------------------- #

DOSED_COORD = 0
CONC_COORD = 1


def _interp_uniform(grid_values, t_grid_max, n_steps, t_out):
	"""Linear interpolation of a trajectory sampled on a uniform grid.

	grid_values: [B, n_steps+1]  values on linspace(0, t_grid_max, n_steps+1)
	t_out:       [T]             query times in [0, t_grid_max]
	returns:     [B, T]
	"""
	dt = t_grid_max / n_steps
	pos = torch.clamp(t_out / dt, 0.0, float(n_steps))
	lo = torch.clamp(pos.floor().long(), 0, n_steps)
	hi = torch.clamp(lo + 1, 0, n_steps)
	w = (pos - lo.to(pos.dtype)).unsqueeze(0)          # [1, T]
	v_lo = grid_values[:, lo]
	v_hi = grid_values[:, hi]
	return v_lo * (1.0 - w) + v_hi * w


class LuNeuralPK(nn.Module):
	"""Encoder + 2-D recurrent PK state + two-port dosing."""

	def __init__(self, encoder, dynamics, n_euler=240, t_max=24.0,
		readout="identity", init_mode="history", n_prior_doses=6,
		device=torch.device("cpu")):
		super(LuNeuralPK, self).__init__()
		self.encoder = encoder
		self.dynamics = dynamics
		self.n_euler = n_euler
		self.t_max = t_max
		self.device = device
		self.readout_mode = readout
		self.init_mode = init_mode
		self.n_prior_doses = n_prior_doses
		if readout == "linear":
			self.readout = nn.Linear(1, 1)
		else:
			self.readout = None
		# forward-Euler steps taken by the last simulate() call
		self.last_n_steps = 0

	# ---------------- history port ----------------
	def encode(self, conc, times, dose, static=None):
		"""Returns (theta, x0); x0 is None unless init_mode == 'encoder'."""
		return self.encoder(conc, times, dose, static=static)

	@staticmethod
	def interval_from_static(static):
		"""Dosing interval in hours. read_tacro encodes static[:,1] as 1 for the
		12 h formulation (Prograf) and 0 for the 24 h one (Advagraf)."""
		if static is None:
			return None
		return torch.where(static[:, 1] > 0.5,
			torch.full_like(static[:, 1], 12.0),
			torch.full_like(static[:, 1], 24.0))

	# ---------------- forcing port ----------------
	def simulate(self, theta, dose, t_out, interval=None, x0=None):
		"""Integrate the 2-D state, injecting the dose into the dosed coordinate at
		each dosing event, and read the concentration coordinate at `t_out`.

		init_mode controls where the trajectory starts:

		  'history' (default) -- start from zero at the beginning of therapy and
		      simulate the full run-in of `n_prior_doses` prior administrations
		      spaced by each patient's dosing interval, so the state entering the
		      observation window is the model's own steady state. This is what
		      makes a zero initial condition coherent: the observations here are
		      steady-state troughs, not a first-in-human profile.
		  'encoder'           -- let the encoder set the initial state directly.
		      Cheaper, and the usual adaptation when only one interval is observed.
		  'zero'              -- start from zero at t=0 with a single dose. Kept
		      only to reproduce earlier runs; it is mis-specified for this dataset,
		      because it pins the prediction at t=0 to exactly zero while the true
		      trough is far from zero.

		theta: [B, param_dim]; dose: [B]; interval: [B] dosing interval in hours;
		t_out: [T] -> returns [B, T, 1]
		"""
		B = theta.size(0)
		device = theta.device
		dtype = theta.dtype
		t_max = max(float(t_out.max()), 1e-6)
		n_win = self.n_euler
		dt = t_max / n_win
		dose_col = dose.view(B).to(device=device, dtype=dtype)

		use_history = (self.init_mode == 'history') and (interval is not None)
		if use_history:
			interval = interval.view(B).to(device=device, dtype=dtype)
			hist_span = float(self.n_prior_doses) * float(interval.max().item())
			n_hist = int(round(hist_span / dt))
		else:
			n_hist = 0
		n_total = n_hist + n_win
		t_start = -n_hist * dt
		self.last_n_steps = n_total

		# impulse schedule: [B, n_total+1], amount delivered at each grid step
		imp = torch.zeros(B, n_total + 1, device=device, dtype=dtype)
		if use_history:
			for j in range(self.n_prior_doses + 1):
				t_d = -interval * float(j)                       # [B]
				k = torch.round((t_d - t_start) / dt).long().clamp(0, n_total)
				imp.scatter_add_(1, k.unsqueeze(1), dose_col.unsqueeze(1))
		else:
			k0 = int(round((0.0 - t_start) / dt))
			imp[:, k0] = dose_col

		if x0 is not None and self.init_mode == 'encoder':
			x = x0.to(device=device, dtype=dtype)
		else:
			x = torch.zeros(B, 2, device=device, dtype=dtype)

		def inject(state, step):
			add = torch.zeros_like(state)
			add[:, DOSED_COORD] = imp[:, step]
			return state + add

		x = inject(x, 0)
		traj = [x]
		for k in range(n_total):
			x = x + dt * self.dynamics(x, theta)
			x = inject(x, k + 1)
			traj.append(x)

		traj = torch.stack(traj, dim=1)                          # [B, n_total+1, 2]
		conc_grid = traj[:, n_hist:, CONC_COORD]                 # window only, [B, n_win+1]
		out = _interp_uniform(conc_grid, t_max, n_win, t_out.to(device))
		out = out.unsqueeze(-1)
		if self.readout is not None:
			out = self.readout(out)
		return out

	# ---------------- convenience: the counterfactual ----------------
	def predict_counterfactual(self, conc_v1, times_v1, dose_v1, dose_v2,
		tp_v1, tp_v2, static=None, interval=None):
		"""Encode Visit 1 once, then decode at d1 and, with theta frozen, at d2."""
		enc_static = static if self.encoder.n_static else None
		theta, x0 = self.encode(conc_v1, times_v1, dose_v1, static=enc_static)
		if interval is None:
			interval = self.interval_from_static(static)
		pred_v1 = self.simulate(theta, dose_v1, tp_v1, interval=interval, x0=x0)
		pred_v2 = self.simulate(theta, dose_v2, tp_v2, interval=interval, x0=x0)
		return pred_v1, pred_v2, theta

	# ---------------- training loss ----------------
	def compute_l2_losses(self, batch_dict, train_mode="reconstruct", max_out=None,
		diagnostics=True):
		"""Plain L2 on reconstructed concentration -- no ELBO, no KL.

		train_mode:
		  'reconstruct'   -- each visit is encoded from its own observations and
		                     reconstructed at its own dose. This is their original
		                     training procedure: the V1 -> V2 counterfactual is a
		                     test-time operation the two-port design affords, and
		                     is never trained on directly.
		  'counterfactual'-- additionally trains the cross terms (theta from V1,
		                     forcing d2, target y2, and the reverse). Not faithful
		                     to their procedure; provided so the head-to-head
		                     cannot be accused of withholding the target task.
		"""
		# the static vector is always needed for the dosing interval; it reaches the
		# encoder only when the model was built to receive covariates
		static_v1 = batch_dict.get("static_v1", None)
		static_v2 = batch_dict.get("static_v2", static_v1)
		interval = self.interval_from_static(static_v1)

		theta_1, x0_1 = self.encode(batch_dict["observed_data_v1"], batch_dict["observed_tp_v1"],
			batch_dict["dose_v1"], static=static_v1 if self.encoder.n_static else None)
		theta_2, x0_2 = self.encode(batch_dict["observed_data_v2"], batch_dict["observed_tp_v2"],
			batch_dict["dose_v2"], static=static_v2 if self.encoder.n_static else None)

		y1 = batch_dict["data_to_predict_v1"]
		y2 = batch_dict["data_to_predict_v2"]
		tp1 = batch_dict["tp_to_predict_v1"]
		tp2 = batch_dict["tp_to_predict_v2"]

		pred_11 = self.simulate(theta_1, batch_dict["dose_v1"], tp1, interval=interval, x0=x0_1)
		pred_22 = self.simulate(theta_2, batch_dict["dose_v2"], tp2, interval=interval, x0=x0_2)
		mse = nn.functional.mse_loss
		loss = mse(pred_11, y1) + mse(pred_22, y2)
		n_terms = 2

		if train_mode == "counterfactual":
			pred_12 = self.simulate(theta_1, batch_dict["dose_v2"], tp2, interval=interval, x0=x0_1)
			pred_21 = self.simulate(theta_2, batch_dict["dose_v1"], tp1, interval=interval, x0=x0_2)
			loss = loss + mse(pred_12, y2) + mse(pred_21, y1)
			n_terms = 4
		loss = loss / n_terms

		out = {"loss": loss}
		if not diagnostics:
			return out

		# the counterfactual is always *measured*, whether or not it is trained on
		with torch.no_grad():
			pred_cf = self.simulate(theta_1, batch_dict["dose_v2"], tp2, interval=interval, x0=x0_1)
			mse_cf = mse(pred_cf, y2)
			rmse_auc = _relative_auc_rmse(pred_cf, batch_dict["auc_red_v2"], tp2, max_out)
			out.update({
				"mse_v1": mse(pred_11, y1).detach(),
				"mse_v2": mse(pred_22, y2).detach(),
				"mse": mse_cf.detach(),
				"mse_counterfactual": mse_cf.detach(),
				"rmse_auc": torch.tensor(float(rmse_auc), device=loss.device),
			})
		return out


# --------------------------------------------------------------------------- #

def _relative_auc_rmse(pred, auc_target, tp_target, scaler):
	"""Relative RMSE on the AUC, on the un-scaled (physiological) scale."""
	if scaler is None:
		return float("nan")
	from scipy.special import inv_boxcox
	rec = pred.detach().cpu().numpy()
	if isinstance(scaler, dict) and "best_lambda" in scaler:
		rec = inv_boxcox(rec, scaler["best_lambda"])
		rec = np.nan_to_num(rec, nan=0.0)
	max_out = scaler["max_out"] if isinstance(scaler, dict) else scaler
	rec = rec * max_out
	tp = tp_target.detach().cpu().numpy()
	predicted = np.trapezoid(rec.squeeze(-1), tp, axis=1)
	reference = auc_target.detach().cpu().numpy() * max_out
	return float(np.sqrt(np.mean(((reference - predicted) / (reference + 1e-8)) ** 2)))


def create_lu_pk_model(args, device):
	# static width follows --static-dim (4 = + hematocrit/35, see read_tacro.set_static_hematocrit);
	# older checkpoints have no static_dim and keep the original 3 channels
	n_static = int(getattr(args, "static_dim", 3)) if getattr(args, "use_static", False) else 0
	init_mode = getattr(args, "init", "history")
	encoder = LuPKEncoder(
		param_dim=args.param_dim,
		hidden=args.enc_hidden,
		n_layers=getattr(args, "enc_layers", 1),
		n_static=n_static,
		emit_x0=(init_mode == "encoder")).to(device)
	dynamics = LuPKDynamics(
		param_dim=args.param_dim,
		hidden=args.dyn_hidden,
		n_layers=getattr(args, "dyn_layers", 2)).to(device)
	model = LuNeuralPK(encoder, dynamics,
		n_euler=args.n_euler,
		t_max=getattr(args, "t_max", 24.0),
		readout=getattr(args, "readout", "identity"),
		init_mode=init_mode,
		n_prior_doses=getattr(args, "n_prior_doses", 6),
		device=device).to(device)
	return model
