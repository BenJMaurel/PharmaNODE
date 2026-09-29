###########################
# Priority 1 -- disentangled-decoder ablation: EVALUATION
#
# Reports MPE (bias) and RMSPE (precision) on the AUC for
#   Visit 1  -- reconstruction at the observed dose, and
#   Visit 2  -- zero-shot counterfactual at the new dose, decoded from the SAME z0.
#
# The AUC pipeline (dense 100-point 24 h grid, 100 posterior samples, inverse
# Box-Cox, 12 h cut-off for Prograf) is deliberately identical to test_film.py
# so the three-way table is apples-to-apples.
###########################

import os
import sys
import json
import argparse

import numpy as np

# Quantification limit for the pointwise relative-error metric. Below this the
# denominator is small enough that a few points dominate the whole statistic.
POINTWISE_LLOQ = 1.0
import torch
from scipy.special import inv_boxcox
from torch.utils.data import DataLoader

import lib.utils as utils
from lib.read_tacro import (extract_gen_tac_film, TacroFilmDataset,
                            collate_fn_tacro_film, set_static_hematocrit)
from lib.calibration import calibration_report
from lib.dose_conditioned import create_dose_conditioned_model


def build_parser():
	p = argparse.ArgumentParser('Dose-conditioned latent ODE baseline -- testing')
	p.add_argument('--experiment', type=str, required=True, help="Dataset folder name.")
	p.add_argument('--load', type=str, default=None,
		help="Checkpoint experiment ID (defaults to --experiment).")
	p.add_argument('--ckpt', type=str, default=None,
		help="Explicit path to a .ckpt, overriding --load / --save.")
	p.add_argument('--data-dir', type=str, default='./results/exp_film_run')
	p.add_argument('--save', type=str, default='./results/')
	p.add_argument('--best', action='store_true', default=True,
		help="Load the *_best.ckpt (default).")
	p.add_argument('--last', dest='best', action='store_false',
		help="Load the last checkpoint instead of the best one.")
	p.add_argument('-b', '--batch-size', type=int, default=2000)
	p.add_argument('--n-traj-samples', type=int, default=100)
	p.add_argument('--scale-from', type=str, default=None,
		help="Experiment whose TRAIN csv defines the normalisation, instead of refitting "
		     "it on the evaluated cohort. Required for cross-cohort evaluation. "
		     "Default None reproduces previous behaviour exactly.")
	p.add_argument('--eval-split', type=str, default='test', choices=['test', 'all'],
		help="'test' evaluates on the held-out file; 'all' pools train+test, which is "
			 "what test_film.py does for the external cohort.")
	p.add_argument('--tag', type=str, default='',
		help="Config suffix printed by the training script, e.g. '__encdose-zero'. "
			 "Selects between checkpoints trained with different flags.")
	p.add_argument('--tau', type=float, default=1.0,
		help="Post-hoc scaling of the encoder's posterior standard deviation at "
			 "prediction time. tau>1 widens the predictive distribution without "
			 "retraining and WITHOUT moving the posterior mean, so the point "
			 "prediction (and hence RMSPE) is essentially unchanged. Fit it on "
			 "held-out cohorts, never on the ones you report.")
	p.add_argument('--no-json', action='store_true',
		help="Do not write a JSON. By default every run is recorded.")
	p.add_argument('--out-json', type=str, default=None,
		help="Write the metrics to this JSON file (for compare_revision_results.py).")
	p.add_argument('--label', type=str, default='dose-conditioned decoder',
		help="Row label used in the comparison table.")
	return p


def unscale_data(data, scaler_info):
	"""Reverse Box-Cox and max_out scaling -> physiological concentrations."""
	arr = data.detach().cpu().numpy() if torch.is_tensor(data) else np.asarray(data)
	if scaler_info is None:
		return arr
	max_out = scaler_info[0]
	best_lambda = scaler_info[1] if len(scaler_info) > 1 else None
	if best_lambda is not None:
		arr = inv_boxcox(arr, best_lambda)
		arr = np.nan_to_num(arr, nan=0.0)
	return arr * max_out


def calculate_auc(concentrations, times, static_treatments):
	"""Trapezoidal AUC, integrating to 12 h for Prograf and 24 h for Advagraf."""
	aucs = []
	for i in range(concentrations.shape[0]):
		is_prograf = bool(static_treatments[i, 1].item())
		cutoff = 12.0 if is_prograf else 24.0
		valid = times <= cutoff
		aucs.append(np.trapezoid(concentrations[i, valid], times[valid]))
	return np.array(aucs)


def main():
	cli = build_parser().parse_args()
	# Record every run by default: losing a result because a flag was omitted is
	# worse than writing a file nobody reads.
	if cli.out_json is None and not cli.no_json:
		cli.out_json = os.path.join('results', 'revision', str(cli.experiment),
			'dosecond' + cli.tag + '.json')
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	if cli.tau != 1.0:
		# widen every posterior draw by tau; the mean is untouched
		_orig_sample = utils.sample_standard_gaussian
		utils.sample_standard_gaussian = (
			lambda mu, sigma, _o=_orig_sample, _t=cli.tau: _o(mu, sigma * _t))
		print(f"post-hoc posterior scaling: tau = {cli.tau}")



	# ---------------- data ----------------
	load_id = cli.load if cli.load is not None else cli.experiment
	ckpt_path = cli.ckpt or os.path.join(
		cli.save, "exp_dosecond_run", str(cli.experiment),
		f"experiment_dosecond_{load_id}{cli.tag}{'_best' if cli.best else ''}.ckpt")
	# See test_film_matched.py: resolved early so the static width matches the ckpt.
	if os.path.exists(ckpt_path):
		_ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
		set_static_hematocrit(int(getattr(_ck.get('args'), 'static_dim', 3)) >= 4)

	train_csv = os.path.join(cli.data_dir, cli.experiment, "virtual_cohort_film_train.csv")
	test_csv = os.path.join(cli.data_dir, cli.experiment, "virtual_cohort_film_test.csv")
	# The scaler is always fitted on train+test jointly, as in test_film.py.
	_pin = None
	if cli.scale_from:
		_src = os.path.join(cli.data_dir, cli.scale_from, "virtual_cohort_film_train.csv")
		_pin = extract_gen_tac_film(file_path=[_src])[1]
		print(f"normalisation pinned to {cli.scale_from}: "
		      f"max_out={float(_pin[0]):.3f} lambda={float(_pin[1]):.6f}")
	data_all, scaler_info = extract_gen_tac_film(file_path=[train_csv, test_csv], scale=_pin)
	if cli.eval_split == 'test':
		train_ids = set(extract_gen_tac_film(file_path=[train_csv], scale=_pin)[0].keys())
		data_eval = {k: v for k, v in data_all.items() if k not in train_ids}
		if not data_eval:
			raise RuntimeError("No held-out patients found; use --eval-split all.")
	else:
		data_eval = data_all
	print(f"Evaluating on {len(data_eval)} patients ({cli.eval_split} split).")

	loader = DataLoader(TacroFilmDataset(data_eval), batch_size=cli.batch_size, shuffle=False,
		collate_fn=lambda x: collate_fn_tacro_film(x, device))

	# ---------------- model ----------------
	if not os.path.exists(ckpt_path):
		print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
		sys.exit(1)

	checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
	train_args = checkpoint['args']          # architecture exactly as trained
	obsrv_std = torch.Tensor([getattr(train_args, 'noise_weight', 0.01)]).to(device)
	z0_prior = torch.distributions.Normal(
		torch.Tensor([0.0]).to(device), torch.Tensor([1.0]).to(device))
	model = create_dose_conditioned_model(train_args, 1, z0_prior, obsrv_std, device)
	model.load_state_dict(checkpoint['state_dict'])
	model.to(device).eval()
	print(f"Loaded {ckpt_path} (cond_mode={model.cond_mode}, encoder_dose={model.encoder_dose}).")

	true_auc_v1, pred_auc_v1, true_auc_v2, pred_auc_v2 = [], [], [], []
	auc_draws_v2 = []
	pw_err_v1, pw_err_v2 = [], []   # squared relative errors above the LLOQ
	nrmse_v1, nrmse_v2 = [], []     # per-patient RMSE / mean concentration

	with torch.no_grad():
		for batch in loader:
			dense_tp = utils.linspace_vector(
				batch["tp_to_predict_v1"][0], torch.tensor(24.0), 100).to(device)

			pred_v2, info = model.get_reconstruction_counterfactual(
				data_v1=batch["observed_data_v1"],
				time_steps_v1=batch["observed_tp_v1"],
				dose_v1=batch["dose_v1"],
				dose_v2=batch["dose_v2"],
				tp_to_predict_v1=dense_tp,
				tp_to_predict_v2=dense_tp,
				static_v1=batch["static_v1"],
				n_traj_samples=cli.n_traj_samples)

			p_v1 = info["pred_x_v1"].mean(dim=0).squeeze(-1)
			p_v2 = pred_v2.mean(dim=0).squeeze(-1)

			p_v1 = unscale_data(p_v1, scaler_info)
			p_v2 = unscale_data(p_v2, scaler_info)

			# Prograf is a 12 h formulation: blank the second half of the grid.
			is_prograf = batch["static_v1"][:, 1].bool().cpu().numpy()
			p_v1[is_prograf, 50:] = 0.0
			p_v2[is_prograf, 50:] = 0.0

			dense_np = dense_tp.cpu().numpy()

			# The RMSPE above scores the posterior MEAN. Here we keep every draw so
			# the predictive DISTRIBUTION can be scored too -- the thing a
			# deterministic baseline cannot produce at all.
			s_v2 = unscale_data(pred_v2.squeeze(-1), scaler_info)      # [S, B, T]
			s_v2[:, is_prograf, 50:] = 0.0
			auc_draws_v2.append(np.stack([
				calculate_auc(s_v2[i], dense_np, batch["static_v1"])
				for i in range(s_v2.shape[0])]))

			# --- pointwise curve error, scored at the sparse TARGET times ---
			# The AUC integrates the window and therefore cancels shape errors: a peak
			# too high and a trough too low can still give the right exposure. This
			# scores the trajectory itself. Truth here is the noisy observation, so
			# there is an irreducible floor -- identical for every model compared.
			pw_v1, _i1 = model.get_reconstruction_counterfactual(
				data_v1=batch["observed_data_v1"], time_steps_v1=batch["observed_tp_v1"],
				dose_v1=batch["dose_v1"], dose_v2=batch["dose_v2"],
				tp_to_predict_v1=batch["tp_to_predict_v1"],
				tp_to_predict_v2=batch["tp_to_predict_v2"],
				static_v1=batch["static_v1"], n_traj_samples=cli.n_traj_samples)
			q_v1 = unscale_data(_i1["pred_x_v1"].mean(dim=0).squeeze(-1), scaler_info)
			q_v2 = unscale_data(pw_v1.mean(dim=0).squeeze(-1), scaler_info)
			y_v1 = unscale_data(batch["data_to_predict_v1"].squeeze(-1).clone(), scaler_info)
			y_v2 = unscale_data(batch["data_to_predict_v2"].squeeze(-1).clone(), scaler_info)
			# Relative error explodes at the trough: true concentrations reach
			# ~0.02 ng/mL, and dividing by those made a handful of points dominate
			# the whole metric (91% of the MSE came from t=0 and t=24). Gate at a
			# quantification limit, and also report a scale-free per-patient nRMSE.
			for acc, nacc, q, y in ((pw_err_v1, nrmse_v1, q_v1, y_v1),
									(pw_err_v2, nrmse_v2, q_v2, y_v2)):
				m = y >= POINTWISE_LLOQ
				if m.any():
					acc.append(((q[m] - y[m]) / y[m]) ** 2)
				nacc.append(np.sqrt(np.mean((q - y) ** 2, axis=1)) / np.mean(y, axis=1))

			true_auc_v1.extend((batch["auc_red_v1"] * scaler_info[0]).cpu().numpy())
			true_auc_v2.extend((batch["auc_red_v2"] * scaler_info[0]).cpu().numpy())
			pred_auc_v1.extend(calculate_auc(p_v1, dense_np, batch["static_v1"]))
			pred_auc_v2.extend(calculate_auc(p_v2, dense_np, batch["static_v1"]))

	pw_rmspe_v1 = float(np.sqrt(np.mean(np.concatenate(pw_err_v1))) * 100) if pw_err_v1 else None
	pw_rmspe_v2 = float(np.sqrt(np.mean(np.concatenate(pw_err_v2))) * 100) if pw_err_v2 else None
	nrmse_pct_v1 = float(np.mean(np.concatenate(nrmse_v1)) * 100) if nrmse_v1 else None
	nrmse_pct_v2 = float(np.mean(np.concatenate(nrmse_v2)) * 100) if nrmse_v2 else None
	auc_draws_v2 = np.concatenate(auc_draws_v2, axis=1) if auc_draws_v2 else None
	true_auc_v1 = np.array(true_auc_v1); pred_auc_v1 = np.array(pred_auc_v1)
	true_auc_v2 = np.array(true_auc_v2); pred_auc_v2 = np.array(pred_auc_v2)

	def mpe(t, p):   return float(np.mean((t - p) / t))
	def rmspe(t, p): return float(np.sqrt(np.mean(((t - p) / t) ** 2)))

	results = {
		"label": cli.label,
		"model": "dose_conditioned",
		"checkpoint": ckpt_path,
		"experiment": cli.experiment,
		"eval_split": cli.eval_split,
		"n_patients": int(len(true_auc_v1)),
		"cond_mode": model.cond_mode,
		"encoder_dose": model.encoder_dose,
		"v1": {"mpe_pct": mpe(true_auc_v1, pred_auc_v1) * 100,
			   "rmspe_pct": rmspe(true_auc_v1, pred_auc_v1) * 100,
			   "pointwise_rmspe_pct": pw_rmspe_v1,
			   "pointwise_lloq": POINTWISE_LLOQ, "nrmse_pct": nrmse_pct_v1},
		"v2": {"mpe_pct": mpe(true_auc_v2, pred_auc_v2) * 100,
			   "rmspe_pct": rmspe(true_auc_v2, pred_auc_v2) * 100,
			   "pointwise_rmspe_pct": pw_rmspe_v2,
			   "pointwise_lloq": POINTWISE_LLOQ, "nrmse_pct": nrmse_pct_v2},
		# kept so two runs can be compared with a PAIRED test rather than by
		# eyeballing two summary numbers computed on only n patients
		"tau": cli.tau,
		"calibration_v2": (calibration_report(auc_draws_v2, np.array(true_auc_v2))
						   if auc_draws_v2 is not None and auc_draws_v2.shape[0] > 1 else None),
		"per_patient": {
			"true_auc_v1": true_auc_v1.tolist(), "pred_auc_v1": pred_auc_v1.tolist(),
			"true_auc_v2": true_auc_v2.tolist(), "pred_auc_v2": pred_auc_v2.tolist(),
		},
	}

	print("\n" + "=" * 50)
	print("DOSE-CONDITIONED BASELINE -- RESULTS")
	print("=" * 50)
	print("VISIT 1 (reconstruction at the observed dose):")
	print(f"  - MPE (Bias)        : {results['v1']['mpe_pct']:.2f}%")
	print(f"  - RMSPE (Precision) : {results['v1']['rmspe_pct']:.2f}%   [on AUC]")
	print(f"  - RMSPE pointwise   : {results['v1']['pointwise_rmspe_pct']:.2f}%   [curve, >{POINTWISE_LLOQ} ng/mL]")
	print(f"  - nRMSE (scale-free): {results['v1']['nrmse_pct']:.2f}%   [RMSE / mean conc.]")
	print("-" * 50)
	print("VISIT 2 (zero-shot dose extrapolation from the same z0):")
	print(f"  - MPE (Bias)        : {results['v2']['mpe_pct']:.2f}%")
	print(f"  - RMSPE (Precision) : {results['v2']['rmspe_pct']:.2f}%   [on AUC]")
	print(f"  - RMSPE pointwise   : {results['v2']['pointwise_rmspe_pct']:.2f}%   [curve, >{POINTWISE_LLOQ} ng/mL]")
	print(f"  - nRMSE (scale-free): {results['v2']['nrmse_pct']:.2f}%   [RMSE / mean conc.]")
	print("=" * 50)

	cal = results.get("calibration_v2")
	if cal:
		print("PREDICTIVE CALIBRATION on the Visit 2 AUC "
			  f"({cal['n_samples']} posterior draws):")
		for k, iv in sorted(cal["intervals"].items()):
			print(f"  {int(iv['nominal']*100):>3d}% interval -> empirical coverage "
				  f"{iv['coverage']*100:5.1f}%   mean width {iv['mean_width_pct']:5.1f}% of true AUC")
		print(f"  calibration error {cal['calibration_error']:.3f} | CRPS {cal['crps_pct']:.2f}% "
			  f"| PIT mean {cal['pit_mean']:.3f} (0.5 ideal) std {cal['pit_std']:.3f} (0.289 ideal)")
		print("=" * 50)

	if cli.out_json:
		utils.makedirs(os.path.dirname(cli.out_json) or ".")
		with open(cli.out_json, "w") as f:
			json.dump(results, f, indent=2)
		print(f"Metrics written to {cli.out_json}")


if __name__ == '__main__':
	main()
