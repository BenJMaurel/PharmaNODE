###########################
# OT-FiLM evaluation, matched to the revision baselines
#
# Why this exists alongside test_film.py:
#
#   * test_film.py pools train+test (it passes BOTH csv files to
#     extract_gen_tac_film and evaluates on the whole dict), so its RMSPE
#     includes the training patients. Comparing that number against a baseline
#     scored on held-out patients only would flatter the FiLM row. This script
#     takes the same --eval-split flag as test_dose_cond.py / test_lu_pk.py, so
#     every row of the table is scored on the same patients.
#   * create_latent_ode_model() never forwards a film_time flag, so a checkpoint
#     trained with the time-conditioned FiLM MLPs loads with those weights
#     silently dropped. This script re-attaches them from the checkpoint.
#
# The AUC arithmetic itself is copied from test_film.py unchanged, so on
# --eval-split all it reproduces that script's numbers.
###########################

import os
import sys
import json
import argparse

import numpy as np

# Quantification limit for the pointwise relative-error metric: below it the
# denominator is small enough that a few trough points dominate the statistic.
POINTWISE_LLOQ = 1.0
import torch
from torch.utils.data import DataLoader

import lib.utils as utils
from lib.read_tacro import (extract_gen_tac_film, TacroFilmDataset,
                            collate_fn_tacro_film, set_static_hematocrit)
from lib.calibration import calibration_report
from test_dose_cond import unscale_data, calculate_auc
from probe_entanglement import load_film_model


def build_parser():
	p = argparse.ArgumentParser("OT-FiLM evaluation matched to the revision baselines")
	p.add_argument('--experiment', type=str, required=True, help="Dataset folder name.")
	p.add_argument('--load', type=str, default=None,
		help="Checkpoint experiment ID (defaults to --experiment).")
	p.add_argument('--ckpt', type=str, default=None, help="Explicit path to a .ckpt.")
	p.add_argument('--tag', type=str, default=None,
		help="Config suffix of the checkpoint, e.g. '__noz0_sig0.217_sc7.5'. Omit to "
		     "take the untagged (all-default) file; the error message lists what is "
		     "actually present if the guess is wrong.")
	p.add_argument('--data-dir', type=str, default='./results/exp_film_run')
	p.add_argument('--save', type=str, default='./results/')
	p.add_argument('--best', action='store_true', default=True)
	p.add_argument('--last', dest='best', action='store_false')
	p.add_argument('-b', '--batch-size', type=int, default=2000)
	p.add_argument('--n-traj-samples', type=int, default=100)
	p.add_argument('--scale-from', type=str, default=None,
		help="Experiment whose TRAIN csv defines the Box-Cox/max normalisation, instead "
		     "of refitting it on the evaluated cohort. Required for cross-cohort "
		     "evaluation: a model trained elsewhere expects its own transform, and "
		     "refitting silently feeds it differently-scaled inputs. Default None "
		     "reproduces previous behaviour exactly.")
	p.add_argument('--eval-split', type=str, default='test', choices=['test', 'all'],
		help="'test' = held-out patients only. 'all' pools train+test, reproducing "
			 "what test_film.py does.")
	p.add_argument('--tau', type=float, default=1.0,
		help="Post-hoc scaling of the encoder's posterior standard deviation at "
			 "prediction time. tau>1 widens the predictive distribution without "
			 "retraining and WITHOUT moving the posterior mean, so the point "
			 "prediction (and hence RMSPE) is essentially unchanged. Fit it on "
			 "held-out cohorts, never on the ones you report.")
	p.add_argument('--no-json', action='store_true',
		help="Do not write a JSON. By default every run is recorded.")
	p.add_argument('--out-json', type=str, default=None)
	p.add_argument('--label', type=str, default='Full OT-FiLM (ours)')
	p.add_argument('--transport', type=str, default='film', choices=['film', 'none', 'oracle'],
		help="Ablation on the extrapolation step. 'film' is the real model. 'none' skips "
			 "the FiLM/OT transport entirely and decodes the Visit 1 state at the Visit 2 "
			 "dose -- the identity-transport control, which measures what the transport "
			 "machinery actually buys. 'oracle' encodes the true Visit 2 observations and "
			 "decodes those: not a counterfactual at all, but an upper bound that separates "
			 "decoder error from transport error.")
	return p


def main():
	cli = build_parser().parse_args()
	# Record every run by default: losing a result because a flag was omitted is
	# worse than writing a file nobody reads.
	if cli.out_json is None and not cli.no_json:
		cli.out_json = os.path.join('results', 'revision', str(cli.experiment),
			'film' + ('' if cli.transport == 'film' else '_transport-' + cli.transport) + '.json')
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	if cli.tau != 1.0:
		# widen every posterior draw by tau; the mean is untouched
		_orig_sample = utils.sample_standard_gaussian
		utils.sample_standard_gaussian = (
			lambda mu, sigma, _o=_orig_sample, _t=cli.tau: _o(mu, sigma * _t))
		print(f"post-hoc posterior scaling: tau = {cli.tau}")



	load_id = cli.load if cli.load is not None else cli.experiment
	ckpt_path = cli.ckpt or os.path.join(
		cli.save, "exp_film_run", str(cli.experiment),
		f"experiment_film_{load_id}{cli.tag or ''}{'_best' if cli.best else ''}.ckpt")
	# Resolved before the data is built: the static width must match what this
	# checkpoint was trained with, otherwise the encoder gets the wrong covariates.
	if os.path.exists(ckpt_path):
		_ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
		set_static_hematocrit(int(getattr(_ck.get('args'), 'static_dim', 3)) >= 4)

	train_csv = os.path.join(cli.data_dir, cli.experiment, "virtual_cohort_film_train.csv")
	test_csv = os.path.join(cli.data_dir, cli.experiment, "virtual_cohort_film_test.csv")
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

	if not os.path.exists(ckpt_path):
		print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
		d = os.path.dirname(ckpt_path)
		if os.path.isdir(d):
			found = sorted(f for f in os.listdir(d) if f.endswith('.ckpt'))
			if found:
				print("  present in that directory:", file=sys.stderr)
				for f in found:
					print(f"    {f}", file=sys.stderr)
				print("  pass --tag with the suffix between the id and '_best', "
				      "or --ckpt with the full path.", file=sys.stderr)
		sys.exit(1)

	model, train_args = load_film_model(ckpt_path, device)
	print(f"Loaded {ckpt_path} (film_time={getattr(model, 'film_time', False)}).")

	true_auc_v1, pred_auc_v1, true_auc_v2, pred_auc_v2 = [], [], [], []
	pw_err_v1, pw_err_v2 = [], []   # squared relative errors above the LLOQ
	nrmse_v1, nrmse_v2 = [], []     # per-patient RMSE / mean concentration
	auc_draws_v2 = []   # per-posterior-draw AUCs, for the calibration report

	with torch.no_grad():
		for batch in loader:
			dense_tp = utils.linspace_vector(
				batch["tp_to_predict_v1"][0], torch.tensor(24.0), 100).to(device)

			if cli.transport == 'film':
				pred_v2, info = model.get_reconstruction_extrapolation(
					data_v1=batch["observed_data_v1"],
					time_steps_v1=batch["observed_tp_v1"],
					time_steps_v2=dense_tp,
					dose_v1=batch["dose_v1"],
					dose_v2=batch["dose_v2"],
					time_steps_to_predict_v1=dense_tp,
					static_v1=batch["static_v1"],
					delta_t=batch["delta_t"],
					t_v1=batch["t_v1"],
					n_traj_samples=cli.n_traj_samples)
				pred_v1 = info["pred_x_v1"]
			else:
				# Encode, then decode WITHOUT the transport step.
				#   'none'   -> the Visit 1 posterior, i.e. identity transport
				#   'oracle' -> the true Visit 2 posterior (not a counterfactual)
				src = 'v1' if cli.transport == 'none' else 'v2'
				data_e = batch[f"observed_data_{src}"]
				tp_e = batch[f"observed_tp_{src}"]
				dose_e = batch[f"dose_{src}"]
				static_e = batch.get(f"static_{src}", batch["static_v1"])
				seq_len = data_e.size(1)
				dose_ch = dose_e.view(-1, 1, 1).expand(-1, seq_len, 1)
				mu, std = model.encoder_z0(torch.cat((data_e, dose_ch), -1), tp_e,
					static=static_e, run_backwards=True)
				z0 = utils.sample_standard_gaussian(
					mu.repeat(cli.n_traj_samples, 1, 1),
					std.abs().repeat(cli.n_traj_samples, 1, 1))
				pred_v2 = model.decoder(model.diffeq_solver(z0, dense_tp))

				mu1, std1 = model.encoder_z0(
					torch.cat((batch["observed_data_v1"],
						batch["dose_v1"].view(-1, 1, 1).expand(-1, batch["observed_data_v1"].size(1), 1)), -1),
					batch["observed_tp_v1"], static=batch["static_v1"], run_backwards=True)
				z0_1 = utils.sample_standard_gaussian(
					mu1.repeat(cli.n_traj_samples, 1, 1),
					std1.abs().repeat(cli.n_traj_samples, 1, 1))
				pred_v1 = model.decoder(model.diffeq_solver(z0_1, dense_tp))

			p_v1 = pred_v1.mean(dim=0).squeeze(-1)
			p_v2 = pred_v2.mean(dim=0).squeeze(-1)

			p_v1 = unscale_data(p_v1, scaler_info)
			p_v2 = unscale_data(p_v2, scaler_info)

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
			# AUC integrates the window and cancels shape errors; this scores the
			# trajectory. Truth is the noisy observation, so there is a floor that is
			# identical across models.
			if cli.transport == 'film':
				pw2, pwi = model.get_reconstruction_extrapolation(
					data_v1=batch["observed_data_v1"], time_steps_v1=batch["observed_tp_v1"],
					time_steps_v2=batch["tp_to_predict_v2"],
					dose_v1=batch["dose_v1"], dose_v2=batch["dose_v2"],
					time_steps_to_predict_v1=batch["tp_to_predict_v1"],
					static_v1=batch["static_v1"], delta_t=batch["delta_t"],
					t_v1=batch["t_v1"], n_traj_samples=cli.n_traj_samples)
				q_v1 = unscale_data(pwi["pred_x_v1"].mean(dim=0).squeeze(-1), scaler_info)
				q_v2 = unscale_data(pw2.mean(dim=0).squeeze(-1), scaler_info)
				y_v1 = unscale_data(batch["data_to_predict_v1"].squeeze(-1).clone(), scaler_info)
				y_v2 = unscale_data(batch["data_to_predict_v2"].squeeze(-1).clone(), scaler_info)
				# Relative error explodes at the trough (true conc. reaches ~0.02
				# ng/mL); gate at a quantification limit and also report a
				# scale-free per-patient nRMSE.
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

	auc_draws_v2 = np.concatenate(auc_draws_v2, axis=1) if auc_draws_v2 else None
	pw_rmspe_v1 = float(np.sqrt(np.mean(np.concatenate(pw_err_v1))) * 100) if pw_err_v1 else None
	pw_rmspe_v2 = float(np.sqrt(np.mean(np.concatenate(pw_err_v2))) * 100) if pw_err_v2 else None
	nrmse_pct_v1 = float(np.mean(np.concatenate(nrmse_v1)) * 100) if nrmse_v1 else None
	nrmse_pct_v2 = float(np.mean(np.concatenate(nrmse_v2)) * 100) if nrmse_v2 else None
	true_auc_v1 = np.array(true_auc_v1); pred_auc_v1 = np.array(pred_auc_v1)
	true_auc_v2 = np.array(true_auc_v2); pred_auc_v2 = np.array(pred_auc_v2)

	def mpe(t, p):   return float(np.mean((t - p) / t))
	def rmspe(t, p): return float(np.sqrt(np.mean(((t - p) / t) ** 2)))

	results = {
		"label": cli.label,
		"model": "ot_film",
		"checkpoint": ckpt_path,
		"experiment": cli.experiment,
		"eval_split": cli.eval_split,
		"n_patients": int(len(true_auc_v1)),
		"film_time": bool(getattr(model, 'film_time', False)),
		"transport": cli.transport,
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
	print(f"OT-FiLM (MATCHED PROTOCOL, transport={cli.transport}) -- RESULTS")
	print("=" * 50)
	print("VISIT 1 (base reconstruction context):")
	print(f"  - MPE (Bias)        : {results['v1']['mpe_pct']:.2f}%")
	print(f"  - RMSPE (Precision) : {results['v1']['rmspe_pct']:.2f}%   [on AUC]")
	if results['v1'].get('pointwise_rmspe_pct') is not None:
		print(f"  - RMSPE pointwise   : {results['v1']['pointwise_rmspe_pct']:.2f}%   [curve, >{POINTWISE_LLOQ} ng/mL]")
		print(f"  - nRMSE (scale-free): {results['v1']['nrmse_pct']:.2f}%   [RMSE / mean conc.]")
	print("-" * 50)
	print("VISIT 2 (zero-shot dose extrapolation):")
	print(f"  - MPE (Bias)        : {results['v2']['mpe_pct']:.2f}%")
	print(f"  - RMSPE (Precision) : {results['v2']['rmspe_pct']:.2f}%   [on AUC]")
	if results['v2'].get('pointwise_rmspe_pct') is not None:
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
