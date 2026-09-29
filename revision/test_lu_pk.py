###########################
# Priority 4 -- head-to-head against Lu et al.'s neural-PK submodule: EVALUATION
#
# Same AUC protocol as test_film.py and test_dose_cond.py (dense 100-point 24 h
# grid, inverse Box-Cox, 12 h cut-off for Prograf, identical MPE/RMSPE
# definitions), so the result is a row-for-row addition to the existing table.
#
# The model is deterministic, so there is no posterior sampling to average over.
###########################

import os
import sys
import json
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

import lib.utils as utils
from lib.read_tacro import (extract_gen_tac_film, TacroFilmDataset, collate_fn_tacro_film,
                            set_static_hematocrit)
from lib.lu_neural_pk import create_lu_pk_model
from test_dose_cond import unscale_data, calculate_auc


def build_parser():
	p = argparse.ArgumentParser("Lu et al. neural-PK baseline -- testing")
	p.add_argument('--experiment', type=str, required=True)
	p.add_argument('--load', type=str, default=None)
	p.add_argument('--ckpt', type=str, default=None)
	p.add_argument('--data-dir', type=str, default='./results/exp_film_run')
	p.add_argument('--save', type=str, default='./results/')
	p.add_argument('--best', action='store_true', default=True)
	p.add_argument('--last', dest='best', action='store_false')
	p.add_argument('-b', '--batch-size', type=int, default=2000)
	p.add_argument('--eval-split', type=str, default='test', choices=['test', 'all'])
	p.add_argument('--tag', type=str, default='',
		help="Config suffix printed by the training script, e.g. '__mode-counterfactual'.")
	p.add_argument('--no-json', action='store_true',
		help="Do not write a JSON. By default every run is recorded.")
	p.add_argument('--out-json', type=str, default=None)
	p.add_argument('--label', type=str, default='Lu et al. neural-PK')
	p.add_argument('--scale-from', type=str, default=None,
		help="Pin the normalisation (max_out, Box-Cox lambda, dose max) to this cohort's TRAINING "
			 "csv, i.e. the cohort the model was trained on. Without it the scaler is refit on the "
			 "evaluated cohort and cross-cohort numbers are invalid (handoff landmine 1.2).")
	return p


def main():
	cli = build_parser().parse_args()
	# Record every run by default: losing a result because a flag was omitted is
	# worse than writing a file nobody reads.
	if cli.out_json is None and not cli.no_json:
		cli.out_json = os.path.join('results', 'revision', str(cli.experiment),
			'lupk' + cli.tag + '.json')
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	load_id = cli.load if cli.load is not None else cli.experiment
	ckpt_path = cli.ckpt or os.path.join(
		cli.save, "exp_lupk_run", str(cli.experiment),
		f"experiment_lupk_{load_id}{cli.tag}{'_best' if cli.best else ''}.ckpt")
	# As in test_dose_cond.py: resolved before the data so the static width matches the ckpt.
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
		sys.exit(1)

	checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
	train_args = checkpoint['args']
	model = create_lu_pk_model(train_args, device)
	model.load_state_dict(checkpoint['state_dict'])
	model.to(device).eval()
	print(f"Loaded {ckpt_path} (train_mode={getattr(train_args, 'train_mode', '?')}, "
		  f"use_static={getattr(train_args, 'use_static', False)}).")

	true_auc_v1, pred_auc_v1, true_auc_v2, pred_auc_v2 = [], [], [], []
	nrmse_v1, nrmse_v2 = [], []     # per-patient RMSE / mean concentration, at the target times

	with torch.no_grad():
		for batch in loader:
			dense_tp = utils.linspace_vector(
				batch["tp_to_predict_v1"][0], torch.tensor(24.0), 100).to(device)

			p_v1, p_v2, _ = model.predict_counterfactual(
				conc_v1=batch["observed_data_v1"],
				times_v1=batch["observed_tp_v1"],
				dose_v1=batch["dose_v1"],
				dose_v2=batch["dose_v2"],
				tp_v1=dense_tp, tp_v2=dense_tp,
				static=batch["static_v1"])

			p_v1 = unscale_data(p_v1.squeeze(-1), scaler_info)
			p_v2 = unscale_data(p_v2.squeeze(-1), scaler_info)

			is_prograf = batch["static_v1"][:, 1].bool().cpu().numpy()
			p_v1[is_prograf, 50:] = 0.0
			p_v2[is_prograf, 50:] = 0.0

			dense_np = dense_tp.cpu().numpy()

			# curve error at the 12 target times, same definition as test_dose_cond.py
			# (truth = the noisy observation; scale-free per-patient nRMSE)
			q_v1, q_v2, _ = model.predict_counterfactual(
				conc_v1=batch["observed_data_v1"], times_v1=batch["observed_tp_v1"],
				dose_v1=batch["dose_v1"], dose_v2=batch["dose_v2"],
				tp_v1=batch["tp_to_predict_v1"], tp_v2=batch["tp_to_predict_v2"],
				static=batch["static_v1"])
			q_v1 = unscale_data(q_v1.squeeze(-1), scaler_info)
			q_v2 = unscale_data(q_v2.squeeze(-1), scaler_info)
			y_v1 = unscale_data(batch["data_to_predict_v1"].squeeze(-1).clone(), scaler_info)
			y_v2 = unscale_data(batch["data_to_predict_v2"].squeeze(-1).clone(), scaler_info)
			for nacc, q, y in ((nrmse_v1, q_v1, y_v1), (nrmse_v2, q_v2, y_v2)):
				nacc.append(np.sqrt(np.mean((q - y) ** 2, axis=1)) / np.mean(y, axis=1))

			true_auc_v1.extend((batch["auc_red_v1"] * scaler_info[0]).cpu().numpy())
			true_auc_v2.extend((batch["auc_red_v2"] * scaler_info[0]).cpu().numpy())
			pred_auc_v1.extend(calculate_auc(p_v1, dense_np, batch["static_v1"]))
			pred_auc_v2.extend(calculate_auc(p_v2, dense_np, batch["static_v1"]))

	nrmse_v1 = np.concatenate(nrmse_v1) if nrmse_v1 else np.array([])
	nrmse_v2 = np.concatenate(nrmse_v2) if nrmse_v2 else np.array([])
	true_auc_v1 = np.array(true_auc_v1); pred_auc_v1 = np.array(pred_auc_v1)
	true_auc_v2 = np.array(true_auc_v2); pred_auc_v2 = np.array(pred_auc_v2)

	def mpe(t, p):   return float(np.mean((t - p) / t))
	def rmspe(t, p): return float(np.sqrt(np.mean(((t - p) / t) ** 2)))

	results = {
		"label": cli.label,
		"model": "lu_neural_pk",
		"checkpoint": ckpt_path,
		"experiment": cli.experiment,
		"eval_split": cli.eval_split,
		"n_patients": int(len(true_auc_v1)),
		"train_mode": getattr(train_args, 'train_mode', None),
		"use_static": bool(getattr(train_args, 'use_static', False)),
		"init": getattr(train_args, 'init', 'zero'),
		"scale_from": cli.scale_from,
		"v1": {"mpe_pct": mpe(true_auc_v1, pred_auc_v1) * 100,
			   "rmspe_pct": rmspe(true_auc_v1, pred_auc_v1) * 100,
			   "nrmse_pct": float(np.mean(nrmse_v1) * 100) if nrmse_v1.size else None},
		"v2": {"mpe_pct": mpe(true_auc_v2, pred_auc_v2) * 100,
			   "rmspe_pct": rmspe(true_auc_v2, pred_auc_v2) * 100,
			   "nrmse_pct": float(np.mean(nrmse_v2) * 100) if nrmse_v2.size else None},
		# kept so two runs can be compared with a PAIRED test rather than by
		# eyeballing two summary numbers computed on only n patients
		"per_patient": {
			"true_auc_v1": true_auc_v1.tolist(), "pred_auc_v1": pred_auc_v1.tolist(),
			"true_auc_v2": true_auc_v2.tolist(), "pred_auc_v2": pred_auc_v2.tolist(),
			"nrmse_v1": nrmse_v1.tolist(), "nrmse_v2": nrmse_v2.tolist(),
		},
	}

	print("\n" + "=" * 50)
	print("LU ET AL. NEURAL-PK BASELINE -- RESULTS")
	print("=" * 50)
	print("VISIT 1 (reconstruction at the observed dose):")
	print(f"  - MPE (Bias)        : {results['v1']['mpe_pct']:.2f}%")
	print(f"  - RMSPE (Precision) : {results['v1']['rmspe_pct']:.2f}%")
	print("-" * 50)
	print("VISIT 2 (two-port counterfactual: encoder frozen on V1, forcing swapped to d2):")
	print(f"  - MPE (Bias)        : {results['v2']['mpe_pct']:.2f}%")
	print(f"  - RMSPE (Precision) : {results['v2']['rmspe_pct']:.2f}%")
	print("=" * 50)

	if cli.out_json:
		utils.makedirs(os.path.dirname(cli.out_json) or ".")
		with open(cli.out_json, "w") as f:
			json.dump(results, f, indent=2)
		print(f"Metrics written to {cli.out_json}")


if __name__ == '__main__':
	main()
