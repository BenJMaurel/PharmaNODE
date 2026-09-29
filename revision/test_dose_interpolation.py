###########################
# Dose-interpolation test
#
# The counterfactual RMSPE comparison only ever queries the models at doses that
# occur in training (the generator draws from 7 fixed levels, as PK trials
# usually do). This script asks a different question: what happens BETWEEN the
# training dose levels, and beyond them?
#
# Ground truth is exact. Scenario-2 PK is linear and the generator samples each
# patient's parameters once for both visits, so at steady state
#       AUC(d) = AUC_v1 * d / d_1
# holds to numerical precision (verified: max deviation 2.3e-06). No simulation
# needed -- every patient is their own analytic dose-response curve.
#
# Reported per dose category:
#   on-grid        the 7 training levels
#   interpolated   midpoints between them  -- the hypothesis under test
#   extrapolated   outside the training range
#
# The statistic that matters is the GAP between on-grid and interpolated error
# for each model, not the absolute level.
###########################

import os, sys, json, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

import lib.utils as utils
from lib.read_tacro import extract_gen_tac_film, TacroFilmDataset, collate_fn_tacro_film
from test_dose_cond import unscale_data, calculate_auc


def build_parser():
	p = argparse.ArgumentParser("Dose-interpolation test")
	p.add_argument('--experiment', type=str, required=True)
	p.add_argument('--model', type=str, required=True, choices=['film', 'dosecond', 'lupk'])
	p.add_argument('--ckpt', type=str, default=None)
	p.add_argument('--tag', type=str, default='')
	p.add_argument('--data-dir', type=str, default='./results/exp_film_run')
	p.add_argument('--save', type=str, default='./results/')
	p.add_argument('-b', '--batch-size', type=int, default=2000)
	p.add_argument('--n-traj-samples', type=int, default=50)
	p.add_argument('--eval-split', type=str, default='test', choices=['test', 'all'])
	p.add_argument('--out-json', type=str, default=None)
	p.add_argument('--label', type=str, default=None)
	p.add_argument('--truth-csv', type=str, default=None,
		help="dose_sweep_truth.csv from gen_tacro_bimodal.py: simulated AUC per "
			 "(patient, dose). Required whenever AUC is not proportional to dose "
			 "(scenario 3), where the analytic shortcut is invalid. If omitted, the "
			 "linear-PK shortcut AUC(d)=AUC_v1*d/d1 is used.")
	return p


def default_ckpt(cli):
	stem = {'film': ('exp_film_run', 'experiment_film'),
			'dosecond': ('exp_dosecond_run', 'experiment_dosecond'),
			'lupk': ('exp_lupk_run', 'experiment_lupk')}[cli.model]
	return os.path.join(cli.save, stem[0], str(cli.experiment),
		f"{stem[1]}_{cli.experiment}{cli.tag}_best.ckpt")


def load_model(cli, device):
	path = cli.ckpt or default_ckpt(cli)
	if not os.path.exists(path):
		print(f"Checkpoint not found: {path}", file=sys.stderr); sys.exit(1)
	if cli.model == 'film':
		from probe_entanglement import load_film_model
		m, _ = load_film_model(path, device)
		return m, path
	ck = torch.load(path, map_location=device, weights_only=False)
	if cli.model == 'dosecond':
		from lib.dose_conditioned import create_dose_conditioned_model
		z0 = torch.distributions.Normal(torch.Tensor([0.]).to(device), torch.Tensor([1.]).to(device))
		m = create_dose_conditioned_model(ck['args'], 1,
			z0, torch.Tensor([getattr(ck['args'], 'noise_weight', .01)]).to(device), device)
	else:
		from lib.lu_neural_pk import create_lu_pk_model
		m = create_lu_pk_model(ck['args'], device)
	m.load_state_dict(ck['state_dict']); m.to(device).eval()
	return m, path


def predict_auc_at_dose(model, kind, batch, dose_t, dense_tp, scaler_info, n_samples):
	"""Predicted AUC for every patient, if the target dose were `dose_t`."""
	B = batch["dose_v1"].size(0)
	dose_v2 = torch.full_like(batch["dose_v1"], float(dose_t))
	if kind == 'film':
		pred, _ = model.get_reconstruction_extrapolation(
			data_v1=batch["observed_data_v1"], time_steps_v1=batch["observed_tp_v1"],
			time_steps_v2=dense_tp, dose_v1=batch["dose_v1"], dose_v2=dose_v2,
			time_steps_to_predict_v1=dense_tp, static_v1=batch["static_v1"],
			delta_t=batch["delta_t"], t_v1=batch["t_v1"], n_traj_samples=n_samples)
		p = pred.mean(dim=0).squeeze(-1)
	elif kind == 'dosecond':
		pred, _ = model.get_reconstruction_counterfactual(
			data_v1=batch["observed_data_v1"], time_steps_v1=batch["observed_tp_v1"],
			dose_v1=batch["dose_v1"], dose_v2=dose_v2,
			tp_to_predict_v1=dense_tp, tp_to_predict_v2=dense_tp,
			static_v1=batch["static_v1"], n_traj_samples=n_samples)
		p = pred.mean(dim=0).squeeze(-1)
	else:
		_, pred, _ = model.predict_counterfactual(
			conc_v1=batch["observed_data_v1"], times_v1=batch["observed_tp_v1"],
			dose_v1=batch["dose_v1"], dose_v2=dose_v2,
			tp_v1=dense_tp, tp_v2=dense_tp, static=batch["static_v1"])
		p = pred.squeeze(-1)
	p = unscale_data(p, scaler_info)
	is_prograf = batch["static_v1"][:, 1].bool().cpu().numpy()
	p[is_prograf, 50:] = 0.0
	return calculate_auc(p, dense_tp.cpu().numpy(), batch["static_v1"])


def main():
	cli = build_parser().parse_args()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	tr = os.path.join(cli.data_dir, cli.experiment, "virtual_cohort_film_train.csv")
	te = os.path.join(cli.data_dir, cli.experiment, "virtual_cohort_film_test.csv")
	data_all, scaler_info = extract_gen_tac_film(file_path=[tr, te])
	if cli.eval_split == 'test':
		train_ids = set(extract_gen_tac_film(file_path=[tr])[0].keys())
		data_eval = {k: v for k, v in data_all.items() if k not in train_ids}
	else:
		data_eval = data_all
	loader = DataLoader(TacroFilmDataset(data_eval), batch_size=cli.batch_size, shuffle=False,
		collate_fn=lambda x: collate_fn_tacro_film(x, device))
	model, path = load_model(cli, device)

	truth_tbl, dose_max = None, None
	if cli.truth_csv:
		import pandas as pd
		tdf = pd.read_csv(cli.truth_csv)
		raw = pd.concat([pd.read_csv(tr), pd.read_csv(te)])
		dose_max = float(pd.to_numeric(raw['AMT'], errors='coerce').max())
		train_lvls = sorted(pd.to_numeric(pd.read_csv(tr)['AMT'], errors='coerce').dropna().unique())
		truth_tbl = {(int(r.ID), round(float(r.DOSE), 4)): float(r.AUC)
					 for r in tdf.itertuples()}
		sweep_mg = sorted(tdf['DOSE'].unique())
		# normalised dose is what the models consume
		grid = [round(d / dose_max, 6) for d in sweep_mg]
		lo, hi = min(train_lvls), max(train_lvls)
		ON   = [round(d / dose_max, 6) for d in sweep_mg if any(abs(d - t) < 1e-6 for t in train_lvls)]
		INTP = [round(d / dose_max, 6) for d in sweep_mg if lo < d < hi
				and not any(abs(d - t) < 1e-6 for t in train_lvls)]
		EXTP = [round(d / dose_max, 6) for d in sweep_mg if d < lo or d > hi]
		print(f"training dose levels: {train_lvls} mg  (normalised by {dose_max})")
		print(f"  on-grid {len(ON)} | interpolated {len(INTP)} | extrapolated {len(EXTP)}")
	else:
		ON   = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]      # training levels
		INTP = [0.45, 0.55, 0.65, 0.75, 0.85, 0.95]     # midpoints -- unseen
		EXTP = [0.25, 0.30, 0.35, 1.05, 1.10, 1.20]     # outside the range
		grid = sorted(ON + INTP + EXTP)

	err = {d: [] for d in grid}
	curves = []
	with torch.no_grad():
		for batch in loader:
			dense_tp = utils.linspace_vector(batch["tp_to_predict_v1"][0],
				torch.tensor(24.0), 100).to(device)
			d1 = batch["dose_v1"].cpu().numpy()
			auc1 = (batch["auc_red_v1"] * scaler_info[0]).cpu().numpy()
			per_dose = {}
			pids = [int(x) for x in batch["patient_ids"]]
			for d in grid:
				pred = predict_auc_at_dose(model, cli.model, batch, d, dense_tp,
					scaler_info, cli.n_traj_samples)
				if truth_tbl is not None:
					truth = np.array([truth_tbl[(pid, round(d * dose_max, 4))] for pid in pids])
				else:
					truth = auc1 * d / d1                  # exact only for linear PK
				err[d].append((pred - truth) / truth)
				per_dose[d] = pred
			curves.append(np.stack([per_dose[d] for d in grid]))   # [n_dose, B]
	curves = np.concatenate(curves, axis=1)
	for d in grid:
		err[d] = np.concatenate(err[d])

	def agg(ds):
		e = np.concatenate([err[d] for d in ds])
		return dict(rmspe_pct=float(np.sqrt(np.mean(e**2))*100), mpe_pct=float(np.mean(e)*100))
	res_on, res_in, res_ex = agg(ON), agg(INTP), agg(EXTP)

	# is the predicted dose-response monotone, and how close to proportional?
	mono = float(np.mean(np.all(np.diff(curves, axis=0) >= 0, axis=0)))
	gi = np.array(grid)
	slope_r2 = []
	for j in range(curves.shape[1]):
		y = curves[:, j]
		A = np.vstack([gi, np.ones_like(gi)]).T
		coef, res_, *_ = np.linalg.lstsq(A, y, rcond=None)
		ss = np.sum((y - y.mean())**2)
		slope_r2.append(1 - (res_[0]/ss if len(res_) and ss > 0 else 0.0))
	label = cli.label or {'film':'OT-FiLM','dosecond':'dose-conditioned','lupk':'Lu et al. (fair)'}[cli.model]

	out = dict(label=label, model=cli.model, experiment=cli.experiment, checkpoint=path,
		n_patients=int(len(err[grid[0]])), on_grid=res_on, interpolated=res_in,
		extrapolated=res_ex,
		interp_penalty_pct=res_in['rmspe_pct'] - res_on['rmspe_pct'],
		extrap_penalty_pct=res_ex['rmspe_pct'] - res_on['rmspe_pct'],
		monotone_frac=mono, linearity_r2=float(np.mean(slope_r2)),
		per_dose_rmspe={str(d): float(np.sqrt(np.mean(err[d]**2))*100) for d in grid})

	print(f"\n{'='*62}\n{label} -- dose-response quality ({out['n_patients']} patients)\n{'='*62}")
	print(f"  on-grid doses      RMSPE {res_on['rmspe_pct']:6.2f}%   MPE {res_on['mpe_pct']:+6.2f}%")
	print(f"  interpolated       RMSPE {res_in['rmspe_pct']:6.2f}%   MPE {res_in['mpe_pct']:+6.2f}%"
		  f"   -> penalty {out['interp_penalty_pct']:+.2f} pts")
	print(f"  extrapolated       RMSPE {res_ex['rmspe_pct']:6.2f}%   MPE {res_ex['mpe_pct']:+6.2f}%"
		  f"   -> penalty {out['extrap_penalty_pct']:+.2f} pts")
	print(f"  monotone in dose   {mono*100:.1f}% of patients")
	lin_note = "truth is exactly linear" if truth_tbl is None else "truth is NOT linear here"
	print(f"  linearity R^2      {out['linearity_r2']:.4f}   ({lin_note})")
	print('='*62)

	if cli.out_json is None:
		cli.out_json = os.path.join('results','revision',str(cli.experiment),
			f'doseinterp_{cli.model}{cli.tag}.json')
	out['truth'] = 'simulated' if truth_tbl is not None else 'linear-analytic'
	utils.makedirs(os.path.dirname(cli.out_json) or '.')
	json.dump(out, open(cli.out_json,'w'), indent=2)
	print(f"written to {cli.out_json}")


if __name__ == '__main__':
	main()
