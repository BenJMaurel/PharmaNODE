###########################
# Priority 3 -- quantitative entanglement diagnostic
#
# Trains lightweight probes (ridge / shallow MLP / logistic) to predict the
# administered dose from a latent representation, on held-out patients.
#
#   z_base   = mu( q(z0 | O_1, s_i) )        encoded from Visit 1
#   z_new    = FiLM/OT transport of z_base towards the Visit 2 dose
#   z_base2  = mu( q(z0 | O_2, s_i) )        encoded from Visit 2 (reference)
#   raw      = the sparse Visit 1 observations themselves (upper-bound control)
#
# Claim B is supported if d_1 is highly predictable from z_base (entanglement),
# while after transport d_2 becomes predictable and d_1 does not.
#
# Purely additive: no existing file is imported for anything but reading.
###########################

import os
import sys
import json
import argparse

import numpy as np
import torch
import torch.nn as nn

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import r2_score, balanced_accuracy_score
from scipy.stats import spearmanr

import lib.utils as utils
from lib.read_tacro import extract_gen_tac_film, TacroFilmDataset, collate_fn_tacro_film
from torch.utils.data import DataLoader


# --------------------------------------------------------------------------- #
#  Model loading
# --------------------------------------------------------------------------- #

def _priors(device, noise_weight):
	obsrv_std = torch.Tensor([noise_weight]).to(device)
	z0_prior = torch.distributions.Normal(
		torch.Tensor([0.0]).to(device), torch.Tensor([1.0]).to(device))
	return obsrv_std, z0_prior


def load_film_model(ckpt_path, device):
	"""Rebuild the OT-FiLM model exactly as it was trained.

	`lib.create_latent_ode_model` does not forward the `film_time` flag, so the
	time-conditioned FiLM MLPs are re-attached here from the checkpoint when the
	model was trained with --use_time. Nothing in lib/ is modified.
	"""
	from lib.create_latent_ode_model import create_LatentODE_model

	ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
	args = ckpt['args']
	args.use_film = True
	state_dict = ckpt['state_dict']

	obsrv_std, z0_prior = _priors(device, getattr(args, 'noise_weight', 0.01))
	model = create_LatentODE_model(args, 1, z0_prior, obsrv_std, device)

	if any(k.startswith('film_gamma_t') for k in state_dict) and not hasattr(model, 'film_gamma_t'):
		n_in = state_dict['film_gamma_t.0.weight'].shape[1]
		n_hid = state_dict['film_gamma_t.0.weight'].shape[0]
		n_out = state_dict['film_gamma_t.2.weight'].shape[0]
		make = lambda: nn.Sequential(nn.Linear(n_in, n_hid), nn.ReLU(), nn.Linear(n_hid, n_out)).to(device)
		model.film_gamma_t = make()
		model.film_beta_t = make()
		model.film_time = True

	model.film_no_z0_cond = bool(getattr(args, 'film_no_z0_cond', False))
	model.film_residual_dose = bool(getattr(args, 'film_residual_dose', False))
	model.encoder_dose_zero = bool(getattr(args, 'encoder_dose_zero', False))
	missing, unexpected = model.load_state_dict(state_dict, strict=False)
	if missing:
		print(f"[warn] checkpoint is missing {len(missing)} keys, e.g. {missing[:4]}", file=sys.stderr)
	if unexpected:
		print(f"[warn] checkpoint has {len(unexpected)} unused keys, e.g. {unexpected[:4]}", file=sys.stderr)
	model.to(device).eval()
	return model, args


def load_dosecond_model(ckpt_path, device):
	from lib.dose_conditioned import create_dose_conditioned_model
	ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
	args = ckpt['args']
	obsrv_std, z0_prior = _priors(device, getattr(args, 'noise_weight', 0.01))
	model = create_dose_conditioned_model(args, 1, z0_prior, obsrv_std, device)
	model.load_state_dict(ckpt['state_dict'])
	model.to(device).eval()
	return model, args


# --------------------------------------------------------------------------- #
#  Representation extraction
# --------------------------------------------------------------------------- #

def encode_mu(model, data, tp, dose, static):
	"""Posterior mean of q(z0 | O, s). Returns [B, latent_dim]."""
	if hasattr(model, '_encoder_input'):
		truth_w_dose, static = model._encoder_input(data, dose, static)
	else:
		seq_len = data.size(1)
		truth_w_dose = torch.cat((data, dose.view(-1,1,1).expand(-1, seq_len, 1)), dim=-1)
	mu, _ = model.encoder_z0(truth_w_dose, tp, static=static, run_backwards=True)
	return mu.squeeze(0)


def film_transport(model, z_base, dose_v1, dose_v2, delta_t, t_v1):
	"""Reproduces the transport in LatentODE.get_reconstruction_extrapolation."""
	z0 = z_base.unsqueeze(0)                                    # [1, B, L]
	c_doses = torch.stack([dose_v1, dose_v2], dim=1).unsqueeze(0)   # [1, B, 2]
	# Use the MODEL's own context/map builders rather than reimplementing them here:
	# a local copy silently ignores film_no_z0_cond and film_residual_dose, and then
	# reports transports the trained model would never produce.
	if hasattr(model, '_film_context'):
		c = model._film_context(c_doses, z0.detach())
	else:
		c = torch.cat([c_doses, z0.detach()], dim=-1)

	z0_mod = z0
	if getattr(model, 'film_time', False):
		delta = torch.stack([delta_t.squeeze(-1), t_v1.squeeze(-1)], dim=1).unsqueeze(0)
		c_t = (model._film_context(delta, z0.detach()) if hasattr(model, '_film_context')
			   else torch.cat([delta, z0.detach()], dim=-1))
		z0_mod = z0 * model.film_gamma_t(c_t) + model.film_beta_t(c_t)

	if hasattr(model, '_film_maps'):
		gamma, beta = model._film_maps(c, dose_v1, dose_v2, 1)
	else:
		gamma, beta = model.film_gamma(c), model.film_beta(c)
	z_new = z0_mod * gamma + beta
	return z_new.squeeze(0)


def collect_representations(model, loader, kind, device):
	reps = {}
	targets = {}
	with torch.no_grad():
		for batch in loader:
			d1, d2 = batch["dose_v1"], batch["dose_v2"]
			z1 = encode_mu(model, batch["observed_data_v1"], batch["observed_tp_v1"],
				d1, batch["static_v1"])
			z2 = encode_mu(model, batch["observed_data_v2"], batch["observed_tp_v2"],
				d2, batch.get("static_v2", batch["static_v1"]))
			raw = batch["observed_data_v1"].reshape(batch["observed_data_v1"].size(0), -1)

			blocks = {"z_v1": z1, "z_v2": z2, "raw_v1": raw}
			if kind == "film":
				blocks["z_new"] = film_transport(
					model, z1, d1, d2, batch["delta_t"], batch["t_v1"])

			for k, v in blocks.items():
				reps.setdefault(k, []).append(v.detach().cpu().numpy())
			targets.setdefault("d1", []).append(d1.detach().cpu().numpy())
			targets.setdefault("d2", []).append(d2.detach().cpu().numpy())

	reps = {k: np.concatenate(v, axis=0) for k, v in reps.items()}
	targets = {k: np.concatenate(v, axis=0).ravel() for k, v in targets.items()}
	return reps, targets


PRETTY = {
	"z_v1": "z_base  (Visit 1 posterior mean)",
	"z_v2": "z_base  (Visit 2 posterior mean)",
	"raw_v1": "raw Visit 1 observations (control)",
	"z_new": "z_new   (after FiLM/OT transport to d2)",
}


# --------------------------------------------------------------------------- #
#  Probes
# --------------------------------------------------------------------------- #

def regression_probe(X, y, n_splits=5, seed=0):
	"""Cross-validated R^2 of a linear and a shallow-MLP probe, plus a
	label-shuffled control that gives the empirical chance level."""
	kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

	def cv_r2(target):
		lin = Pipeline([("sc", StandardScaler()),
						("m", RidgeCV(alphas=np.logspace(-3, 3, 13)))])
		mlp = Pipeline([("sc", StandardScaler()),
						("m", MLPRegressor(hidden_layer_sizes=(64,), max_iter=3000,
							random_state=seed, early_stopping=False))])
		p_lin = cross_val_predict(lin, X, target, cv=kf)
		p_mlp = cross_val_predict(mlp, X, target, cv=kf)
		return p_lin, p_mlp

	p_lin, p_mlp = cv_r2(y)
	rng = np.random.RandomState(seed)
	y_shuf = y[rng.permutation(len(y))]
	p_lin_s, _ = cv_r2(y_shuf)

	lin_r2 = float(r2_score(y, p_lin))
	mlp_r2 = float(r2_score(y, p_mlp))
	best = p_lin if lin_r2 >= mlp_r2 else p_mlp
	rho = spearmanr(best, y).correlation
	return {
		"linear_r2": lin_r2,
		"mlp_r2": mlp_r2,
		# headline number: the probe is an upper bound on the dose information that
		# is linearly *or* non-linearly decodable, so take the better of the two.
		"best_r2": max(lin_r2, mlp_r2),
		"shuffled_linear_r2": float(r2_score(y_shuf, p_lin_s)),
		"best_spearman": float(rho) if rho == rho else float("nan"),
	}


def classification_probe(X, y, n_bins=4, n_splits=5, seed=0):
	"""Balanced accuracy of a logistic probe on dose bins; chance = 1/n_classes."""
	uniq = np.unique(y)
	if len(uniq) <= n_bins * 2:
		classes = np.searchsorted(uniq, y)
	else:
		edges = np.quantile(y, np.linspace(0, 1, n_bins + 1)[1:-1])
		classes = np.digitize(y, edges)
	n_classes = len(np.unique(classes))
	if n_classes < 2:
		return {"balanced_accuracy": float("nan"), "chance": float("nan"), "n_classes": n_classes}

	counts = np.bincount(classes)
	n_splits = int(min(n_splits, counts[counts > 0].min()))
	if n_splits < 2:
		return {"balanced_accuracy": float("nan"), "chance": 1.0 / n_classes, "n_classes": n_classes}

	skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
	clf = Pipeline([("sc", StandardScaler()),
					("m", LogisticRegression(max_iter=5000))])
	pred = cross_val_predict(clf, X, classes, cv=skf)
	return {
		"balanced_accuracy": float(balanced_accuracy_score(classes, pred)),
		"chance": 1.0 / n_classes,
		"n_classes": int(n_classes),
	}


def transfer_probe(z_v1, z_v2, z_new, d1, d2, n_splits=5, seed=0):
	"""Where does the transported state sit in *dose* coordinates?

	A probe that predicts the dose is fitted on natural encodings only (the
	posterior means of both visits) and is then applied, unchanged, to the
	transported state z_new of patients it never saw. If the transport really
	moves the patient into the target-dose domain, this frozen "dose readout"
	should report d2 when applied to z_new.

	Unlike a probe retrained on z_new itself, this cannot be satisfied
	trivially: d1 and d2 are explicit inputs of the FiLM MLPs, so *some*
	probe can always recover them from z_new. What is being tested here is
	whether z_new lands on the same dose axis that natural encodings use.
	"""
	kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
	pred_new = np.zeros(len(d1))
	pred_v2 = np.zeros(len(d1))
	for tr, te in kf.split(z_v1):
		X = np.vstack([z_v1[tr], z_v2[tr]])
		y = np.concatenate([d1[tr], d2[tr]])
		m = Pipeline([("sc", StandardScaler()),
					  ("m", RidgeCV(alphas=np.logspace(-3, 3, 13)))]).fit(X, y)
		pred_new[te] = m.predict(z_new[te])
		pred_v2[te] = m.predict(z_v2[te])
	return {
		"r2_readout_znew_vs_d2": float(r2_score(d2, pred_new)),
		"r2_readout_znew_vs_d1": float(r2_score(d1, pred_new)),
		"r2_readout_zv2_vs_d2_ceiling": float(r2_score(d2, pred_v2)),
		"mae_readout_znew_vs_d2": float(np.mean(np.abs(pred_new - d2))),
		"mae_readout_znew_vs_d1": float(np.mean(np.abs(pred_new - d1))),
		"dose_sd": float(np.std(d2)),
	}


# --------------------------------------------------------------------------- #

def build_parser():
	p = argparse.ArgumentParser('Dose-entanglement probe (revision Priority 3)')
	p.add_argument('--ckpt', type=str, required=True, help="Path to the .ckpt to probe.")
	p.add_argument('--model', type=str, default='film', choices=['film', 'dosecond'],
		help="Which architecture the checkpoint holds.")
	p.add_argument('--experiment', type=str, required=True, help="Dataset folder name.")
	p.add_argument('--data-dir', type=str, default='./results/exp_film_run')
	p.add_argument('--eval-split', type=str, default='all', choices=['test', 'all'],
		help="Patients whose representations are probed. The probe itself is always "
			 "cross-validated, so its scores are on patients it did not fit.")
	p.add_argument('-b', '--batch-size', type=int, default=2000)
	p.add_argument('--folds', type=int, default=5)
	p.add_argument('--seed', type=int, default=0)
	p.add_argument('--out-json', type=str, default=None)
	return p


def main():
	args = build_parser().parse_args()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	train_csv = os.path.join(args.data_dir, args.experiment, "virtual_cohort_film_train.csv")
	test_csv = os.path.join(args.data_dir, args.experiment, "virtual_cohort_film_test.csv")
	data_all, _ = extract_gen_tac_film(file_path=[train_csv, test_csv])
	if args.eval_split == 'test':
		train_ids = set(extract_gen_tac_film(file_path=[train_csv])[0].keys())
		data_eval = {k: v for k, v in data_all.items() if k not in train_ids}
	else:
		data_eval = data_all
	print(f"Probing representations of {len(data_eval)} patients ({args.eval_split} split).")

	loader = DataLoader(TacroFilmDataset(data_eval), batch_size=args.batch_size, shuffle=False,
		collate_fn=lambda x: collate_fn_tacro_film(x, device))

	if args.model == 'film':
		model, _ = load_film_model(args.ckpt, device)
	else:
		model, _ = load_dosecond_model(args.ckpt, device)

	reps, targets = collect_representations(model, loader, args.model, device)

	rows = []
	for rep_key, X in reps.items():
		for tgt_name in ("d1", "d2"):
			y = targets[tgt_name]
			reg = regression_probe(X, y, n_splits=args.folds, seed=args.seed)
			clf = classification_probe(X, y, n_splits=args.folds, seed=args.seed)
			rows.append({"representation": PRETTY.get(rep_key, rep_key), "key": rep_key,
						 "target": tgt_name, "dim": int(X.shape[1]), **reg, **clf})

	header = (f"{'representation':<46}{'target':<8}{'lin R2':>9}{'MLP R2':>9}{'best R2':>9}"
			  f"{'shuf R2':>9}{'bal.acc':>9}{'chance':>9}")
	print("\n" + "=" * len(header))
	print("DOSE-PREDICTABILITY PROBE  (higher = more dose information in the representation)")
	print("=" * len(header))
	print(header)
	print("-" * len(header))
	for r in rows:
		print(f"{r['representation']:<46}{r['target']:<8}"
			  f"{r['linear_r2']:>9.3f}{r['mlp_r2']:>9.3f}{r['best_r2']:>9.3f}"
			  f"{r['shuffled_linear_r2']:>9.3f}"
			  f"{r['balanced_accuracy']:>9.3f}{r['chance']:>9.3f}")
	print("=" * len(header))

	transfer = None
	if args.model == 'film':
		def get(key, tgt):
			for r in rows:
				if r["key"] == key and r["target"] == tgt:
					return r["best_r2"]
			return float("nan")

		transfer = transfer_probe(reps["z_v1"], reps["z_v2"], reps["z_new"],
			targets["d1"], targets["d2"], n_splits=args.folds, seed=args.seed)

		print("\nClaim B, part 1 -- is z_base entangled with the administered dose?")
		print(f"  d1 from z_base            : R2 = {get('z_v1', 'd1'):.3f}")
		print(f"  d1 from the raw Visit 1 observations (control) : R2 = {get('raw_v1', 'd1'):.3f}")
		print("  A z_base score at or above the raw-data control means the dose survives "
			  "encoding rather than being averaged out.")

		print("\nClaim B, part 2 -- does the transport land in the target-dose domain?")
		print("  NOTE: a probe *retrained* on z_new recovers d1 and d2 almost perfectly "
			  "by construction,\n        since both are explicit inputs of the FiLM MLPs. "
			  "The frozen dose readout below\n        (fitted on natural encodings only) is the "
			  "informative test.")
		print("  CAVEAT: this readout is hypersensitive. It is a ridge fit on natural")
		print("        encodings, so a modest displacement of z_new along the dose direction")
		print("        produces a huge negative R2 even when z_new's distribution sits right")
		print("        on top of z_v2's. A large negative value here means 'z_new is not at")
		print("        z_v2's exact location', NOT 'z_new is off-manifold', and empirically it")
		print("        does not predict counterfactual accuracy. For evidence that the")
		print("        transport works, use the ablation instead:")
		print("            test_film_matched.py --transport {film,none,oracle}")
		print(f"  frozen dose readout applied to z_new, scored against d2 : R2 = "
			  f"{transfer['r2_readout_znew_vs_d2']:.3f}")
		print(f"  frozen dose readout applied to z_new, scored against d1 : R2 = "
			  f"{transfer['r2_readout_znew_vs_d1']:.3f}")
		print(f"  same readout applied to the true Visit 2 encoding (ceiling) : R2 = "
			  f"{transfer['r2_readout_zv2_vs_d2_ceiling']:.3f}")
		print(f"  |readout(z_new) - d2| = {transfer['mae_readout_znew_vs_d2']:.4f} vs "
			  f"|readout(z_new) - d1| = {transfer['mae_readout_znew_vs_d1']:.4f} "
			  f"(dose SD = {transfer['dose_sd']:.4f}, doses are max-normalised)")

	if args.out_json:
		utils.makedirs(os.path.dirname(args.out_json) or ".")
		with open(args.out_json, "w") as f:
			json.dump({"checkpoint": args.ckpt, "model": args.model,
					   "experiment": args.experiment, "eval_split": args.eval_split,
					   "n_patients": int(len(targets['d1'])), "rows": rows,
					   "transfer_probe": transfer}, f, indent=2)
		print(f"\nProbe results written to {args.out_json}")


if __name__ == '__main__':
	main()
