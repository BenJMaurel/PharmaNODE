###########################
# Calibration of the predictive AUC distribution (paper revision)
#
# The counterfactual RMSPE comparison scores a point prediction, which is the one
# axis on which a probabilistic model has no structural advantage. These metrics
# score the *distribution* the model emits, which the deterministic baselines
# cannot produce at all.
#
# Purely additive.
###########################

import numpy as np


def coverage_and_width(auc_samples, auc_true, levels=(0.5, 0.8, 0.95)):
	"""Empirical coverage of central predictive intervals, and their width.

	auc_samples: [n_samples, n_patients] AUCs, one per posterior draw
	auc_true:    [n_patients]
	Returns {level: {"coverage", "mean_width_pct"}}. Coverage should match the
	nominal level; width is reported relative to the true AUC so it is
	comparable across cohorts (an interval can only be trusted if it is both
	well covered and tight).
	"""
	out = {}
	for lv in levels:
		lo = np.percentile(auc_samples, 100 * (1 - lv) / 2, axis=0)
		hi = np.percentile(auc_samples, 100 * (1 + lv) / 2, axis=0)
		inside = (auc_true >= lo) & (auc_true <= hi)
		out[str(lv)] = {
			"nominal": lv,
			"coverage": float(inside.mean()),
			"mean_width_pct": float(np.mean((hi - lo) / auc_true) * 100),
		}
	return out


def pit_values(auc_samples, auc_true):
	"""Probability integral transform: the predictive CDF evaluated at the truth.

	For a perfectly calibrated model these are Uniform(0,1). Systematic
	departures say what is wrong: mass near 0/1 means intervals too narrow,
	mass in the middle means too wide, a shifted mean means bias.
	"""
	return (auc_samples < auc_true[None, :]).mean(axis=0)


def crps(auc_samples, auc_true):
	"""Continuous ranked probability score, ensemble form. Lower is better.

	CRPS = E|X - y| - 0.5 E|X - X'|. A proper scoring rule, so it cannot be
	gamed by widening or narrowing the distribution -- it rewards being both
	calibrated and sharp. Reported relative to the true AUC so cohorts compare.
	"""
	n = auc_samples.shape[0]
	term1 = np.mean(np.abs(auc_samples - auc_true[None, :]), axis=0)
	diffs = np.abs(auc_samples[:, None, :] - auc_samples[None, :, :])
	term2 = diffs.sum(axis=(0, 1)) / (2.0 * n * n)
	return float(np.mean((term1 - term2) / auc_true) * 100)


def calibration_report(auc_samples, auc_true, levels=(0.5, 0.8, 0.95)):
	auc_samples = np.asarray(auc_samples, dtype=float)
	auc_true = np.asarray(auc_true, dtype=float)
	cw = coverage_and_width(auc_samples, auc_true, levels)
	pit = pit_values(auc_samples, auc_true)
	# mean absolute deviation of empirical from nominal coverage, over the levels
	ece = float(np.mean([abs(cw[k]["coverage"] - cw[k]["nominal"]) for k in cw]))
	return {
		"n_samples": int(auc_samples.shape[0]),
		"intervals": cw,
		"calibration_error": ece,
		"crps_pct": crps(auc_samples, auc_true),
		"pit_mean": float(pit.mean()),          # 0.5 if unbiased
		"pit_std": float(pit.std()),            # 0.289 for Uniform(0,1)
		"pit": pit.tolist(),
	}


def enable_learnable_obsrv_std(model, init=None):
	"""Turn the fixed observation-noise constant into a learned parameter.

	The Gaussian log-likelihood in lib/likelihood_eval.py uses Normal(...).log_prob,
	which carries the -log(sigma) normaliser, so sigma is identifiable: shrinking it
	is punished by that term and widening it is punished by the squared-error term.
	It therefore settles at the residual scale on its own, instead of being pinned at
	a hand-set constant that was ~20x too small.

	Call BEFORE building the optimiser, so the new parameter is included.
	"""
	import torch
	import torch.nn as nn
	cur = model.obsrv_std.detach().clone().float()
	if init is not None:
		cur = torch.full_like(cur, float(init))
	model._log_obsrv_std = nn.Parameter(torch.log(cur))
	model._learn_obsrv_std = True
	return model
