###########################
# Builds the three-way comparison table asked for by the reviewer:
#   base probabilistic latent ODE  /  dose-conditioned baseline  /  full OT-FiLM
#
# Reads the JSON files written by test_dose_cond.py (--out-json) and lets you
# type in the numbers printed by the original test_model.py / test_film.py,
# which do not emit JSON.
#
#   python3 compare_revision_results.py results/revision/*.json \
#       --add "Full OT-FiLM (ours):-1.20,10.45,-2.30,11.80" \
#       --add "Base latent ODE (no counterfactual):-0.90,9.80,," \
#       --out results/revision/table.md
###########################

import argparse
import json
import os
import sys


def parse_manual(spec):
	"""'label:v1_mpe,v1_rmspe,v2_mpe,v2_rmspe' -> record. Empty fields become None."""
	if ":" not in spec:
		raise ValueError(f"--add expects 'label:v1_mpe,v1_rmspe,v2_mpe,v2_rmspe', got {spec!r}")
	label, numbers = spec.rsplit(":", 1)
	parts = [x.strip() for x in numbers.split(",")]
	if len(parts) != 4:
		raise ValueError(f"--add needs exactly 4 comma-separated values, got {len(parts)} in {spec!r}")
	vals = [float(x) if x else None for x in parts]
	return {"label": label.strip(),
			"n_patients": None,
			"v1": {"mpe_pct": vals[0], "rmspe_pct": vals[1]},
			"v2": {"mpe_pct": vals[2], "rmspe_pct": vals[3]}}


def fmt(x):
	return "--" if x is None else f"{x:.2f}"


def across_cohorts(root, reference="film", cohorts=None, visit="v2"):
	"""Cohort-level summary and paired comparison.

	Each cohort is one independent draw, so a paired comparison across cohorts
	(same cohorts, different arms) is the right test -- far more informative
	than a within-cohort bootstrap over the held-out patients of one draw.
	"""
	import glob
	import numpy as np

	per_arm = {}
	dirs = sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d))
	for d in dirs:
		cohort = os.path.basename(d)
		if cohorts and cohort not in cohorts:
			continue
		for f in sorted(glob.glob(os.path.join(d, "*.json"))):
			arm = os.path.splitext(os.path.basename(f))[0]
			try:
				rec = json.load(open(f))
			except Exception:
				continue
			if visit not in rec:
				continue
			per_arm.setdefault(arm, {})[cohort] = {
				"rmspe": rec[visit].get("rmspe_pct"),
				"mpe": rec[visit].get("mpe_pct"),
				"label": rec.get("label", arm),
			}

	if not per_arm:
		print(f"No per-cohort JSONs found under {root}.", file=sys.stderr)
		return

	print(f"Visit {visit[-1]} results across cohorts (root: {root})")
	print()
	hdr = f"{'arm':<44}{'cohorts':>8}{'RMSPE mean':>12}{'SD':>8}{'MPE mean':>10}"
	print(hdr); print("-" * len(hdr))
	for arm in sorted(per_arm):
		vals = [v["rmspe"] for v in per_arm[arm].values() if v["rmspe"] is not None]
		mpes = [v["mpe"] for v in per_arm[arm].values() if v["mpe"] is not None]
		if not vals:
			continue
		label = next(iter(per_arm[arm].values()))["label"]
		print(f"{label[:43]:<44}{len(vals):>8}{np.mean(vals):>12.2f}"
			  f"{(np.std(vals, ddof=1) if len(vals) > 1 else float('nan')):>8.2f}"
			  f"{np.mean(mpes):>10.2f}")

	if reference not in per_arm:
		print(f"\nReference arm '{reference}' not found; skipping paired comparison.")
		return

	print()
	print(f"Paired across-cohort comparison against '{reference}'")
	print("(positive difference = the other arm has the HIGHER error, i.e. is worse)")
	print()
	hdr2 = f"{'arm':<44}{'n':>5}{'mean diff':>11}{'95% CI':>20}{'wins':>7}"
	print(hdr2); print("-" * len(hdr2))
	ref = per_arm[reference]
	for arm in sorted(per_arm):
		if arm == reference:
			continue
		shared = sorted(set(ref) & set(per_arm[arm]))
		diffs = np.array([per_arm[arm][c]["rmspe"] - ref[c]["rmspe"] for c in shared
						  if per_arm[arm][c]["rmspe"] is not None and ref[c]["rmspe"] is not None])
		if len(diffs) < 2:
			continue
		m = diffs.mean()
		se = diffs.std(ddof=1) / np.sqrt(len(diffs))
		lo, hi = m - 1.96 * se, m + 1.96 * se
		wins = int((diffs < 0).sum())
		label = next(iter(per_arm[arm].values()))["label"]
		verdict = "" if lo <= 0 <= hi else ("  *" )
		print(f"{label[:43]:<44}{len(diffs):>5}{m:>+11.2f}"
			  f"{f'[{lo:+.2f}, {hi:+.2f}]':>20}{wins:>7}{verdict}")
	print()
	print("'wins' = cohorts where that arm beat the reference. '*' = CI excludes zero.")


def paired_bootstrap(fa, fb, n_boot=10000, visit="v2"):
	"""Is the difference in Visit 2 RMSPE real, or is it n-patient noise?

	Both runs are scored on the same patients, so the comparison must be paired:
	resample patients, and recompute BOTH models' RMSPE on the same resample.
	"""
	import numpy as np
	A, B = json.load(open(fa)), json.load(open(fb))
	for d, f in ((A, fa), (B, fb)):
		if "per_patient" not in d:
			print(f"{f} has no per-patient AUCs -- re-run its test script to get them.",
				  file=sys.stderr)
			sys.exit(1)
	ta = np.array(A["per_patient"][f"true_auc_{visit}"])
	pa = np.array(A["per_patient"][f"pred_auc_{visit}"])
	tb = np.array(B["per_patient"][f"true_auc_{visit}"])
	pb = np.array(B["per_patient"][f"pred_auc_{visit}"])
	if len(ta) != len(tb):
		print(f"Different cohorts ({len(ta)} vs {len(tb)} patients) -- not comparable.",
			  file=sys.stderr)
		sys.exit(1)
	if not np.allclose(ta, tb, rtol=1e-6):
		print("WARNING: the two runs disagree on the true AUCs, so they were not scored "
			  "on the same patients in the same order. Treat the result with suspicion.",
			  file=sys.stderr)

	ea = ((ta - pa) / ta) ** 2
	eb = ((tb - pb) / tb) ** 2
	rmspe = lambda e: float(np.sqrt(np.mean(e)) * 100)
	obs = rmspe(ea) - rmspe(eb)

	rng = np.random.RandomState(0)
	n = len(ea)
	diffs = np.empty(n_boot)
	for i in range(n_boot):
		idx = rng.randint(0, n, n)
		diffs[i] = rmspe(ea[idx]) - rmspe(eb[idx])
	lo, hi = np.percentile(diffs, [2.5, 97.5])

	print(f"Paired bootstrap on Visit {visit[-1]} RMSPE, n = {n} patients, {n_boot} resamples")
	print(f"  A: {A.get('label', fa)}  RMSPE = {rmspe(ea):.2f}%")
	print(f"  B: {B.get('label', fb)}  RMSPE = {rmspe(eb):.2f}%")
	print(f"  difference (A - B) = {obs:+.2f}%   95% CI [{lo:+.2f}, {hi:+.2f}]")
	if lo <= 0.0 <= hi:
		print("  -> the CI spans zero: this data cannot distinguish the two models.")
	else:
		print("  -> the CI excludes zero.")


def main():
	p = argparse.ArgumentParser("Assemble the revision comparison table")
	p.add_argument("json_files", nargs="*", help="JSON files written by test_dose_cond.py")
	p.add_argument("--add", action="append", default=[],
		help="Manual row: 'label:v1_mpe,v1_rmspe,v2_mpe,v2_rmspe' (percentages).")
	p.add_argument("--out", type=str, default=None, help="Write the markdown table here.")
	p.add_argument("--paired", nargs=2, metavar=("A.json", "B.json"),
		help="Paired bootstrap comparing the Visit 2 RMSPE of two runs on the same "
			 "patients. Requires both JSONs to carry per-patient AUCs.")
	p.add_argument("--n-boot", type=int, default=10000)
	p.add_argument("--across-cohorts", metavar="ROOT",
		help="Aggregate results/revision/<cohort>/<arm>.json over many cohorts: "
			 "per-arm mean +/- SD and a PAIRED across-cohort comparison against "
			 "--reference. With many cohorts this is the statistic to report.")
	p.add_argument("--cohorts", type=str, default=None,
		help="Restrict --across-cohorts to these cohort IDs (space separated).")
	p.add_argument("--reference", type=str, default="film",
		help="Arm every other arm is compared against (default: film).")
	p.add_argument("--visit", type=str, default="v2", choices=["v1", "v2"])
	args = p.parse_args()

	if args.across_cohorts:
		across_cohorts(args.across_cohorts, reference=args.reference,
			cohorts=args.cohorts.split() if args.cohorts else None, visit=args.visit)
		return

	if args.paired:
		paired_bootstrap(args.paired[0], args.paired[1], n_boot=args.n_boot)
		if not args.json_files and not args.add:
			return

	rows = []
	for f in args.json_files:
		if not os.path.exists(f):
			print(f"[skip] {f} not found", file=sys.stderr)
			continue
		with open(f) as fh:
			rows.append(json.load(fh))
	rows.extend(parse_manual(s) for s in args.add)

	if not rows:
		print("Nothing to tabulate.", file=sys.stderr)
		sys.exit(1)

	lines = [
		"| Model | n | Visit 1 MPE (%) | Visit 1 RMSPE (%) | Visit 2 MPE (%) | Visit 2 RMSPE (%) |",
		"|---|---:|---:|---:|---:|---:|",
	]
	for r in rows:
		n = r.get("n_patients")
		lines.append("| {} | {} | {} | {} | {} | {} |".format(
			r.get("label", r.get("model", "?")),
			"--" if n is None else n,
			fmt(r["v1"].get("mpe_pct")), fmt(r["v1"].get("rmspe_pct")),
			fmt(r["v2"].get("mpe_pct")), fmt(r["v2"].get("rmspe_pct"))))
	table = "\n".join(lines)

	print(table)
	if args.out:
		os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
		with open(args.out, "w") as fh:
			fh.write(table + "\n")
		print(f"\nWritten to {args.out}", file=sys.stderr)


if __name__ == "__main__":
	main()
