#!/usr/bin/env python3
"""Emit the job list consumed by run_jobs.sh, one evaluation cell per line.

    build_jobs.py --arms narrow --seeds 7 8 9 --epochs 3000 6000 > jobs.tsv
    build_jobs.py --skip-done results/confound_eval/cells.tsv > jobs.tsv

Every combination of arm x seed x epoch x model x training cohort x regime x
evaluation cohort is listed.  A missing checkpoint or evaluation cohort is an
error, not a skip.  --skip-done drops cells already present in a cells table.
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import registry as reg  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=list(reg.ARMS), choices=list(reg.ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, help="default: every seed trained in the arm")
    ap.add_argument("--epochs", nargs="+", type=int, default=[3000, 6000])
    ap.add_argument("--regimes", nargs="+", default=list(reg.REGIMES), choices=list(reg.REGIMES))
    ap.add_argument("--models", nargs="+", default=list(reg.MODELS), choices=list(reg.ALL_MODELS))
    ap.add_argument("--skip-done", metavar="CELLS_TSV")
    a = ap.parse_args()

    done = set()
    if a.skip_done and os.path.exists(a.skip_done):
        with open(a.skip_done) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                done.add((r["seed"], r["epoch"], r["arm"], r["regime"], r["model"], r["trained"], r["evaluated"]))

    out = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
    n = skipped = 0
    for arm in a.arms:
        seeds = [s for s in (a.seeds or reg.ARMS[arm][1]) if s in reg.ARMS[arm][1]]
        for seed in seeds:
            if "w2" in a.models and seed not in reg.W2_ARMS[arm]:
                sys.exit(f"{arm} seed {seed} has no w2ana retrain")
            for ep in a.epochs:
                for model in a.models:
                    for tr in reg.cohorts(arm):
                        ck = reg.checkpoint(arm, seed, model, tr, ep)
                        for regime in a.regimes:
                            for base in reg.cohorts(arm):
                                ev = reg.eval_cohort(base, regime)
                                if not os.path.isdir(os.path.join(reg.REPO, "results", "exp_film_run", ev)):
                                    sys.exit(f"evaluation cohort missing: results/exp_film_run/{ev}")
                                if (str(seed), str(ep), arm, regime, model, tr, ev) in done:
                                    skipped += 1
                                    continue
                                out.writerow([arm, seed, ep, regime, model, tr, ev, ck])
                                n += 1
    print(f"{n} job(s) written, {skipped} already done", file=sys.stderr)


if __name__ == "__main__":
    main()
