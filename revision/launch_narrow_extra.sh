#!/bin/bash
# ---------------------------------------------------------------------------
# Add further narrow-arm seeds once the machine is free.
#
#   ./launch_narrow_extra.sh [--dry-run] [seed ...]      default seeds: 7 8 9
#   NITERS_OVERRIDE=15000 ./launch_narrow_extra.sh       to match seeds 4-6
#
# WHY NEW SEEDS RATHER THAN A SEED-3 RERUN
#   run_models.py:180 and train_dose_cond.py:210 set torch.manual_seed(args.seed)
#   and np.random.seed(args.seed); the loader has no workers and everything runs
#   on CPU.  Re-running --seed 3 therefore reproduces the same optimisation
#   basin exactly.  Seed 3's narrow/FiLM/confounded cell converged to a distinct
#   worse optimum (KL 1.373 at ep6000 against 0.920-0.964 for the other eleven
#   cells); that is a property of the seed, not a transient fault, so it cannot
#   be "re-run away".  Adding seeds dilutes the outlier instead of deleting it,
#   which keeps the reported statistic free of any post-hoc exclusion.
#
# It waits for the current training jobs to drain before starting, so it is safe
# to launch at any time.  Each seed is 4 cells; 3 seeds = 12 concurrent jobs,
# the same load the machine has been carrying.
# ---------------------------------------------------------------------------
set -u
cd /Users/benjaminmaurel/Documents/PharmaNODE
DRY=""; [ "${1:-}" = "--dry-run" ] && { DRY="--dry-run"; shift; }
SEEDS="${*:-7 8 9}"
NIT="${NITERS_OVERRIDE:-6000}"
LOG=results/narrow_extra.log
mkdir -p results

say () { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

for s in $SEEDS; do
  if [ -d "results/seeds/s${s}narrow" ]; then
    echo "REFUSING: results/seeds/s${s}narrow already exists (seed $s would overwrite)." >&2
    exit 1
  fi
done

say "queued: narrow seeds [$SEEDS] at $NIT epochs, ckpt every 1000${DRY:+  [DRY RUN]}"

if [ -z "$DRY" ]; then
  say "waiting for the running jobs to drain..."
  while :; do
    n=$(pgrep -f 'run_models\.py|train_dose_cond\.py' | wc -l | tr -d ' ')
    [ "$n" -eq 0 ] && break
    sleep 300
  done
  say "machine free; launching"
fi

for s in $SEEDS; do
  say "--- seed $s ---"
  NITERS_OVERRIDE="$NIT" ./launch_seed.sh "$s" narrow $DRY 2>&1 | tee -a "$LOG"
done
say "all launched ($(echo $SEEDS | wc -w | tr -d ' ') seeds x 4 cells)"
