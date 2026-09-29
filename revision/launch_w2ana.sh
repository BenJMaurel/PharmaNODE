#!/bin/bash
# ---------------------------------------------------------------------------
# The confounding matrix retrained with the corrected W2 penalty.
#
#   ./launch_w2ana.sh <seed> <wide|narrow|both> [--dry-run]
#
# Identical to launch_seed.sh except for --w2-analytic-moments, which computes
# the transported posterior's moments in closed form instead of estimating them
# from n_traj_samples draws. Checkpoints carry a "_w2ana" tag, so these runs can
# never be confused with, or written over, anything trained under the old
# objective.
#
# Only the FiLM arm is retrained: dose-cond has no transported posterior and so
# no W2 term (verified: no w2/emp_mu/emp_std in lib/dose_conditioned.py), which
# means the existing dose-cond checkpoints remain valid comparators.
# ---------------------------------------------------------------------------
set -u
SEED="${1:?usage: ./launch_w2ana.sh <seed> <wide|narrow|both> [--dry-run]}"
ARM="${2:?usage: ./launch_w2ana.sh <seed> <wide|narrow|both> [--dry-run]}"
DRY=""; [ "${3:-}" = "--dry-run" ] && DRY=1
cd /Users/benjaminmaurel/Documents/PharmaNODE
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
NITERS="${NITERS_OVERRIDE:-6000}"
run () { if [ -n "$DRY" ]; then echo "    $*"; else nohup "$@" >/dev/null 2>&1 & echo "    pid $!"; fi; }

launch_arm () {
  local arm=$1 cells sel tag R
  if [ "$arm" = wide ]; then
    cells="confound_km09 confound_km00"; sel="--select-on mse_v2"; tag="w2ana_s${SEED}wide"
  else
    cells="confound_kmn09 confound_kmn00"; sel=""; tag="w2ana_s${SEED}narrow"
  fi
  R="results/w2ana/${tag}"
  echo "  [$arm] -> $R"
  for c in $cells; do
    [ -n "$DRY" ] || mkdir -p "$R/exp_film_run/$c"
    echo "  OT-FiLM $c"
    run python3 run_models.py --niters "$NITERS" -n 200 -s 40 -l 10 --dataset PK_Tacro \
        --latent-ode --use_film --noise-weight 0.01 --max-t 5. -b 512 --seed "$SEED" \
        --experiment "$c" --film-no-z0-cond --obsrv-std 0.217 --film-self-consistency 7.5 \
        --decoder-hidden 50 --w2-analytic-moments $sel --patience 1000000 \
        --save "./$R/" --ckpt-every 1000 --log-suffix "__${tag}"
  done
}
echo "seed $SEED | arm $ARM | $NITERS epochs | analytic W2 moments${DRY:+  [DRY RUN]}"
case "$ARM" in
  wide) launch_arm wide ;; narrow) launch_arm narrow ;;
  both) launch_arm wide; launch_arm narrow ;;
  *) echo "arm must be wide|narrow|both"; exit 1 ;;
esac
