#!/bin/bash
# ---------------------------------------------------------------------------
# Launch one seed of the confounding experiment, matching the protocol already
# used for seed 2.  NOTHING runs until you invoke this.
#
#   ./launch_seed.sh <seed> <wide|narrow|both> [--dry-run]
#
# Protocol (identical across seeds so checkpoints are epoch-comparable):
#   * stop at 6000 epochs   (--niters 6000, patience disabled)
#     override with  NITERS_OVERRIDE=15000 ./launch_seed.sh ...
#   * --ckpt-every 1000     (6 trajectory checkpoints per cell)
#   * isolated --save tree per seed, so nothing existing is touched
#   * OT-FiLM  = main config + --decoder-hidden 50
#       wide   also uses --select-on mse_v2   (matches seed 1 + 2 wide)
#       narrow uses the default nll_v2        (matches seed 1 + 2 narrow)
#   * dose-cond = plain config
# ---------------------------------------------------------------------------
set -u
SEED="${1:?usage: ./launch_seed.sh <seed> <wide|narrow|both> [--dry-run]}"
ARM="${2:?usage: ./launch_seed.sh <seed> <wide|narrow|both> [--dry-run]}"
DRY=""; [ "${3:-}" = "--dry-run" ] && DRY=1

cd /Users/benjaminmaurel/Documents/PharmaNODE
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
NITERS="${NITERS_OVERRIDE:-6000}"   # default unchanged; export NITERS_OVERRIDE to lengthen a run
run () { if [ -n "$DRY" ]; then echo "    $*"; else nohup "$@" >/dev/null 2>&1 & echo "    pid $!"; fi; }

launch_arm () {   # $1 = wide|narrow
  local arm=$1 tag sel
  if [ "$arm" = wide ]; then
    cells="confound_km09 confound_km00"; sel="--select-on mse_v2"; tag="s${SEED}wide"
  else
    cells="confound_kmn09 confound_kmn00"; sel=""; tag="s${SEED}narrow"
  fi
  local R="results/seeds/${tag}"
  for c in $cells; do
    mkdir -p "$R/exp_film_run/$c" "$R/exp_dosecond_run/$c"
  done
  echo "  [$arm]  -> $R"
  for c in $cells; do
    echo "  OT-FiLM $c"
    run python3 run_models.py --niters $NITERS -n 200 -s 40 -l 10 --dataset PK_Tacro \
        --latent-ode --use_film --noise-weight 0.01 --max-t 5. -b 512 --seed "$SEED" \
        --experiment "$c" --film-no-z0-cond --obsrv-std 0.217 --film-self-consistency 7.5 \
        --decoder-hidden 50 $sel --patience 1000000 --save "./$R/" \
        --ckpt-every 1000 --log-suffix "__${tag}"
    echo "  dose-cond $c"
    run python3 train_dose_cond.py --experiment "$c" --data-dir ./results/exp_film_run \
        --niters $NITERS -b 512 -l 10 --lr 1e-2 --seed "$SEED" --patience 1000000 \
        --save "./$R/" --ckpt-every 1000 --log-suffix "__${tag}"
  done
}

echo "seed $SEED | arm $ARM | stop at $NITERS epochs | ckpt every 1000${DRY:+  [DRY RUN]}"
case "$ARM" in
  wide)   launch_arm wide ;;
  narrow) launch_arm narrow ;;
  both)   launch_arm wide; launch_arm narrow ;;
  *) echo "arm must be wide|narrow|both"; exit 1 ;;
esac
echo "done"
