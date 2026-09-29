#!/bin/bash
# Sweep the explicit transported-width ridge.
#
#   ./queue_ridge_sweep.sh [seeds...]      default: 1 2 3
#
# lambda = 0.548 reproduces, in expectation, the implicit ridge the S=3 empirical
# W2 was applying before --w2-analytic-moments; lambda = 2 asks whether more of it
# helps.  lambda = 0 is NOT re-run: the existing results/w2ana/ seeds 1-3 already
# are that arm, at the same epochs and hyperparameters.
#
# Wide arm only -- the narrow arm's DiD is null, so it carries no signal for this
# question.  Saves under results/ridge/, never touching results/seeds/ or
# results/w2ana/; the _wr<lambda> filename tag keeps every checkpoint distinct.
set -u
cd "$(dirname "$0")"
SEEDS=${@:-"1 2 3"}
MAXJOBS=${MAXJOBS:-11}
# One BLAS thread per process.  Unthrottled, each run spawns ~18 threads, so 12
# concurrent runs put ~216 threads on 11 cores: measured 10 ep/min per process
# against 28-30 ep/min for a single-threaded process on the same machine.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LOG=results/ridge_queue.log
mkdir -p results/ridge
say () { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
busy () { pgrep -f 'run_models\.py|train_dose_cond\.py' | wc -l | tr -d ' '; }
say "sweep start: seeds $SEEDS, lambda 0.548 and 2.0, wide arm, 6000 epochs"
for lam in 0.548 2.0; do
  for s in $SEEDS; do
    for c in confound_km09 confound_km00; do
      tag="ridge_s${s}wide_wr${lam}"; R="results/ridge/${tag}"
      if ls "$R/exp_film_run/$c/"*_wr*_best.ckpt >/dev/null 2>&1; then
        say "  skip seed $s lambda $lam $c (already trained)"; continue
      fi
      while [ "$(busy)" -ge "$MAXJOBS" ]; do sleep 60; done
      mkdir -p "$R/exp_film_run/$c"
      nohup python3 run_models.py --niters 6000 -n 200 -s 40 -l 10 --dataset PK_Tacro \
          --latent-ode --use_film --noise-weight 0.01 --max-t 5. -b 512 --seed "$s" \
          --experiment "$c" --film-no-z0-cond --obsrv-std 0.217 --film-self-consistency 7.5 \
          --decoder-hidden 50 --w2-analytic-moments --width-ridge "$lam" --select-on mse_v2 \
          --patience 1000000 --save "./$R/" --ckpt-every 1000 \
          --ckpt-dense-from 5000 --ckpt-dense-every 100 \
          --log-suffix "__${tag}" >/dev/null 2>&1 &
      say "  launched seed $s lambda $lam $c (pid $!), jobs now $(busy)"
      sleep 3
    done
  done
done
say "all queued"
wait
say "ridge sweep complete"
