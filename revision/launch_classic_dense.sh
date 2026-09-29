#!/bin/bash
# Retrain classic FiLM with dense late checkpoints so it can be compared to the
# ridge arms on a matched 11-point window.
#
# The flags are byte-identical to the original classic runs (no
# --w2-analytic-moments, no --width-ridge) EXCEPT for --ckpt-dense-*, which only
# adds save points and does not touch training.  Saved under results/classic_dense/
# because the filename tag is the same as the originals -- writing into
# results/seeds/ would overwrite them.
#
# Single-threaded to match how the ridge arms were trained, so the two sides of
# the comparison share their numerics.
set -u
cd "$(dirname "$0")"
SEEDS=${@:-"1 2 3"}
MAXJOBS=${MAXJOBS:-10}
LOG=results/classic_dense_queue.log
mkdir -p results/classic_dense
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
say () { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
busy () { pgrep -f 'run_models\.py|train_dose_cond\.py' | wc -l | tr -d ' '; }
say "classic FiLM dense retrain: seeds $SEEDS, wide arm, 6000 epochs"
for s in $SEEDS; do
  for c in confound_km09 confound_km00; do
    tag="classic_s${s}wide"; R="results/classic_dense/${tag}"
    if ls "$R/exp_film_run/$c/"*_best.ckpt >/dev/null 2>&1; then
      say "  skip seed $s $c (already trained)"; continue
    fi
    while [ "$(busy)" -ge "$MAXJOBS" ]; do sleep 60; done
    mkdir -p "$R/exp_film_run/$c"
    nohup python3 run_models.py --niters 6000 -n 200 -s 40 -l 10 --dataset PK_Tacro \
        --latent-ode --use_film --noise-weight 0.01 --max-t 5. -b 512 --seed "$s" \
        --experiment "$c" --film-no-z0-cond --obsrv-std 0.217 --film-self-consistency 7.5 \
        --decoder-hidden 50 --select-on mse_v2 --patience 1000000 \
        --save "./$R/" --ckpt-every 1000 --ckpt-dense-from 5000 --ckpt-dense-every 100 \
        --log-suffix "__${tag}" >/dev/null 2>&1 &
    say "  launched seed $s $c (pid $!), jobs now $(busy)"
    sleep 3
  done
done
say "all queued"
wait
say "classic dense retrain complete"
