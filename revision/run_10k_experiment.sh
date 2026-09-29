#!/bin/bash
# Ceiling / convergence run at 10x data: 8000 train, 2000 internal test, one seed.
#
# Batch stays at 512.  The benchmark showed larger batches raise epochs/min
# (1.25 -> 1.96) but CUT optimiser steps/min five-fold (20.0 -> 3.9), because an
# epoch at b=4096 holds 2 updates where b=512 holds 16.  Epochs/min is the wrong
# unit; learning tracks steps.  b=512 also keeps the per-step dynamics identical
# to every 800-patient run, so cohort size is the only variable that changed.
set -eu
cd "$(dirname "$0")"
SEED=1; N=3000
say () { echo "[$(date '+%F %T')] $*" | tee -a results/vc10k_experiment.log; }
for c in confound_vc09_10k confound_vc00_10k; do
  R=results/vc10k/film; mkdir -p "$R/exp_film_run/$c"
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 nohup \
    python3 run_models.py --niters $N -n 200 -s 40 -l 10 --dataset PK_Tacro \
      --latent-ode --use_film --noise-weight 0.01 --max-t 5. -b 512 --seed $SEED \
      --experiment "$c" --film-no-z0-cond --obsrv-std 0.217 --film-self-consistency 7.5 \
      --decoder-hidden 50 --select-on mse_v2 --patience 1000000 --save "./$R/" \
      --eval-every 300 --ckpt-every 300 --log-suffix "__10k_film" >/dev/null 2>&1 &
  say "film  $c (pid $!)"; sleep 3
  R=results/vc10k/dc; mkdir -p "$R/exp_dosecond_run/$c"
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 nohup \
    python3 train_dose_cond.py --experiment "$c" --data-dir ./results/exp_film_run \
      --niters $N -b 512 -l 10 --lr 1e-2 --seed $SEED --patience 1000000 \
      --save "./$R/" --eval-every 300 --ckpt-every 300 \
      --log-suffix "__10k_dc" >/dev/null 2>&1 &
  say "dc    $c (pid $!)"; sleep 3
done
say "4 cells launched at 8000 train / 2000 test, niters=$N"
