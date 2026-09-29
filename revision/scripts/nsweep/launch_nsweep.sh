#!/bin/bash
# Patient-count sweep, scenario 4, CONTROL cohort, main config (identical flags to
# launch_s4.sh: sigma 0.05, sc 141.27, --static-dim 4, dose-cond at the same sigma).
#
#   scripts/nsweep/launch_nsweep.sh "<sizes>" "<seeds>" [niters] [threads]
#   e.g. scripts/nsweep/launch_nsweep.sh "100 200 400" "1" 12000 1
#
# Cohorts confound_vc00_s4_n<N> are nested subsets (make_subsets.py) sharing the 1000
# test patients of confound_vc00_s4.  Each (N, seed, arm) gets its own save root, so no
# run can overwrite another's traj/ (handoff 1.6).  The 800-patient point is the existing
# results/s4_n12000 run.  With N <= 512 and -b 512 every epoch is ONE full-batch step
# (800 patients: 2 steps), so 12000 epochs = 12000 optimiser steps here.
set -eu
cd "$(dirname "$0")/../.."
SIZES=${1:?sizes}; SEEDS=${2:?seeds}; N=${3:-12000}; TH=${4:-1}
# NSWEEP_ROOT picks a fresh root so a relaunch never overwrites an earlier run's traj/.
ROOT=${NSWEEP_ROOT:-results/s4_nsweep}
# NSWEEP_BASE: base cohort whose _n<N> subsets are trained (default confound_vc00_s4)
BASE=${NSWEEP_BASE:-confound_vc00_s4}
# NSWEEP_SIG / NSWEEP_SC: observation std (both arms) and FiLM self-consistency (default main config)
SIG=${NSWEEP_SIG:-0.05}; SC=${NSWEEP_SC:-141.27}
# NSWEEP_ARCHS restricts the arms launched (default both), for CPU-budgeted scheduling.
ARCHS=" ${NSWEEP_ARCHS:-film dc} "
PY=/opt/miniconda3/bin/python3.12
LOG=$ROOT/launch.log; mkdir -p $ROOT
say () { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
for n in $SIZES; do
  c=${BASE}_n$n
  [ -f results/exp_film_run/$c/virtual_cohort_film_train.csv ] || { say "missing cohort $c"; exit 1; }
  for s in $SEEDS; do
    if [[ "$ARCHS" == *" film "* ]]; then
    R=$ROOT/film_n${n}_s$s; mkdir -p "$R/exp_film_run/$c"
    OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH VECLIB_MAXIMUM_THREADS=$TH nohup nice -n 5 \
      $PY run_models.py --niters $N -n 200 -s 40 -l 10 --dataset PK_Tacro \
        --latent-ode --use_film --noise-weight 0.01 --max-t 5. -b 512 --seed "$s" \
        --experiment "$c" --film-no-z0-cond --obsrv-std $SIG --film-self-consistency $SC \
        --decoder-hidden 50 --select-on mse_v2 --patience 1000000 --static-dim 4 \
        --save "./$R/" --eval-every 300 --ckpt-every 300 \
        --log-suffix "__nsweep_n${n}_s$s" >/dev/null 2>&1 &
    say "film      N=$n seed $s (pid $!)"; sleep 2
    fi
    if [[ "$ARCHS" == *" dc "* ]]; then
    R=$ROOT/dc_n${n}_s$s; mkdir -p "$R/exp_dosecond_run/$c"
    OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH VECLIB_MAXIMUM_THREADS=$TH nohup nice -n 5 \
      $PY train_dose_cond.py --experiment "$c" --data-dir ./results/exp_film_run \
        --niters $N -b 512 -l 10 --lr 1e-2 --seed "$s" --patience 1000000 \
        --obsrv-std $SIG --static-dim 4 --save "./$R/" \
        --eval-every 300 --ckpt-every 300 \
        --log-suffix "__nsweep_n${n}_s$s" >/dev/null 2>&1 &
    say "dose-cond N=$n seed $s (pid $!)"; sleep 2
    fi
  done
done
say "launched root=$ROOT base=$BASE sig=$SIG sc=$SC sizes='$SIZES' seeds='$SEEDS' niters=$N threads=$TH"
