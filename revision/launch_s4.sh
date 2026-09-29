#!/bin/bash
# Scenario-4 runs: hematocrit varies and carries a covariate on Km, plus an
# omega block corr(eta_Km, eta_Vc) = -0.70.  800 train / 1000 test.
#
#   ./launch_s4.sh <obsrv_std> <film_self_consistency> [seeds] [niters] [threads] [save_root]
#
# save_root matters: checkpoint filenames encode the CONFIG, not the epoch budget,
# so two runs of the same config under one root overwrite each other's traj/.
#
# e.g.  ./launch_s4.sh 0.05 141.27 "1"                the scaled config, 3000 epochs
#       ./launch_s4.sh 0.05 141.27 "2 3 4" 6000 1     three seeds to 6000, 1 thread each
#
# threads matters: each seed launches 4 processes (2 cohorts x 2 arms), so three
# seeds is 12 processes. At 2 threads each that oversubscribes an 11-core box.
#
# --static-dim 4 is what lets the encoder see hematocrit at all; without it the
# whole scenario-4 design is invisible to both arms.  dose-cond takes the same
# --obsrv-std so the two arms stay matched outside the vector field.
set -eu
cd "$(dirname "$0")"
SIG="${1:-0.05}"; SC="${2:-141.27}"; SEEDS="${3:-1}"
N="${4:-3000}"; TH="${5:-2}"; ROOT="${6:-results/s4}"
TAG="s4_sig${SIG}_sc${SC}_n${N}"
say () { echo "[$(date '+%F %T')] $*" | tee -a results/s4_experiment.log; }

for s in $SEEDS; do
  for c in confound_vc00_s4 confound_vc09_s4; do
    R="${ROOT}/film_s${s}"; mkdir -p "$R/exp_film_run/$c"
    OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH VECLIB_MAXIMUM_THREADS=$TH nohup nice -n 5 \
      python3 run_models.py --niters $N -n 200 -s 40 -l 10 --dataset PK_Tacro \
        --latent-ode --use_film --noise-weight 0.01 --max-t 5. -b 512 --seed "$s" \
        --experiment "$c" --film-no-z0-cond --obsrv-std "$SIG" \
        --film-self-consistency "$SC" --decoder-hidden 50 --select-on mse_v2 \
        --patience 1000000 --static-dim 4 --save "./$R/" \
        --eval-every 300 --ckpt-every 300 \
        --log-suffix "__${TAG}_s${s}" >/dev/null 2>&1 &
    say "film      $c seed $s (pid $!)"; sleep 3

    R="${ROOT}/dc_s${s}"; mkdir -p "$R/exp_dosecond_run/$c"
    OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH VECLIB_MAXIMUM_THREADS=$TH nohup nice -n 5 \
      python3 train_dose_cond.py --experiment "$c" --data-dir ./results/exp_film_run \
        --niters $N -b 512 -l 10 --lr 1e-2 --seed "$s" --patience 1000000 \
        --obsrv-std "$SIG" --static-dim 4 --save "./$R/" \
        --eval-every 300 --ckpt-every 300 \
        --log-suffix "__${TAG}_s${s}" >/dev/null 2>&1 &
    say "dose-cond $c seed $s (pid $!)"; sleep 3
  done
done
say "launched: sigma=$SIG, sc=$SC, seeds='$SEEDS', niters=$N, threads=$TH, root=$ROOT, static-dim=4"
