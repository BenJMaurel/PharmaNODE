#!/bin/bash
# The Vc confounding experiment -- a genuine shortcut test.
#
# Unlike Km, Vc IS inferable from a single-dose curve (R^2 = 0.86 from the latent
# on a control cohort), so a legitimate route to it exists.  If a confounded-trained
# model encodes Vc WORSE than its control-trained twin, it substituted the dose for
# inference it was capable of doing -- a shortcut in the strict sense, which the Km
# design could never demonstrate because Km has no legitimate route.
#
# Vc also acts on the Michaelis-Menten dynamics (C = A/Vc sets the saturation), so
# it shifts the dose-exposure curvature by ~16% and is genuinely needed for the
# counterfactual rather than derivable by proportional scaling.
set -eu
cd "$(dirname "$0")"
G=1,2,3,4,5,6,7,8
SEEDS="1 2 3"
LOG=results/vc_experiment.log
say () { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

say "step 1: training cohorts (800 train / 200 test, physiology seed 0)"
for spec in "confound_vc09 0.9" "confound_vc00 0.0"; do
  set -- $spec
  python3 gen_tacro_confound.py --exp "$1" --num_patients 1000 --rho "$2" \
      --test-fraction 0.2 --seed 0 --dose-seed 1234 --confound-param Vc \
      --scenario 3 --dose-grid "$G" --out-root ./results/exp_film_run \
      > "logs/gen_$1.log" 2>&1 &
done
wait
for e in confound_vc09 confound_vc00; do grep -E "realised corr" "logs/gen_${e}.log" | head -1; done | tee -a "$LOG"
say "step 1 done"

say "step 2: launching training (FiLM + dose-cond, seeds $SEEDS) and the held-out cohorts"
# fresh 1000-patient held-out cohorts, pinned to the training cohorts' dose scale
for spec in "confound_vc09_big 0.9 confound_vc09" "confound_vc00_big 0.0 confound_vc00"; do
  set -- $spec
  python3 gen_tacro_confound.py --exp "$1" --num_patients 1000 --rho "$2" \
      --test-fraction 1.0 --seed 777 --dose-seed 4321 --confound-param Vc \
      --scenario 3 --dose-grid "$G" --zstats-from "$3" --id-offset 10000 \
      --out-root ./results/exp_film_run > "logs/gen_$1.log" 2>&1 &
done
busy () { pgrep -f 'run_models\.py|train_dose_cond\.py' | wc -l | tr -d ' '; }
for s in $SEEDS; do
  for c in confound_vc09 confound_vc00; do
    R="results/vc/film_s${s}"; mkdir -p "$R/exp_film_run/$c"
    while [ "$(busy)" -ge 10 ]; do sleep 60; done
    nohup python3 run_models.py --niters 6000 -n 200 -s 40 -l 10 --dataset PK_Tacro \
        --latent-ode --use_film --noise-weight 0.01 --max-t 5. -b 512 --seed "$s" \
        --experiment "$c" --film-no-z0-cond --obsrv-std 0.217 --film-self-consistency 7.5 \
        --decoder-hidden 50 --select-on mse_v2 --patience 1000000 --save "./$R/" \
        --ckpt-every 1000 --ckpt-dense-from 5000 --ckpt-dense-every 100 \
        --log-suffix "__vc_film_s${s}" >/dev/null 2>&1 &
    say "  film  seed $s $c (pid $!)"; sleep 3
    R="results/vc/dc_s${s}"; mkdir -p "$R/exp_dosecond_run/$c"
    while [ "$(busy)" -ge 10 ]; do sleep 60; done
    nohup python3 train_dose_cond.py --experiment "$c" --data-dir ./results/exp_film_run \
        --niters 6000 -b 512 -l 10 --lr 1e-2 --seed "$s" --patience 1000000 \
        --save "./$R/" --ckpt-every 1000 --ckpt-dense-from 5000 --ckpt-dense-every 100 \
        --log-suffix "__vc_dc_s${s}" >/dev/null 2>&1 &
    say "  dc    seed $s $c (pid $!)"; sleep 3
  done
done
say "all 12 training cells queued"
wait
say "vc experiment training complete"
