#!/bin/bash
# ---------------------------------------------------------------------------
# Unattended chain of the long revision experiments.  Stages run one after the
# other; each stage saturates the machine (~12 concurrent single-threaded jobs)
# and the next starts only when the previous has drained.
#
#   ./run_revision_chain.sh [--dry-run]
#
# Stage 0  generate the rho cohorts (wide grid, same patients, dose rule varied)
# Stage 1  wide arm seeds 4,5,6            -> powers the central DiD result
# Stage 2  OT ablation (self-consistency 0)-> reviewer 2's requested ablation
# Stage 3  rho sweep, training side        -> does the DiD grow with confounding?
#
# Everything is additive: new cohorts, new save trees, new log suffixes.
# ---------------------------------------------------------------------------
set -u
cd /Users/benjaminmaurel/Documents/PharmaNODE
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
DRY=""; [ "${1:-}" = "--dry-run" ] && DRY=1
LOG=results/revision_chain.log
WIDE_GRID="1,2,3,4,5,6,7,8"
FILM_MAIN="--film-no-z0-cond --obsrv-std 0.217 --film-self-consistency 7.5 --decoder-hidden 50"
say () { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
run () { if [ -n "$DRY" ]; then echo "      $*"; else nohup "$@" >/dev/null 2>&1 & fi; }
drain () { [ -n "$DRY" ] && return 0
           while [ "$(pgrep -f 'run_models\.py|train_dose_cond\.py|gen_tacro_confound\.py' | wc -l | tr -d ' ')" -gt 0 ]; do sleep 120; done; }

say "=== revision chain starting${DRY:+ [DRY RUN]} ==="

# ---------------- Stage 0: cohorts ----------------------------------------
say "stage 0: generating rho cohorts"
for spec in "confound_km03 0.3" "confound_km06 0.6" "confound_kmneg9 -0.9"; do
  set -- $spec
  if [ -d "results/exp_film_run/$1" ]; then say "  $1 exists, skipping"; continue; fi
  say "  generating $1 (rho=$2)"
  run python3 gen_tacro_confound.py --exp "$1" --rho "$2" --confound-param Km --dose-grid "$WIDE_GRID"
done
drain; say "stage 0 done"

# ---------------- Stages 1-2: wide seeds ----------------------------------
say "stage 1: wide arm seeds 4 5 6"
for s in 4 5 6; do
  [ -d "results/seeds/s${s}wide" ] && { say "  s${s}wide exists, skipping"; continue; }
  if [ -n "$DRY" ]; then echo "      ./launch_seed.sh $s wide"; else ./launch_seed.sh "$s" wide >>"$LOG" 2>&1; fi
done
drain; say "stage 1 done"

# ---------------- Stage 2: OT ablation ------------------------------------
say "stage 2: OT ablation, FiLM with --film-self-consistency 0"
for c in 90000 93000 confound_km09 confound_km00; do
  for s in 1 2 3; do
    R="results/ablation_noOT/${c}_s${s}"; [ -d "$R" ] && { say "  $R exists, skipping"; continue; }
    [ -n "$DRY" ] || mkdir -p "$R/exp_film_run/$c"
    run python3 run_models.py --niters 6000 -n 200 -s 40 -l 10 --dataset PK_Tacro --latent-ode \
        --use_film --noise-weight 0.01 --max-t 5. -b 512 --seed "$s" --experiment "$c" \
        --film-no-z0-cond --obsrv-std 0.217 --film-self-consistency 0 --decoder-hidden 50 \
        --patience 1000000 --save "./$R/" --ckpt-every 1000 --log-suffix "__noOT_${c}_s${s}"
  done
done
drain; say "stage 2 done"
# ---------------- Stage 3: rho sweep, training side -----------------------
say "stage 3: rho sweep (train on rho=0.3 and 0.6; controls already exist)"
for c in confound_km03 confound_km06; do
  for s in 1 2 3; do
    R="results/rho_sweep/${c}_s${s}"; [ -d "$R" ] && { say "  $R exists, skipping"; continue; }
    [ -n "$DRY" ] || mkdir -p "$R/exp_film_run/$c" "$R/exp_dosecond_run/$c"
    run python3 run_models.py --niters 6000 -n 200 -s 40 -l 10 --dataset PK_Tacro --latent-ode \
        --use_film --noise-weight 0.01 --max-t 5. -b 512 --seed "$s" --experiment "$c" \
        $FILM_MAIN --select-on mse_v2 --patience 1000000 --save "./$R/" \
        --ckpt-every 1000 --log-suffix "__rho_${c}_s${s}"
    run python3 train_dose_cond.py --experiment "$c" --data-dir ./results/exp_film_run \
        --niters 6000 -b 512 -l 10 --lr 1e-2 --seed "$s" --patience 1000000 \
        --save "./$R/" --ckpt-every 1000 --log-suffix "__rho_${c}_s${s}"
  done
done
drain; say "stage 3 done"

say "=== revision chain complete ==="
