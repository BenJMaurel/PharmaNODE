#!/bin/bash
# Re-run the FiLM arm on the 16 sound Michaelis-Menten cohorts (91xxx) with the
# non-linear decoder, so the three-way extrapolation comparison is not the only
# place where our arm runs a non-production configuration.
#
# The command is byte-identical to the original 91xxx FiLM runs except for
# --decoder-hidden 50, so the contrast against those runs isolates the decoder.
#
# Capacity-aware: keeps total training jobs at MAXJOBS, launching a new cohort
# only as a core frees, so it shares the machine with whatever else is running.
set -u
cd /Users/benjaminmaurel/Documents/PharmaNODE
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
MAXJOBS=${MAXJOBS:-11}
LOG=results/film91_dech50.log
COHORTS="91000 91006 91007 91008 91009 91010 91011 91012 91013 91014 91015 91016 91017 91018 91019 91020"
say () { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
busy () { pgrep -f 'run_models\.py|train_dose_cond\.py' | wc -l | tr -d ' '; }

say "film 91xxx + --decoder-hidden 50: 16 cohorts, holding total training jobs at $MAXJOBS"
for c in $COHORTS; do
  R="results/film91_dech50/$c"
  if [ -d "$R" ]; then say "  $c exists, skipping"; continue; fi
  while [ "$(busy)" -ge "$MAXJOBS" ]; do sleep 60; done
  mkdir -p "$R/exp_film_run/$c"
  nohup python3 run_models.py --niters 20000 -n 200 -s 40 -l 10 --dataset PK_Tacro --latent-ode \
      --use_film --noise-weight 0.01 --max-t 5. -b 512 --seed 101 --experiment "$c" \
      --patience 100 --obsrv-std 0.217 --film-self-consistency 7.5 --film-no-z0-cond \
      --encoder-dose-zero --decoder-hidden 50 --save "./$R/" --log-suffix "__dech50_91" \
      >/dev/null 2>&1 &
  say "  launched $c (pid $!), total jobs now $(busy)"
  sleep 3
done
say "all 16 queued; waiting for completion"
wait
say "film 91xxx dech50 complete"
