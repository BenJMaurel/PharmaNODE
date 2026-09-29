#!/bin/bash
# Scenario 4 with a NON-GAUSSIAN parameter distribution (Benjamin 2026-09-23): Student-t (df 4, variance-matched)
# random effects on every parameter + an unrecorded slow-absorber subgroup (30%, Ktr x 0.3). Paper noise.
# 100 training + 1000 test patients (the fit/training size is N=100 only).
#   gen_tslow.sh <suffix> [ood grid]      e.g.  gen_tslow.sh ""   |   gen_tslow.sh _lo 0.25,0.5
cd "$(dirname "$0")/../.."
EXP=confound_vc00_s4_tslow$1
EXTRA=(); [ -n "${2:-}" ] && EXTRA=(--ood-v2-grid "$2")
OMP_NUM_THREADS=1 nice -n 5 /opt/miniconda3/bin/python3.12 gen_tacro_confound.py --exp "$EXP" --num_patients 1100 --rho 0.0 \
  --test-fraction 0.9090909 --seed 0 --dose-seed 1234 --confound-param Vc --scenario 4 --dose-grid 1,2,3,4,5,6,7,8 \
  --prop-sd 0.113 --add-sd 0.71 --eta-dist log_t --t-df 4 --ktr-slow-frac 0.3 --ktr-slow-factor 0.3 "${EXTRA[@]}"
