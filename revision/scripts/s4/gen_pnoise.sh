#!/bin/bash
# Scenario-4 control cohorts with the PAPER's residual noise (sd = 0.71 + 0.113*C ng/mL, the values in
# the committed gen_tacro_film.py) instead of the working copy's 0.03 + 0.03*C.  Same command as
# confound_vc00_s4 (handoff §5.2) and gen_s4_ood.sh otherwise: identical physiology, doses and true AUCs
# (checked on 20 patients), only the noise differs.
#   gen_pnoise.sh <suffix> [ood grid]      e.g.  gen_pnoise.sh ""   |   gen_pnoise.sh _lo 0.25,0.5
cd "$(dirname "$0")/../.."
EXP=confound_vc00_s4_pnoise$1
EXTRA=(); [ -n "${2:-}" ] && EXTRA=(--ood-v2-grid "$2")
nice -n 5 /opt/miniconda3/bin/python3.12 gen_tacro_confound.py --exp "$EXP" --num_patients 1800 --rho 0.0 \
  --test-fraction 0.555556 --seed 0 --dose-seed 1234 --confound-param Vc --scenario 4 \
  --dose-grid 1,2,3,4,5,6,7,8 --prop-sd 0.113 --add-sd 0.71 "${EXTRA[@]}"
