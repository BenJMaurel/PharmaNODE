#!/bin/bash
# Clinically plausible cohorts (2026-09-24 convention, handoff §1.12): true average concentration over the
# VISIT-1 dosing interval in 3-25 ng/mL; visit-2 (counterfactual) dose unrestricted; out-of-range cohorts keep
# every patient's visit-1 dose identical (--ood-keep-v1). Paper noise. ~42-49% of patients pass, so 2500 are
# drawn: ~240 train -> >=100 kept (subset to exactly 100 with make_subsets.py), ~2260 test -> ~1000 kept.
#   gen_window.sh <gauss|tslow> <suffix> [ood grid]     e.g.  gen_window.sh tslow _lo 0.25,0.5
cd "$(dirname "$0")/../.."
SC=$1; SFX=${2:-}; EXTRA=()
[ -n "${3:-}" ] && EXTRA=(--ood-v2-grid "$3" --ood-keep-v1)
if [ "$SC" = tslow ]; then DIST=(--eta-dist log_t --t-df 4 --ktr-slow-frac 0.3 --ktr-slow-factor 0.3); EXP=confound_vc00_s4_tslow_win$SFX
else DIST=(); EXP=confound_vc00_s4_pnoise_win$SFX; fi
OMP_NUM_THREADS=1 nice -n 5 /opt/miniconda3/bin/python3.12 gen_tacro_confound.py --exp "$EXP" --num_patients 2500 --rho 0.0 \
  --test-fraction 0.904 --seed 0 --dose-seed 1234 --confound-param Vc --scenario 4 --dose-grid 1,2,3,4,5,6,7,8 \
  --prop-sd 0.113 --add-sd 0.71 --v1-cavg-window 3,25 "${DIST[@]}" "${EXTRA[@]}"
