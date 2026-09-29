#!/bin/bash
# Fresh 1000-patient held-out cohorts for leakage-free checkpoint selection.
#
# Same physiology stream in both arms (--seed 777), so the confounded and control
# cohorts are the SAME patients differing only in how the dose was assigned --
# matching the original design.  test-fraction 1.0 makes every patient held out,
# so the visit-2 dose is the randomised intervention throughout.  z_CL is pinned
# to the original cohort: re-deriving it here would shift the dose scale the
# models were trained on.  IDs offset by 10000 so these never collide with the
# existing 1..1000.
set -eu
cd "$(dirname "$0")"
G=1,2,3,4,5,6,7,8
for spec in "confound_km09_big 0.9 confound_km09" "confound_km00_big 0.0 confound_km00"; do
  set -- $spec
  exp=$1; rho=$2; ref=$3
  echo "[$(date '+%T')] generating $exp (rho=$rho)"
  python3 gen_tacro_confound.py --exp "$exp" --num_patients 1000 --rho "$rho" \
      --test-fraction 1.0 --seed 777 --dose-seed 4321 --confound-param Km \
      --scenario 3 --dose-grid "$G" --zstats-from "$ref" --id-offset 10000 \
      --out-root ./results/exp_film_run > "logs/gen_${exp}.log" 2>&1 &
done
wait
echo "[$(date '+%T')] both cohorts generated"
for e in confound_km09_big confound_km00_big; do tail -5 "logs/gen_${e}.log"; echo; done
