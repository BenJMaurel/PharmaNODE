#!/bin/bash
# A THIRD independent 1000-patient cohort (physiology seed 888).
#
# The model contrast is stable WITHIN a cohort (selA +3.29, selB +3.44) but flips
# between the seed-0 cohort (-3.77) and the seed-777 one (+3.3).  Two draws cannot
# say which is typical; a third can.  Everything else identical to the seed-777
# generation, including the pinned z_CL standardisation.
set -eu
cd "$(dirname "$0")"
G=1,2,3,4,5,6,7,8
for spec in "confound_km09_c3 0.9 confound_km09" "confound_km00_c3 0.0 confound_km00"; do
  set -- $spec
  exp=$1; rho=$2; ref=$3
  python3 gen_tacro_confound.py --exp "$exp" --num_patients 1000 --rho "$rho" \
      --test-fraction 1.0 --seed 888 --dose-seed 4321 --confound-param Km \
      --scenario 3 --dose-grid "$G" --zstats-from "$ref" --id-offset 20000 \
      --out-root ./results/exp_film_run > "logs/gen_${exp}.log" 2>&1 &
done
wait
echo "third cohort generated"
for e in confound_km09_c3 confound_km00_c3; do grep "realised corr(log Km, d1)" "logs/gen_${e}.log"; done
