#!/bin/bash
# Out-of-support dose cohorts for scenario 4, CONTROL arm.
#
# Identical to the command that built confound_vc00_s4 (recovered from the session
# transcript, 2026-09-21) plus --ood-v2-grid.  Same physiology seed and patient
# count, so the 1000 test patients are the SAME people as confound_vc00_s4's test
# split; only their visit-2 dose moves outside the 1-8 mg training grid.  Visit 1
# stays in-grid so the encoder sees a familiar visit (the Km experiment's protocol:
# lo = {0.25, 0.5} mg, hi = {10, 12} mg).
set -eu
cd "$(dirname "$0")"
gen () {
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 nohup nice -n 12 \
    /opt/miniconda3/bin/python3.12 -u gen_tacro_confound.py --exp "$1" \
      --num_patients 1800 --rho 0.0 --test-fraction 0.555556 --seed 0 --dose-seed 1234 \
      --confound-param Vc --scenario 4 --dose-grid 1,2,3,4,5,6,7,8 \
      --ood-v2-grid "$2" --out-root ./results/exp_film_run > "logs/gen_$1.log" 2>&1 &
  echo "  started $1 (V2 doses $2) pid $!"
}
gen confound_vc00_s4_lo 0.25,0.5
gen confound_vc00_s4_hi 10,12
