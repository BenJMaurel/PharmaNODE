#!/bin/bash
# Hidden-saturation experiment (handoff 11), step 1: calibration.
# Train cohorts at k = 10, 20, 30 (Km and Vmax scaled together; windowed 3-25 ng/mL, paper noise, rho 0,
# doses 1-8 mg), subset to exactly 100 train patients, then Monolix linear vs Michaelis-Menten fits with
# AIC/BIC (importance sampling). 4 fits at a time x 2 threads = 800% cap.
cd "$(dirname "$0")/../.."
PY=/opt/miniconda3/bin/python3.12; R=results/exp_film_run; OUT=results/sat/calib
say() { echo "[$(date +%H:%M:%S)] $*"; }
mkdir -p $OUT logs
KS="10 20 30"
for k in $KS; do
  B=confound_vc00_s4_sat${k}_win
  [ -d $R/$B ] && { say "$B exists -- left untouched"; continue; }
  OMP_NUM_THREADS=1 nice -n 5 $PY gen_tacro_confound.py --exp $B --num_patients 300 --rho 0.0 \
    --test-fraction 0.1 --seed 0 --dose-seed 1234 --confound-param Vc --scenario 4 --dose-grid 1,2,3,4,5,6,7,8 \
    --prop-sd 0.113 --add-sd 0.71 --v1-cavg-window 3,25 --sat-scale $k > logs/gen_sat${k}.log 2>&1 &
done
wait
for k in $KS; do
  B=confound_vc00_s4_sat${k}_win
  grep -h "rejected" logs/gen_sat${k}.log
  [ -d $R/${B}_n100 ] || $PY scripts/nsweep/make_subsets.py $B "100" || { say "SUBSET FAILED $k"; exit 1; }
  cp -n $R/$B/noise.json $R/${B}_n100/noise.json
  $PY scripts/monolix/build_monolix_data.py ${B}_n100 $OUT/data_sat${k}_n100.csv || { say "DATA FAILED $k"; exit 1; }
done
say "cohorts ready; starting fits"
JOBS=()
for k in $KS; do for s in lin mm; do JOBS+=("$k $s"); done; done
run_fit() {
  set -- $1
  Rscript scripts/monolix/fit_s4_select.R $OUT/data_sat${1}_n100.csv $OUT/sat${1}_$2 2 $2 $1 > logs/fit_sat${1}_$2.log 2>&1
  echo "[$(date +%H:%M:%S)] fit k=$1 $2 exit $?"
}
i=0
for j in "${JOBS[@]}"; do
  run_fit "$j" &
  i=$((i+1)); if [ $((i % 4)) -eq 0 ]; then wait; fi
done
wait
$PY scripts/sat/calib_table.py | tee $OUT/calib_table.txt
say "CALIB_DONE"
