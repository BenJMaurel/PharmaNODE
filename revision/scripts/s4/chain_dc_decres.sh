#!/bin/bash
# Score dose-cond + residual decoder at ep6000: paper noise N=100 (seeds 1-3), then low noise N=800 (seeds 1-4).
# Test patients in/lo/hi + training cohort (recalibration factor).
cd /Users/benjaminmaurel/Documents/PharmaNODE
PY=/opt/miniconda3/bin/python3.12; TAG=__dech-50_decres_sig-0.05_ep006000.ckpt
score () {  # root cohort_train seedlist evaldir recaldir
  R=$1; T=$2; SEEDS=$3; E=$4; RC=$5; mkdir -p $E $RC
  for s in $SEEDS; do
    CK=$R/dc_${6}s$s/exp_dosecond_run/$T/traj/experiment_dosecond_$T$TAG
    for reg in "" _lo _hi; do
      nm=${reg#_}; nm=${nm:-in}
      [ -f $E/ep006000_dc_${6}s${s}_on_${nm}.json ] && continue   # restart-safe
      OMP_NUM_THREADS=1 nice -n 10 $PY test_dose_cond.py --experiment ${T%_n100}${reg} --scale-from $T --ckpt $CK \
        --label dcres_s${s}${reg} --out-json $E/ep006000_dc_${6}s${s}_on_${nm}.json > logs/eval_dcres_${6}s${s}${reg}.log 2>&1 &
    done
    [ -f $RC/dc_s${s}_all.json ] || \
    OMP_NUM_THREADS=1 nice -n 10 $PY test_dose_cond.py --experiment $T --scale-from $T --eval-split all --ckpt $CK \
      --label dcres_all_s$s --out-json $RC/dc_s${s}_all.json > logs/recal_dcres_${6}s$s.log 2>&1 &
    wait
  done
}
T=confound_vc00_s4_pnoise_n100; R=results/s4_pnoise_decres
for s in 1 2 3; do while [ ! -f $R/dc_n100_s$s/exp_dosecond_run/$T/traj/experiment_dosecond_$T$TAG ]; do sleep 120; done; done
echo "[$(date '+%F %T')] paper-noise dc seeds at ep6000; scoring"
score $R $T "1 2 3" $R/eval_dc results/recal/paper_decres n100_
echo "DC_DECRES_PNOISE_DONE $(date '+%F %T')"
T=confound_vc00_s4; R=results/s4_decres
for s in 1 2 3 4; do while [ ! -f $R/dc_s$s/exp_dosecond_run/$T/traj/experiment_dosecond_$T$TAG ]; do sleep 300; done; done
echo "[$(date '+%F %T')] low-noise dc seeds at ep6000; scoring"
score $R $T "1 2 3 4" $R/eval_dc results/recal/low_n800_decres ""
echo "DC_DECRES_ALL_DONE $(date '+%F %T')"
