#!/bin/bash
# Residual-decoder FiLM (linear + zero-init MLP): score both regimes at ep6000 on the test patients
# (in/lo/hi) and on the training cohort (recalibration factor).
cd /Users/benjaminmaurel/Documents/PharmaNODE
PY=/opt/miniconda3/bin/python3.12
TAG=__noz0_sig0.05_sc141.27_dech50_sel-mse_v2_decres_ep006000.ckpt
# --- paper noise, N=100, seeds 1-3 ---
T=confound_vc00_s4_pnoise_n100; R=results/s4_pnoise_decres
for s in 1 2 3; do while [ ! -f $R/film_n100_s$s/exp_film_run/$T/traj/experiment_film_$T$TAG ]; do sleep 120; done; done
echo "[$(date '+%F %T')] paper-noise seeds at ep6000; scoring"
for s in 1 2 3; do
  NSWEEP_ROOT=$R NSWEEP_BASE=confound_vc00_s4_pnoise NSWEEP_FTAG=_decres scripts/nsweep/eval_nsweep.sh 100 $s 006000 3 | tail -1
done
mkdir -p results/recal/paper_decres
for s in 1 2 3; do
  OMP_NUM_THREADS=1 $PY test_film_matched.py --experiment $T --scale-from $T --eval-split all \
    --ckpt $R/film_n100_s$s/exp_film_run/$T/traj/experiment_film_$T$TAG --label decres_pn_s$s \
    --out-json results/recal/paper_decres/film_s${s}_all.json > logs/recal_decres_pn_s$s.log 2>&1 &
done; wait
echo "DECRES_PNOISE_DONE $(date '+%F %T')"
# --- low noise, N=800, seeds 1-4 ---
T2=confound_vc00_s4; R2=results/s4_decres; mkdir -p $R2/eval results/recal/low_n800_decres
for s in 1 2 3 4; do while [ ! -f $R2/film_s$s/exp_film_run/$T2/traj/experiment_film_$T2$TAG ]; do sleep 300; done; done
echo "[$(date '+%F %T')] low-noise seeds at ep6000; scoring"
for s in 1 2 3 4; do
  CK=$R2/film_s$s/exp_film_run/$T2/traj/experiment_film_$T2$TAG
  for reg in "" _lo _hi; do
    OMP_NUM_THREADS=1 nice -n 10 $PY test_film_matched.py --experiment ${T2}${reg} --scale-from $T2 --ckpt $CK \
      --label decres_ln_s${s}${reg} --out-json $R2/eval/ep006000_film_s${s}_vc00_on_vc00${reg#_}.json \
      > logs/eval_decres_ln_s${s}${reg}.log 2>&1 &
  done
  OMP_NUM_THREADS=1 nice -n 10 $PY test_film_matched.py --experiment $T2 --scale-from $T2 --eval-split all --ckpt $CK \
    --label decres_ln_all_s$s --out-json results/recal/low_n800_decres/film_s${s}_all.json > logs/recal_decres_ln_s$s.log 2>&1 &
  wait
done
echo "DECRES_ALL_DONE $(date '+%F %T')"
