#!/bin/bash
# FiLM with a LINEAR decoder, LOW noise, N=800 (confound_vc00_s4), seeds 1-4: score ep6000 on the control
# test patients in range and out of range, and on the training cohort (recalibration factor).
cd /Users/benjaminmaurel/Documents/PharmaNODE
T=confound_vc00_s4; F=__noz0_sig0.05_sc141.27_sel-mse_v2_ep006000.ckpt; R=results/s4_dech0
PY=/opt/miniconda3/bin/python3.12; mkdir -p $R/eval results/recal/low_n800_dech0
for s in 1 2 3 4; do while [ ! -f $R/film_s$s/exp_film_run/$T/traj/experiment_film_$T$F ]; do sleep 300; done; done
echo "[$(date '+%F %T')] all seeds at ep6000; scoring"
for s in 1 2 3 4; do
  CK=$R/film_s$s/exp_film_run/$T/traj/experiment_film_$T$F
  for reg in "" _lo _hi; do
    OMP_NUM_THREADS=1 nice -n 10 $PY test_film_matched.py --experiment ${T}${reg} --scale-from $T --ckpt $CK \
      --label dech0_s${s}${reg} --out-json $R/eval/ep006000_film_s${s}_vc00_on_vc00${reg#_}.json \
      > logs/eval_dech0_ln_s${s}${reg}.log 2>&1 &
  done
  OMP_NUM_THREADS=1 nice -n 10 $PY test_film_matched.py --experiment $T --scale-from $T --eval-split all --ckpt $CK \
    --label dech0_all_s$s --out-json results/recal/low_n800_dech0/film_s${s}_all.json > logs/recal_dech0_ln_s$s.log 2>&1 &
  wait
done
echo "DECH0_LOWNOISE_DONE $(date '+%F %T')"
