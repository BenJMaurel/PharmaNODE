#!/bin/bash
# FiLM with a LINEAR decoder (full objective), paper noise N=100: score at ep6000 on the test patients
# (in/lo/hi) and on the training cohort (recalibration factor).
cd /Users/benjaminmaurel/Documents/PharmaNODE
T=confound_vc00_s4_pnoise_n100; F=__noz0_sig0.05_sc141.27_sel-mse_v2_ep006000.ckpt; R=results/s4_pnoise_dech0
for s in 1 2 3; do while [ ! -f $R/film_n100_s$s/exp_film_run/$T/traj/experiment_film_$T$F ]; do sleep 120; done; done
echo "[$(date '+%F %T')] all seeds at ep6000; scoring"
for s in 1 2 3; do
  NSWEEP_ROOT=$R NSWEEP_BASE=confound_vc00_s4_pnoise NSWEEP_DECHTAG= scripts/nsweep/eval_nsweep.sh 100 $s 006000 3 | tail -1
done
mkdir -p results/recal/paper_dech0
for s in 1 2 3; do
  OMP_NUM_THREADS=1 /opt/miniconda3/bin/python3.12 test_film_matched.py --experiment $T --scale-from $T --eval-split all \
    --ckpt $R/film_n100_s$s/exp_film_run/$T/traj/experiment_film_$T$F --label dech0_s$s \
    --out-json results/recal/paper_dech0/film_s${s}_all.json > logs/recal_dech0_s$s.log 2>&1 &
done; wait
echo "DECH0_DONE $(date '+%F %T')"
