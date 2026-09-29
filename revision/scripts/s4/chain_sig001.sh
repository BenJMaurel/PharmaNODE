#!/bin/bash
# FiLM σ=0.01 (sc 3531.68) on paper-noise N=100: score at ep6000 once all 3 seeds reach it, then the
# training-set recalibration, then the tables.
cd /Users/benjaminmaurel/Documents/PharmaNODE
T=confound_vc00_s4_pnoise_n100; F=__noz0_sig0.01_sc3531.68_dech50_sel-mse_v2_ep006000.ckpt
for s in 1 2 3; do
  while [ ! -f results/s4_pnoise_sig001/film_n100_s$s/exp_film_run/$T/traj/experiment_film_$T$F ]; do sleep 120; done
done
echo "[$(date '+%F %T')] all seeds at ep6000; scoring"
for s in 1 2 3; do
  NSWEEP_ROOT=results/s4_pnoise_sig001 NSWEEP_BASE=confound_vc00_s4_pnoise NSWEEP_SIG=0.01 NSWEEP_SC=3531.68 \
    scripts/nsweep/eval_nsweep.sh 100 $s 006000 3 | tail -1
done
/opt/miniconda3/bin/python3.12 scripts/s4/recal_jobs.py > results/recal/jobs_sig001.txt
scripts/s4/run_jobs.sh results/recal/jobs_sig001.txt 3
/opt/miniconda3/bin/python3.12 scripts/s4/recal_tables.py > results/recal/recal_tables.txt 2>&1
echo "SIG001_DONE $(date '+%F %T')"
