#!/bin/bash
# Launch dose-cond + residual decoder, LOW noise, N=800 (confound_vc00_s4), seeds 1-4, one at a time while
# total %cpu (non-stopped processes) + 100 <= 800.
cd /Users/benjaminmaurel/Documents/PharmaNODE
cpu () { ps -A -o pcpu=,stat= | awk '$2 !~ /T/ {s+=$1} END {printf "%d", s}'; }
C=confound_vc00_s4; : > logs/dc_decres_ln.pids
for s in 1 2 3 4; do
  while [ $(( $(cpu) + 100 )) -gt 800 ]; do sleep 120; done
  R=results/s4_decres/dc_s$s; mkdir -p $R/exp_dosecond_run/$C
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup nice -n 5 /opt/miniconda3/bin/python3.12 train_dose_cond.py --experiment $C \
    --data-dir ./results/exp_film_run --niters 6000 -b 512 -l 10 --lr 1e-2 --seed $s --patience 1000000 --obsrv-std 0.05 \
    --static-dim 4 --decoder-hidden 50 --decoder-residual --save ./$R/ --eval-every 300 --ckpt-every 300 \
    --log-suffix "__decres_ln_s$s" > logs/train_dc_decres_ln_s$s.log 2>&1 &
  echo "$!" >> logs/dc_decres_ln.pids; echo "[$(date '+%F %T')] launched dc decres low-noise seed $s (pid $!)"
  sleep 120
done
