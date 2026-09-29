#!/bin/bash
# Plain latent ODE with --decoder-hidden 50 (paper noise, N=100): when the 3 runs exit, score final and _best
# on the test patients (paper convention) and the training series (recalibration factor), then summarise.
cd /Users/benjaminmaurel/Documents/PharmaNODE
while ps -p $(tr -s ' ' ',' < logs/lode_dech50.pids | sed 's/^,//') >/dev/null 2>&1; do sleep 60; done
E=s4pn_lode_n100; mkdir -p results/lode_s4pn_dech50/eval
for s in 1 2 3; do for k in "" _best; do
  CK=results/lode_s4pn_dech50/s$s/$E/experiment_$E$k.ckpt
  OMP_NUM_THREADS=1 /opt/miniconda3/bin/python3.12 scripts/repro/eval_lode.py $CK results/lode_s4pn_dech50/eval/s${s}${k:-_final}.json > logs/eval_lode_dech50_s${s}${k:-_final}.log 2>&1 &
  OMP_NUM_THREADS=1 /opt/miniconda3/bin/python3.12 scripts/repro/eval_lode_train.py $CK results/lode_s4pn_dech50/eval/s${s}${k:-_final}_train.json > logs/eval_lode_dech50_train_s${s}${k:-_final}.log 2>&1 &
done; done; wait
echo "LODE_DECH50_DONE $(date '+%F %T')"
