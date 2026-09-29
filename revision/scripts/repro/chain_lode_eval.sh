#!/bin/bash
# After the plain latent-ODE runs on scenario 4 (paper noise, N=100) finish: score the final (ep6000)
# and the paper-style _best checkpoints of seeds 1-3 on the observed-visit AUC of the 1000 test patients.
cd /Users/benjaminmaurel/Documents/PharmaNODE
while ps -p 30810,30814,30817 >/dev/null 2>&1; do sleep 60; done
E=s4pn_lode_n100; mkdir -p results/lode_s4pn/eval
for s in 1 2 3; do for k in "" _best; do
  OMP_NUM_THREADS=2 /opt/miniconda3/bin/python3.12 scripts/repro/eval_lode.py results/lode_s4pn/s$s/$E/experiment_$E$k.ckpt \
    results/lode_s4pn/eval/s${s}${k:-_final}.json > logs/eval_lode_s4pn_s${s}${k:-_final}.log 2>&1 &
done; done; wait
echo "LODE_EVAL_DONE $(date '+%F %T')"
