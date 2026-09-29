#!/bin/bash
# After the latent-ODE test scoring: score the TRAINING series of each checkpoint (recalibration factor).
cd /Users/benjaminmaurel/Documents/PharmaNODE
until grep -q LODE_EVAL_DONE logs/chain_lode_eval.log; do sleep 30; done
E=s4pn_lode_n100
for s in 1 2 3; do for k in "" _best; do
  OMP_NUM_THREADS=1 /opt/miniconda3/bin/python3.12 scripts/repro/eval_lode_train.py results/lode_s4pn/s$s/$E/experiment_$E$k.ckpt \
    results/lode_s4pn/eval/s${s}${k:-_final}_train.json > logs/eval_lode_train_s${s}${k:-_final}.log 2>&1 &
done; done; wait
echo "LODE_TRAIN_DONE $(date '+%F %T')"
