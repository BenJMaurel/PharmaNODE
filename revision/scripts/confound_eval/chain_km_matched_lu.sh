#!/bin/bash
# Lu et al. arm of the matched Km confounding rerun (companion of chain_km_matched.sh), 2026-09-24.
# Variant = "same task": static covariates in the encoder (scenario 3: dose, formulation, CYP; --static-dim 3)
# + cross-visit loss (--train-mode counterfactual). The original-training variant is not run: its dose port is
# nearly inert (elasticity 0.23 vs 1.43 at N=100), so a confounding contrast on it carries no information.
# Same cohorts (confound_km09 / km00), seeds 1-3, 3000 epochs (= 6000 optimiser steps, as the N=100 Lu runs),
# traj every 300, port defaults otherwise (lr 1e-3, history init, 6 run-in doses). Scored at ep2400/2700/3000 on
# km09, km00, km09_big, km00_big with the scaler pinned to the training cohort, into the SAME eval dir as the
# FiLM / dose-cond arms (results/km_matched/eval/ep<EP>_lu_s<seed>_<tr>_on_<ev>.json), then the summary is rebuilt.
# Log: logs/chain_km_matched_lu.log. Do not edit while running.
cd /Users/benjaminmaurel/Documents/PharmaNODE
PY=/opt/miniconda3/bin/python3.12
ROOT=results/km_matched_lu; E=results/km_matched/eval
SEEDS="${SEEDS:-1 2 3}"; NIT=${NIT:-3000}; MAXJ=${MAXJ:-6}; EPS="${EPS:-002400 002700 003000}"
mkdir -p $E logs
say () { echo "[$(date '+%F %T')] $*"; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
throttle () { while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 20; done; }

: > logs/km_matched_lu.pids
for s in $SEEDS; do
  for c in confound_km09 confound_km00; do
    throttle
    R=$ROOT/s$s; mkdir -p $R
    nohup nice -n 5 $PY -W ignore train_lu_pk.py --experiment $c --data-dir ./results/exp_film_run \
      --niters $NIT -b 512 --seed $s --save ./$R/ --static-dim 3 --use-static --train-mode counterfactual \
      --ckpt-every 300 --eval-every 300 > logs/train_kmm_lu_${c#confound_}_s$s.log 2>&1 &
    echo "$!" >> logs/km_matched_lu.pids
    say "launched lu $c seed $s (pid $!)"
    sleep 3
  done
done
wait
say "training finished"

n_ok=0; n_miss=0
for ep in $EPS; do
  for s in $SEEDS; do
    for tr in confound_km09 confound_km00; do
      CK=$(ls $ROOT/s$s/exp_lupk_run/$tr/traj/*_ep$ep.ckpt 2>/dev/null)
      if [ "$(echo "$CK" | grep -c ckpt)" != 1 ]; then say "MISSING/AMBIGUOUS ckpt lu s$s $tr ep$ep: '$CK'"; n_miss=$((n_miss+1)); continue; fi
      for ev in confound_km09 confound_km00 confound_km09_big confound_km00_big; do
        out=$E/ep${ep}_lu_s${s}_${tr#confound_}_on_$(echo ${ev#confound_} | tr -d _).json
        [ -f $out ] && continue
        throttle
        nice -n 5 $PY scripts/confound_eval/seeded_eval.py --eval-seed 0 test_lu_pk.py --experiment $ev \
          --data-dir ./results/exp_film_run --ckpt $CK --scale-from $tr --eval-split test \
          --label kmm_lu_s${s}_${tr#confound_}_on_${ev#confound_}_ep$ep --out-json $out \
          > logs/eval_kmm_lu_s${s}_${tr#confound_}_on_${ev#confound_}_ep$ep.log 2>&1 &
        n_ok=$((n_ok+1))
      done
    done
  done
done
wait
say "scored ($n_ok evaluations launched, $n_miss missing checkpoints)"
$PY scripts/confound_eval/km_matched_did.py > results/km_matched/did_summary.txt 2>&1 && say "summary rebuilt -> results/km_matched/did_summary.txt"
say "KM_LU_DONE"
