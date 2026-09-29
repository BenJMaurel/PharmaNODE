#!/bin/bash
# Windowed MAP-BE references on ALL 1100 test patients (Benjamin, 2026-09-25; the _n1000 files covered 1000 of 1100).
# Existing Monolix fits reused (no refit); true model, Monolix true-cov, Monolix L1. Sequential, one slot, CPU-cap gated.
cd /Users/benjaminmaurel/Documents/PharmaNODE
PY=/opt/miniconda3/bin/python3.12; CAP=${CAP:-1000}; B=confound_vc00_s4_pnoise_win
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 EBE_COHORT=$B
cpu () { ps -A -o pcpu=,stat= | awk '$2 !~ /T/ {s+=$1} END {printf "%d", s}'; }
gate () { while [ $(( $(cpu) + 100 )) -gt "$CAP" ]; do sleep 60; done; }
for spec in "true:true_pnoise_win" "results/monolix/fit_pnoise_win_n100/estimates.json:mlx_pnoise_win_n100" \
            "results/monolix/fit_l1_pnoise_win_n100/estimates.json:mlx_l1_pnoise_win_n100"; do
  w=${spec%%:*}; tag=${spec##*:}; out=results/s4/ebe/popebe_${tag}_n1100.csv
  [ -f $out ] && continue
  gate; echo "[$(date '+%F %T')] start $tag"
  nice -n 5 $PY scripts/monolix/ebe_popmodel.py $w 1100 0 $out > logs/popebe_${tag}_n1100.log 2>&1 \
    && echo "[$(date '+%F %T')] done $tag" || echo "[$(date '+%F %T')] FAILED $tag (see logs/popebe_${tag}_n1100.log)"
done
echo REFS_WIN1100_DONE
