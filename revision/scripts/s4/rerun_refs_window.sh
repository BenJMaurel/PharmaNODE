#!/bin/bash
# Rerun of the three windowed MAP-BE references after the replay-range fix in ebe_popmodel.py (2026-09-24 16:50).
# Relaunched detached (nohup) 17:05, skipping outputs that exist. Monolix fits already exist (fit_and_ebe.sh skips refitting) -> EBE only. Each launch waits for total %cpu + 100 <= CAP.
cd /Users/benjaminmaurel/Documents/PharmaNODE
PY=/opt/miniconda3/bin/python3.12; CAP=${CAP:-1000}; B=confound_vc00_s4_pnoise_win
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cpu () { ps -A -o pcpu=,stat= | awk '$2 !~ /T/ {s+=$1} END {printf "%d", s}'; }
gate () { while [ $(( $(cpu) + 100 )) -gt "$CAP" ]; do sleep 60; done; }
[ -f results/s4/ebe/popebe_mlx_pnoise_win_n100_n1000.csv ] || { gate; COHORT=$B scripts/monolix/fit_and_ebe.sh 100 2 > logs/monolix_pnoise_win_n100.log 2>&1 & sleep 60; }
[ -f results/s4/ebe/popebe_mlx_l1_pnoise_win_n100_n1000.csv ] || { gate; COHORT=$B VARIANT=l1 scripts/monolix/fit_and_ebe.sh 100 2 > logs/monolix_l1_pnoise_win_n100.log 2>&1 & sleep 60; }
gate; EBE_COHORT=$B nice -n 5 $PY scripts/monolix/ebe_popmodel.py true 1000 0 results/s4/ebe/popebe_true_pnoise_win_n1000.csv \
  > logs/popebe_true_pnoise_win.log 2>&1 &
wait
ls results/s4/ebe | grep pnoise_win
echo REFS_WIN_DONE
