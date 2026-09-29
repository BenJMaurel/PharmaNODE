#!/bin/bash
# Windowed versions of the scenario-3 Vc confounding cohorts (Benjamin, 2026-09-24; convention = handoff 1.12).
# Identical to the originals (handoff §9.2: scenario 3, Vc confounder, dose grid 1-8 mg, seed 0 / dose-seed 1234,
# default residual noise) except --v1-cavg-window 3,25 and an oversized draw. Dose assignment pinned to the ORIGINAL
# cohort's z-scale (--zstats-from confound_vc09 / vc00), so the confounded dose rule is the published one.
# Measured on the existing cohorts: pass rate 51% (vc00) / 63% (vc09); the window induces corr(d1, log Vc) ~ +0.49 in
# the control arm (results/vc_matched_window_check.txt). Training cohorts: 2200 drawn (test fraction 0.2) -> ~900 /
# ~1100 kept train, subset to exactly 800 afterwards; held-out _big: 2100 drawn, test only, seed 777, ids +10000.
# Launches when total %cpu + 100 <= CAP. Logs logs/gen_vc_win_<cohort>.log; marker GEN_VC_WIN_DONE in logs/gen_vc_win.log.
cd /Users/benjaminmaurel/Documents/PharmaNODE
PY=/opt/miniconda3/bin/python3.12; CAP=${CAP:-1000}
cpu () { ps -A -o pcpu=,stat= | awk '$2 !~ /T/ {s+=$1} END {printf "%d", s}'; }
say () { echo "[$(date '+%F %T')] $*"; }
COMMON="--confound-param Vc --scenario 3 --dose-grid 1,2,3,4,5,6,7,8 --dose-seed 1234 --v1-cavg-window 3,25"
for spec in "confound_vc09_win|0.9|2200|0.2|0|0|confound_vc09" \
            "confound_vc00_win|0.0|2200|0.2|0|0|confound_vc00" \
            "confound_vc09_big_win|0.9|2100|1.0|777|10000|confound_vc09" \
            "confound_vc00_big_win|0.0|2100|1.0|777|10000|confound_vc00"; do
  IFS='|' read -r exp rho n tf seed off zs <<< "$spec"
  [ -e results/exp_film_run/$exp ] && { say "$exp exists -- skipped"; continue; }
  while [ $(( $(cpu) + 100 )) -gt "$CAP" ]; do sleep 60; done
  OMP_NUM_THREADS=1 nice -n 5 $PY gen_tacro_confound.py --exp $exp --num_patients $n --rho $rho --test-fraction $tf \
    --seed $seed --id-offset $off --zstats-from $zs $COMMON > logs/gen_vc_win_$exp.log 2>&1 &
  say "started $exp (pid $!)"; sleep 90
done
wait
for exp in confound_vc09_win confound_vc00_win; do
  $PY scripts/nsweep/make_subsets.py $exp "800" 0
done
say "GEN_VC_WIN_DONE"
