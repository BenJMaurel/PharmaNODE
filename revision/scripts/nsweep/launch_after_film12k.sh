#!/bin/bash
# Waits for the FiLM seed-1/4 12k runs to exit, then launches the patient-count sweep
# (control cohort only: confound_vc00_s4_n{100,200,400}) into a FRESH root so the
# killed 13:00 partial runs in results/s4_nsweep are not overwritten.
# Seeds 2-3 of results/s4_n12000 stay PAUSED (Benjamin, 2026-09-21) -- never resumed here.
cd "$(dirname "$0")/../.."
say () { echo "[$(date '+%F %T')] $*"; }
PIDS="73183 73202 73309"   # FiLM s1 vc00, s1 vc09, s4 vc09 (s4 vc00 = 73289 already exited)
say "waiting for FiLM 12k pids: $PIDS"
while ps -p ${PIDS// /,} >/dev/null 2>&1; do sleep 60; done
T=__noz0_sig0.05_sc141.27_dech50_sel-mse_v2_ep012000.ckpt
for s in 1 4; do for c in vc00 vc09; do
  f=results/s4_n12000/film_s$s/exp_film_run/confound_${c}_s4/traj/experiment_film_confound_${c}_s4$T
  [ -f "$f" ] && say "ok ckpt film s$s $c ep12000" || say "WARNING missing $f"
done; done
say "launching sweep (seed 1, 6000 epochs, 1 thread) into results/s4_nsweep2"
NSWEEP_ROOT=results/s4_nsweep2 scripts/nsweep/launch_nsweep.sh "100 200 400" "1" 6000 1
say "launcher done"
