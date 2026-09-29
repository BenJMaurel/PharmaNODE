#!/bin/bash
# MAP-BE rerun with the combined1 residual-error form (Benjamin, 2026-09-27), same Monolix fits as Table 1.
# Phase 1: seeds 1-5 x scenarios 1-3, sequential, ONE low-priority process (runs above the CPU cap for ~30-40 min,
#          agreed use for a quick first look); the first job carries the regression check (aborts the chain on mismatch).
#          Marker C1_FIRST5_DONE.
# Phase 2: waits for SCEN_DEFINIT_DONE (or pid $DEF_PID gone), then the remaining runs on NSLOT cap-gated workers.
#          Marker SCEN_C1_DONE. Restartable (DONE_c1 markers).
cd /Users/benjaminmaurel/Documents/PharmaNODE
CAP=${CAP:-1000}; NSLOT=${NSLOT:-10}; DEF_PID=${DEF_PID:-84040}
say () { echo "[$(date '+%F %T')] $*"; }
cpu () { ps -A -o pcpu=,stat= | awk '$2 !~ /T/ {s+=$1} END {printf "%d", s}'; }
throttle () { while [ $(( $(cpu) + 100 )) -gt "$CAP" ]; do sleep 60; done; }
say "phase 1: seeds 1-5 x scenarios 1-3"
first=1
for s in 1 2 3 4 5; do for sc in 1 2 3; do
  if [ $first = 1 ]; then REGRESS=1 nice -n 10 bash scripts/scen_rerun/run_c1.sh $sc $s || { grep -h REGRESSION results/scen_rerun/runs/s${sc}_seed00${s}/*.log 2>/dev/null; say "phase 1 first job failed -> chain aborted"; exit 1; }; first=0
  else nice -n 10 bash scripts/scen_rerun/run_c1.sh $sc $s; fi
done; done
say "C1_FIRST5_DONE"
say "phase 2: waiting for the default-init redo (SCEN_DEFINIT_DONE / pid $DEF_PID)"
until grep -q SCEN_DEFINIT_DONE logs/chain_scen_definit.log 2>/dev/null || ! kill -0 $DEF_PID 2>/dev/null; do sleep 120; done
JOBS=(); for s in $(seq 6 100); do for sc in 1 2 3; do JOBS+=("$sc $s"); done; done
say "${#JOBS[@]} jobs over $NSLOT slots"
worker () { local k=$1 i; for (( i = k; i < ${#JOBS[@]}; i += NSLOT )); do set -- ${JOBS[$i]}
  [ -f results/scen_rerun/runs/s$1_seed$(printf %03d $2)/DONE_c1 ] && continue; throttle; bash scripts/scen_rerun/run_c1.sh $1 $2; done; }
for (( k = 0; k < NSLOT; k++ )); do worker $k & sleep 20; done
wait
say "finished: $(ls results/scen_rerun/runs/*/DONE_c1 2>/dev/null | wc -l | tr -d ' ') DONE_c1, $(ls results/scen_rerun/runs/*/FAILED_c1_* 2>/dev/null | wc -l | tr -d ' ') FAILED"
say "SCEN_C1_DONE"
