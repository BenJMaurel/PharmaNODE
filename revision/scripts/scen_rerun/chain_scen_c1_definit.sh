#!/bin/bash
# Combined1 (form-fixed) MAP-BE from the DEFAULT-INIT fits (Benjamin, 27/09 19:40: "use the form fix but run it with the
# default initialisation now"), then from the published-init fits (Table 1's) for the runs not done yet.
# Phase A: NSLOT cap-gated workers claim runs whose default-init fit exists (DONE_definit, or a FAILED_definit_mapbe_*
#          run whose Monolix fit completed) and rescan every 2 min while the redo (pid $DEF_PID) still produces fits;
#          ends when the redo is done (SCEN_DEFINIT_DONE / pid gone) and nothing is left. Marker C1_DEFINIT_DONE.
# Phase B: FIT=published for runs without DONE_c1. Marker SCEN_C1_ALL_DONE. Claims: results/scen_rerun/c1_claims/.
cd /Users/benjaminmaurel/Documents/PharmaNODE
CAP=${CAP:-1000}; NSLOT=${NSLOT:-10}; DEF_PID=${DEF_PID:-84040}; RUNS=results/scen_rerun/runs; CL=results/scen_rerun/c1_claims
mkdir -p $CL
say () { echo "[$(date '+%F %T')] $*"; }
cpu () { ps -A -o pcpu=,stat= | awk '$2 !~ /T/ {s+=$1} END {printf "%d", s}'; }
throttle () { while [ $(( $(cpu) + 100 )) -gt "$CAP" ]; do sleep 15; done; }
redo_done () { grep -q SCEN_DEFINIT_DONE logs/chain_scen_definit.log 2>/dev/null || ! kill -0 $DEF_PID 2>/dev/null; }
ORDER=(); for s in $(seq 1 100); do for sc in 1 2 3; do ORDER+=("$sc $s"); done; done
ready_definit () { local d=$RUNS/s$1_seed$(printf %03d $2)
  [ -f $d/DONE ] || return 1; [ -f $d/DONE_definit_c1 ] && return 1; ls $d/FAILED_definit_c1_* >/dev/null 2>&1 && return 1
  [ -f $d/DONE_definit ] && return 0; ls $d/FAILED_definit_mapbe_* >/dev/null 2>&1 && return 0; return 1; }
workerA () { while :; do local found=0 j
  for j in "${ORDER[@]}"; do set -- $j; ready_definit $1 $2 || continue
    mkdir $CL/definit_s$1_$2 2>/dev/null || continue; found=1; throttle; FIT=definit bash scripts/scen_rerun/run_c1.sh $1 $2; done
  [ $found = 0 ] && { redo_done && break; sleep 120; }
done; }
workerB () { local j; for j in "${ORDER[@]}"; do set -- $j; local d=$RUNS/s$1_seed$(printf %03d $2)
  [ -f $d/DONE ] && [ ! -f $d/DONE_c1 ] || continue; ls $d/FAILED_c1_* >/dev/null 2>&1 && continue
  mkdir $CL/published_s$1_$2 2>/dev/null || continue; throttle; FIT=published bash scripts/scen_rerun/run_c1.sh $1 $2; done; }
say "phase A: form-fixed MAP-BE from the default-init fits ($NSLOT workers, cap-gated)"
for (( k = 0; k < NSLOT; k++ )); do workerA & sleep 10; done; wait
say "phase A finished: $(ls $RUNS/*/DONE_definit_c1 2>/dev/null | wc -l | tr -d ' ') DONE_definit_c1, $(ls $RUNS/*/FAILED_definit_c1_* 2>/dev/null | wc -l | tr -d ' ') FAILED"
say "C1_DEFINIT_DONE"
say "phase B: form-fixed MAP-BE from the published-init fits (remaining runs)"
for (( k = 0; k < NSLOT; k++ )); do workerB & sleep 10; done; wait
say "phase B finished: $(ls $RUNS/*/DONE_c1 2>/dev/null | wc -l | tr -d ' ') DONE_c1, $(ls $RUNS/*/FAILED_c1_* 2>/dev/null | wc -l | tr -d ' ') FAILED"
say "SCEN_C1_ALL_DONE"
