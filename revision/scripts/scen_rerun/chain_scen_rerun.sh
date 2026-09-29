#!/bin/bash
# NMI 3-scenario rerun (Benjamin, 2026-09-25): 3 scenarios x 100 seeds, queued behind the N=800 Vc chain (see GATE below).
# Then a 20-epoch smoke run through every step (aborts the chain if anything is missing), then 10 slot workers
# (worktrees results/paper_repro/w01..w10 of HEAD 2eaadf6), each taking every 10th job of the list
# seed 1..100 x scenario 1,2,3 (interleaved, so partial results cover all scenarios). Each job waits for
# total %cpu + 100 <= CAP. Restartable: finished runs (DONE) are skipped. Summary: scripts/scen_rerun/summarise.py.
# Log logs/chain_scen_rerun.log; per-run outputs results/scen_rerun/runs/. Markers SCEN_SMOKE_OK, SCEN_RERUN_DONE.
cd /Users/benjaminmaurel/Documents/PharmaNODE
CAP=${CAP:-1000}; NSLOT=${NSLOT:-10}; CHAIN_PID=${CHAIN_PID:-20198}; SEEDS=${SEEDS:-$(seq 1 100)}
RUN=scripts/scen_rerun/run_one.sh
say () { echo "[$(date '+%F %T')] $*"; }
cpu () { ps -A -o pcpu=,stat= | awk '$2 !~ /T/ {s+=$1} END {printf "%d", s}'; }
throttle () { while [ $(( $(cpu) + 100 )) -gt "$CAP" ]; do sleep 60; done; }
mkdir -p results/scen_rerun/runs
# GATE=launched (default since 2026-09-25 10:40, after the N=800 Lu runs were stopped): start once the N=800 chain has
# LAUNCHED its last run (dc vc00 seed 3), then take only spare CPU (every job is cap-gated). GATE=done: wait for
# VCW_MATCHED_DONE (the original behaviour). Either gate also opens if the N=800 chain (pid $CHAIN_PID) is gone.
GATE=${GATE:-launched}
if [ "$GATE" = launched ]; then
  say "waiting for the N=800 chain to launch its last run (or pid $CHAIN_PID gone)"
  until grep -q "launched dc confound_vc00_win_n800 seed 3" logs/chain_vcw_matched.log 2>/dev/null || ! kill -0 $CHAIN_PID 2>/dev/null; do sleep 120; done
else
  say "waiting for the N=800 chain (VCW_MATCHED_DONE or pid $CHAIN_PID gone)"
  until grep -q VCW_MATCHED_DONE logs/chain_vcw_matched.log 2>/dev/null || ! kill -0 $CHAIN_PID 2>/dev/null; do sleep 120; done
  grep -q VCW_MATCHED_DONE logs/chain_vcw_matched.log || say "WARNING: N=800 chain ended WITHOUT VCW_MATCHED_DONE (starting anyway)"
fi
say "gate open -> smoke test (scenario 1, seed 0, 20 epochs, slot w01)"
throttle
NITERS=20 bash $RUN results/paper_repro/w01 1 0
S=results/scen_rerun/runs/s1_seed000_smoke
for f in DONE virtual_cohort_test.csv tacro_mapbayest_auc_sigvar.csv tacro_mapbayest_auc_sigsd.csv populationParameters.txt lode_final.json lode_best.json; do
  [ -s $S/$f ] || [ $f = DONE -a -f $S/DONE ] || { say "SMOKE FAILED: $S/$f missing -> chain aborted"; exit 1; }
done
say "SCEN_SMOKE_OK"
JOBS=(); for s in $SEEDS; do for sc in 1 2 3; do JOBS+=("$sc $s"); done; done
say "${#JOBS[@]} jobs over $NSLOT slots"
worker () {  # slot index 0..NSLOT-1
  local k=$1 w=results/paper_repro/w$(printf %02d $(( $1 + 1 ))) i
  for (( i = k; i < ${#JOBS[@]}; i += NSLOT )); do
    set -- ${JOBS[$i]}
    [ -f results/scen_rerun/runs/s$1_seed$(printf %03d $2)/DONE ] && continue
    throttle
    bash $RUN $w $1 $2
  done
}
for (( k = 0; k < NSLOT; k++ )); do worker $k & sleep 60; done
wait
n_ok=$(ls results/scen_rerun/runs/*/DONE 2>/dev/null | grep -vc smoke); n_fail=$(ls results/scen_rerun/runs/*/FAILED_* 2>/dev/null | wc -l)
say "finished: $n_ok DONE, $n_fail FAILED"
/opt/miniconda3/bin/python3.12 scripts/scen_rerun/summarise.py > results/scen_rerun/summary.txt 2>&1
say "summary -> results/scen_rerun/summary.txt"
say "SCEN_RERUN_DONE"
