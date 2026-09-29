#!/bin/bash
# Generate the 6 windowed cohorts one at a time while total %cpu + 100 <= 800.
cd "$(dirname "$0")/../.."
cpu () { ps -A -o pcpu=,stat= | awk '$2 !~ /T/ {s+=$1} END {printf "%d", s}'; }
for spec in "gauss||" "tslow||" "gauss|_lo|0.25,0.5" "tslow|_lo|0.25,0.5" "gauss|_hi|10,12" "tslow|_hi|10,12"; do
  IFS='|' read -r sc sfx grid <<< "$spec"
  while [ $(( $(cpu) + 100 )) -gt 800 ]; do sleep 60; done
  scripts/s4/gen_window.sh "$sc" "$sfx" $grid > logs/gen_window_${sc}${sfx}.log 2>&1 &
  echo "[$(date '+%F %T')] started $sc$sfx (pid $!)"; sleep 90
done
wait; echo "GEN_WINDOW_DONE $(date '+%F %T')"
