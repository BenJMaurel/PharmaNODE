#!/bin/bash
# Generate the _lo and _hi paper-noise cohorts one at a time, each only when total CPU < 700%.
cd "$(dirname "$0")/../.."
cpu () { ps -A -o pcpu=,stat= | awk '$2 !~ /T/ {s+=$1} END {printf "%d", s}'; }
for spec in "_lo 0.25,0.5" "_hi 10,12"; do
  set -- $spec
  while [ "$(cpu)" -ge 700 ]; do sleep 60; done
  echo "[$(date '+%F %T')] generating confound_vc00_s4_pnoise$1 (cpu $(cpu)%)"
  OMP_NUM_THREADS=1 scripts/s4/gen_pnoise.sh "$1" "$2" > logs/gen_pnoise$1.log 2>&1 &
  sleep 120
done
wait; echo "[$(date '+%F %T')] OOD generation done"
