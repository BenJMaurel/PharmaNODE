#!/bin/bash
# CPU-budgeted launcher for the results/s4_nsweep2 sweep (control cohort, 6000 epochs,
#   sched_nsweep2.sh [cap=800] [seeds="1"]
# 1 thread per run).  Starts one run at a time, longest first, while the machine's total CPU
# (sum of ps %cpu over non-stopped processes) + 100 <= CAP (default 800 = 8 cores).
# Seeds 2-3 of results/s4_n12000 stay paused (Benjamin, 2026-09-21); they use no CPU.
cd "$(dirname "$0")/../.."
CAP=${1:-800}; SEEDS=${2:-1}
JOBS=()
for sd in $SEEDS; do for j in "film 400" "dc 400" "film 200" "dc 200" "film 100" "dc 100"; do JOBS+=("$j $sd"); done; done
say () { echo "[$(date '+%F %T')] $*"; }
cpu () { ps -A -o pcpu=,stat= | awk '$2 !~ /T/ {s+=$1} END {printf "%d", s}'; }
for j in "${JOBS[@]}"; do
  set -- $j; arch=$1; n=$2; sd=$3
  while :; do c=$(cpu); [ $((c + 100)) -le "$CAP" ] && break; sleep 60; done
  say "cpu ${c}% -> launching $arch N=$n seed $sd"
  NSWEEP_ROOT=results/s4_nsweep2 NSWEEP_ARCHS=$arch scripts/nsweep/launch_nsweep.sh "$n" "$sd" 6000 1
  sleep 120   # let the new run's %cpu register before measuring again
done
say "all ${#JOBS[@]} launched"
