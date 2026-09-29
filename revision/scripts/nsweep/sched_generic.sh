#!/bin/bash
# CPU-budgeted launcher for any sweep (generalises sched_nsweep2.sh, which is left untouched).
#   NSWEEP_ROOT=<root> NSWEEP_BASE=<base cohort> sched_generic.sh <cap> "<seeds>" "<sizes>" [niters]
# One run at a time, largest N first, while total %cpu of non-stopped processes + 100 <= cap.
cd "$(dirname "$0")/../.."
CAP=${1:-800}; SEEDS=${2:-1}; SIZES=${3:-"400 200 100"}; NIT=${4:-6000}
say () { echo "[$(date '+%F %T')] $*"; }
cpu () { ps -A -o pcpu=,stat= | awk '$2 !~ /T/ {s+=$1} END {printf "%d", s}'; }
JOBS=(); for sd in $SEEDS; do for n in $SIZES; do for a in film dc; do JOBS+=("$a $n $sd"); done; done; done
for j in "${JOBS[@]}"; do
  set -- $j; arch=$1; n=$2; sd=$3
  while :; do c=$(cpu); [ $((c + 100)) -le "$CAP" ] && break; sleep 60; done
  say "cpu ${c}% -> launching $arch N=$n seed $sd (root ${NSWEEP_ROOT:-?}, base ${NSWEEP_BASE:-confound_vc00_s4})"
  NSWEEP_ARCHS=$arch scripts/nsweep/launch_nsweep.sh "$n" "$sd" "$NIT" 1
  sleep 120
done
say "all ${#JOBS[@]} launched"
