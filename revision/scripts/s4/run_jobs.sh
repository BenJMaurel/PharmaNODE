#!/bin/bash
# Run the shell commands in <jobfile>, <maxpar> at a time.
cd "$(dirname "$0")/../.."
JF=$1; MAXPAR=${2:-4}
while IFS= read -r cmd; do
  [ -z "$cmd" ] && continue
  while [ "$(jobs -rp | wc -l)" -ge "$MAXPAR" ]; do sleep 5; done
  bash -c "$cmd" &
done < "$JF"
wait; echo "JOBS_DONE $(date '+%F %T')"
