#!/bin/bash
# Suspend / resume the training jobs with SIGSTOP / SIGCONT.
#   ./pause_runs.sh pause    freeze every run_models.py / train_dose_cond.py
#   ./pause_runs.sh resume   thaw them and record how long they were frozen
#   ./pause_runs.sh status   show state of each job
# Processes stay resident in RAM while paused; they do NOT survive a reboot,
# a logout, or a kill.  Elapsed wall-clock is therefore no longer a valid
# measure of training rate: results/pause_ledger.txt accumulates the frozen
# time so ETAs can be corrected.
set -u
cd /Users/benjaminmaurel/Documents/PharmaNODE
LEDGER=results/pause_ledger.txt
PIDFILE=results/paused_pids.txt
pids () { pgrep -f 'run_models\.py|train_dose_cond\.py'; }

case "${1:?usage: ./pause_runs.sh pause|resume|status}" in
  pause)
    p=$(pids); n=$(echo "$p" | grep -c . )
    [ "$n" -eq 0 ] && { echo "no training jobs running"; exit 0; }
    echo "$p" > "$PIDFILE"
    for x in $p; do kill -STOP "$x" 2>/dev/null; done
    echo "PAUSED_AT $(date +%s)  $(date '+%F %T')  n=$n" >> "$LEDGER"
    echo "paused $n job(s) at $(date '+%F %T')"
    ;;
  resume)
    p=$(pids); n=0
    for x in $p; do kill -CONT "$x" 2>/dev/null && n=$((n+1)); done
    start=$(grep '^PAUSED_AT' "$LEDGER" 2>/dev/null | tail -1 | awk '{print $2}')
    if [ -n "${start:-}" ]; then
      d=$(( $(date +%s) - start ))
      echo "RESUMED_AT $(date +%s)  $(date '+%F %T')  frozen_s=$d" >> "$LEDGER"
      printf "resumed %d job(s); frozen for %dh %02dm\n" "$n" $((d/3600)) $(( (d%3600)/60 ))
    else
      echo "resumed $n job(s) (no pause record found)"
    fi
    ;;
  status)
    printf "%-8s %-6s %s\n" "PID" "STATE" "job"
    for x in $(pids); do
      st=$(ps -o state= -p "$x" | tr -d ' ')
      tag=$(ps -o command= -p "$x" | grep -oE '\-\-log-suffix [^ ]+' | awk '{print $2}')
      scr=$(ps -o command= -p "$x" | grep -oE '(run_models|train_dose_cond)\.py' | head -1)
      printf "%-8s %-6s %s %s\n" "$x" "$st" "$scr" "$tag"
    done
    echo
    echo "(T = stopped/paused, R/S = running)"
    grep -E '^(PAUSED|RESUMED)_AT' results/pause_ledger.txt 2>/dev/null | tail -4
    ;;
  *) echo "usage: ./pause_runs.sh pause|resume|status"; exit 1 ;;
esac
