#!/bin/bash
# Stage seed 1 (redo, protocol-matched) then seed 3. Each stage = 8 cells,
# stop at 6000 epochs, checkpoint every 1000.
cd /Users/benjaminmaurel/Documents/PharmaNODE
LOG=results/staged_seeds.log
say () { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a $LOG; }

wait_for () {   # $1 = log-suffix fragment identifying this stage
  local pat="$1" n
  while true; do
    n=$(ps -eo command | grep -E "[r]un_models.py|[t]rain_dose_cond.py" | grep -c -- "$pat")
    [ "$n" -eq 0 ] && break
    sleep 120
  done
}

say "=== STAGE 1: seed 1 redo (wide + narrow) ==="
./launch_seed.sh 1 both >> $LOG 2>&1
sleep 30
say "stage 1 running: $(ps -eo command | grep -cE '[r]un_models.py|[t]rain_dose_cond.py') jobs"
wait_for "__s1"
say "=== STAGE 1 COMPLETE ==="
for a in wide narrow; do
  say "  s1$a checkpoints: $(find results/seeds/s1$a -name '*_ep*.ckpt' 2>/dev/null | wc -l | tr -d ' ')"
done

say "=== STAGE 2: seed 3 (wide + narrow) ==="
./launch_seed.sh 3 both >> $LOG 2>&1
sleep 30
say "stage 2 running: $(ps -eo command | grep -cE '[r]un_models.py|[t]rain_dose_cond.py') jobs"
wait_for "__s3"
say "=== STAGE 2 COMPLETE ==="
for a in wide narrow; do
  say "  s3$a checkpoints: $(find results/seeds/s3$a -name '*_ep*.ckpt' 2>/dev/null | wc -l | tr -d ' ')"
done
say "=== ALL STAGES DONE ==="
