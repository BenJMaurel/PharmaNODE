#!/bin/bash
# Full big-held-out-set read-out: select on selA, report on selB, both arms, all
# three dose regimes, scored on nRMSE.
set -u
cd "$(dirname "$0")"
SP=/private/tmp/claude-501/-Users-benjaminmaurel-Documents-PharmaNODE/51888621-a648-4228-87a5-9e1b96bf4841/scratchpad
LOG=results/holdout_pipeline.log
say () { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
OUT=results/confound_eval/cells_bigholdout.tsv

say "waiting for cohort generation"
while pgrep -f gen_tacro_confound.py >/dev/null; do sleep 60; done
say "building 500/500 splits"
python3 scripts/build_holdout_splits.py 2>&1 | tee -a "$LOG" || { say "SPLIT FAILED"; exit 1; }

for ARM in wide narrow; do
  say "=== $ARM : selection pass ==="
  python3 scripts/confound_eval/holdout_select.py "$ARM" > "$SP/sel_$ARM.tsv" 2>>"$LOG" \
    || { say "$ARM select build FAILED"; exit 1; }
  rm -f "$SP"/s_${ARM}_*
  split -l 48 "$SP/sel_$ARM.tsv" "$SP/s_${ARM}_"
  for f in "$SP"/s_${ARM}_*; do
    ./scripts/confound_eval/run_jobs.sh "$f" "$OUT" 9 0 2>&1 | tail -1 | tee -a "$LOG"
  done
  say "=== $ARM : picking checkpoints and reporting on selB ==="
  python3 scripts/confound_eval/holdout_report.py "$ARM" "$OUT" > "$SP/rep_$ARM.tsv" 2>>"$LOG" \
    || { say "$ARM report build FAILED"; exit 1; }
  rm -f "$SP"/r_${ARM}_*
  split -l 48 "$SP/rep_$ARM.tsv" "$SP/r_${ARM}_"
  for f in "$SP"/r_${ARM}_*; do
    ./scripts/confound_eval/run_jobs.sh "$f" "$OUT" 9 0 2>&1 | tail -1 | tee -a "$LOG"
  done
  say "=== $ARM done ==="
done
say "holdout pipeline complete"
