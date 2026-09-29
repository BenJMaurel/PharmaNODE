#!/bin/bash
# Overnight chain, in priority order.
#
#  1. wait for the 1000-patient cohorts to finish generating
#  2. split each into a 500-patient selection half and a 500-patient reporting half
#  3. leakage-free read-out: pick each run's checkpoint on the selection half,
#     score it on the reporting half (the third independent read-out)
#  4. dose-cond with dense checkpoints -- the headline DiD (+9.46 vs +5.05) still
#     reads dose-cond at ONE epoch while the FiLM side has an 11-point median, and
#     single-epoch reads moved by up to 3.7 pp today. This puts both sides level.
set -u
cd "$(dirname "$0")"
SP=/private/tmp/claude-501/-Users-benjaminmaurel-Documents-PharmaNODE/51888621-a648-4228-87a5-9e1b96bf4841/scratchpad
say () { echo "[$(date '+%F %T')] $*" | tee -a results/overnight.log; }

say "step 1: waiting for cohort generation"
while pgrep -f gen_tacro_confound.py >/dev/null; do sleep 60; done
say "step 1 done"

say "step 2: building 500/500 splits"
python3 scripts/build_holdout_splits.py 2>&1 | tee -a results/overnight.log || { say "SPLIT FAILED"; exit 1; }

say "step 3a: selection pass (every checkpoint scored on selA)"
python3 scripts/confound_eval/build_holdout_jobs.py > "$SP/jobs_selA.tsv" 2>>results/overnight.log \
  || { say "JOB BUILD FAILED"; exit 1; }
say "  $(wc -l < "$SP/jobs_selA.tsv") selection evaluations"
rm -f "$SP"/hchunk_*
split -l 60 "$SP/jobs_selA.tsv" "$SP/hchunk_"
for f in "$SP"/hchunk_*; do
  ./scripts/confound_eval/run_jobs.sh "$f" results/confound_eval/cells_holdout.tsv 9 0 2>&1 | tail -1 | tee -a results/overnight.log
done
say "step 3b: picking checkpoints and scoring them on selB"
python3 scripts/confound_eval/pick_and_report.py > "$SP/jobs_selB.tsv" 2>>results/overnight.log \
  || { say "PICK FAILED"; exit 1; }
./scripts/confound_eval/run_jobs.sh "$SP/jobs_selB.tsv" results/confound_eval/cells_holdout.tsv 9 0 2>&1 | tail -1 | tee -a results/overnight.log
say "step 3 done"

say "step 4: dose-cond dense retrain (seeds 1-6)"
./queue_dosecond_dense.sh 1 2 3 4 5 6 >>results/overnight.log 2>&1
say "step 4 done -- overnight chain complete"
