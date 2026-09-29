#!/bin/bash
# Evaluate ONE cell -- a checkpoint on one evaluation cohort -- and print one row
# in the cells-table schema (see run_jobs.sh for the header).
#
#   eval_cell.sh <arm> <seed> <epoch> <regime> <model> <trained> <evaluated> <ckpt> [eval_seed=0]
#
# Scale constants (concentration max, Box-Cox lambda, dose max) are pinned to
# the TRAINING cohort via --scale-from, so a cross-cohort or out-of-range
# evaluation sees the transform the model was trained under.  RNGs are seeded
# through seeded_eval.py, so re-running a cell reproduces it exactly.  If any
# metric fails to parse, the script prints nothing to stdout and exits 1: a
# silent blank cell is how a changed report format would corrupt a table.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
[ $# -ge 8 ] || { echo "usage: eval_cell.sh <arm> <seed> <epoch> <regime> <model> <trained> <evaluated> <ckpt> [eval_seed]" >&2; exit 2; }
ARM=$1; SEED=$2; EPOCH=$3; REGIME=$4; MODEL=$5; TR=$6; EV=$7; CK=$8; ESEED=${9:-0}
case "$MODEL" in
  dc|dcd) H=test_dose_cond.py ;;
  film) H=test_film_matched.py ;;
  w2|wr*|filmd) H=test_film_matched.py ;;   # same architecture; --w2-analytic-moments changed training only
  *)    echo "eval_cell: unknown model '$MODEL'" >&2; exit 2 ;;
esac
out=$(python3 "$HERE/seeded_eval.py" --eval-seed "$ESEED" "$H" --experiment "$EV" \
        --data-dir ./results/exp_film_run --ckpt "$CK" --scale-from "$TR" \
        --eval-split test --no-json 2>&1)
rc=$?
g() { echo "$out" | grep -A 5 "VISIT $1" | grep -- "$2" | grep -oE "[-0-9.]+%" | head -1 | tr -d '%'; }
vals=( "$(g 1 'MPE (Bias)')" "$(g 1 'RMSPE (Precision)')" "$(g 1 'RMSPE pointwise')" "$(g 1 'nRMSE')"
       "$(g 2 'MPE (Bias)')" "$(g 2 'RMSPE (Precision)')" "$(g 2 'RMSPE pointwise')" "$(g 2 'nRMSE')" )
for v in "${vals[@]}"; do
  if [ -z "$v" ]; then
    { echo "eval_cell: unparsable output (exit $rc) for $ARM s$SEED ep$EPOCH $REGIME $MODEL $TR -> $EV"
      echo "$out" | tail -15; } >&2
    exit 1
  fi
done
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s" "$SEED" "$EPOCH" "$ARM" "$REGIME" "$MODEL" "$TR" "$EV"
printf "\t%s" "${vals[@]}"
printf "\t1\t%s\t%s\n" "$ESEED" "$CK"
