#!/bin/bash
# Score one sweep cell at one epoch on the in-range, below- and above-range test cohorts.
#   scripts/nsweep/eval_nsweep.sh <N> <seed> <epoch6> [maxpar]
# --scale-from is the model's OWN training cohort (confound_vc00_s4_n<N>), not the evaluated
# one (landmine 1.2).  All three evaluated cohorts share the 1000 test patients.
set -u
cd "$(dirname "$0")/../.."
N=${1:?N}; S=${2:?seed}; EP=${3:?epoch}; MAXPAR=${4:-2}
ROOT=${NSWEEP_ROOT:-results/s4_nsweep}
BASE=${NSWEEP_BASE:-confound_vc00_s4}
PY="/opt/miniconda3/bin/python3.12 -u"; OUT=$ROOT/eval; mkdir -p "$OUT"
TR=${BASE}_n$N
SIG=${NSWEEP_SIG:-0.05}; SC=${NSWEEP_SC:-141.27}
FTAG=${NSWEEP_FTAG:-}   # extra FiLM tag, e.g. _factonly
DECHTAG=${NSWEEP_DECHTAG-_dech50}   # set to "" for a linear decoder (decoder_hidden 0 leaves no tag)
FT=__noz0_sig${SIG}_sc${SC}${DECHTAG}_sel-mse_v2${FTAG}_ep${EP}.ckpt; DT=__sig-${SIG}_ep${EP}.ckpt
for reg in in lo hi; do
  EV=$BASE; [ $reg != in ] && EV=${EV}_$reg
  [ -f results/exp_film_run/$EV/virtual_cohort_film_test.csv ] || { echo "missing cohort $EV"; continue; }
  for arch in film dc; do
    if [ $arch = film ]; then SC=test_film_matched.py; CK=$ROOT/film_n${N}_s$S/exp_film_run/$TR/traj/experiment_film_$TR$FT
    else SC=test_dose_cond.py; CK=$ROOT/dc_n${N}_s$S/exp_dosecond_run/$TR/traj/experiment_dosecond_$TR$DT; fi
    LBL=ep${EP}_${arch}_n${N}_s${S}_on_$reg
    [ -f "$CK" ] || { echo "missing $CK"; continue; }
    [ -f "$OUT/$LBL.json" ] && continue
    while [ "$(jobs -rp | wc -l)" -ge "$MAXPAR" ]; do sleep 5; done
    ( OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 $PY $SC --experiment $EV --scale-from $TR \
        --ckpt "$CK" --label "$LBL" --out-json "$OUT/$LBL.json" > "logs/eval_$LBL.log" 2>&1 \
        && echo "ok $LBL" || echo "FAIL $LBL" ) &
  done
done
wait; echo "NSWEEP_EVAL_DONE N=$N seed=$S ep=$EP"
