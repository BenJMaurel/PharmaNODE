#!/bin/bash
# In-support 2x2 (train {vc00,vc09} x test {vc00,vc09}) for scenario-4 checkpoints.
#   eval_s4_2x2.sh <root> <arch film|dc> "<epochs6>" "<seeds>" [maxpar] [threads]
# Same harness call and file naming as the 12k queue (other session's queue_12k.sh):
# --experiment <test cohort> --scale-from <TRAINING cohort> (landmine 1.2),
# out <root>/eval/ep<EP>_<arch>_s<N>_<tr>_on_<te>.json.  Existing JSONs are skipped.
set -u
cd "$(dirname "$0")/../.."
ROOT=${1:?root}; ARCH=${2:?arch}; EPS=${3:?epochs}; SEEDS=${4:?seeds}
MAXPAR=${5:-3}; TH=${6:-1}
OUT=$ROOT/eval; mkdir -p "$OUT"
PY="/opt/miniconda3/bin/python3.12 -u"
for EP in $EPS; do for s in $SEEDS; do for tr in vc00 vc09; do for te in vc00 vc09; do
  if [ "$ARCH" = film ]; then SC=test_film_matched.py
    CK=$ROOT/film_s$s/exp_film_run/confound_${tr}_s4/traj/experiment_film_confound_${tr}_s4__noz0_sig0.05_sc141.27_dech50_sel-mse_v2_ep${EP}.ckpt
  else SC=test_dose_cond.py
    CK=$ROOT/dc_s$s/exp_dosecond_run/confound_${tr}_s4/traj/experiment_dosecond_confound_${tr}_s4__sig-0.05_ep${EP}.ckpt
  fi
  LBL=ep${EP}_${ARCH}_s${s}_${tr}_on_${te}
  [ -f "$CK" ] || { echo "missing $CK"; continue; }
  [ -f "$OUT/$LBL.json" ] && continue
  while [ "$(jobs -rp | wc -l)" -ge "$MAXPAR" ]; do sleep 5; done
  ( OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH nice -n 12 $PY $SC --experiment confound_${te}_s4 \
      --scale-from confound_${tr}_s4 --ckpt "$CK" --label "$LBL" --out-json "$OUT/$LBL.json" \
      > "logs/eval_n12k_$LBL.log" 2>&1 && echo "ok $LBL" || echo "FAIL $LBL" ) &
done; done; done; done
wait; echo "EVAL2X2_DONE $ROOT $ARCH $EPS"
