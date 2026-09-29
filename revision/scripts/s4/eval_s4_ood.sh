#!/bin/bash
# Out-of-support dose extrapolation for scenario 4, CONTROL-trained models.
#   eval_s4_ood.sh <root> <epoch6> [seeds] [maxpar]
#     root   results/s4 or results/s4_n12000
#     epoch6 zero-padded epoch, e.g. 006000
# Same harness call and file naming as the in-support scenario-4 evals, so the
# JSONs sit next to results/<root>/eval/ep<EP>_<arch>_s<N>_vc00_on_vc00.json.
# Scaler pinned to the TRAINING cohort (landmine 1.2) -- which also pins the dose
# normalisation to 8 mg, so 10/12 mg are genuine extrapolation in dose input.
set -u
cd "$(dirname "$0")/../.."
ROOT=${1:?root}; EP=${2:?epoch}; SEEDS=${3:-1 2 3 4}; MAXPAR=${4:-2}
OUT=$ROOT/eval; mkdir -p "$OUT"
PY="/opt/miniconda3/bin/python3.12 -u"
FT=__noz0_sig0.05_sc141.27_dech50_sel-mse_v2_ep${EP}.ckpt
DT=__sig-0.05_ep${EP}.ckpt
for s in $SEEDS; do
  for reg in lo hi; do
    for arch in film dc; do
      if [ $arch = film ]; then SC=test_film_matched.py
        CK=$ROOT/film_s$s/exp_film_run/confound_vc00_s4/traj/experiment_film_confound_vc00_s4$FT
      else SC=test_dose_cond.py
        CK=$ROOT/dc_s$s/exp_dosecond_run/confound_vc00_s4/traj/experiment_dosecond_confound_vc00_s4$DT
      fi
      LBL=ep${EP}_${arch}_s${s}_vc00_on_vc00${reg}
      [ -f "$CK" ] || { echo "missing $CK"; continue; }
      [ -f "$OUT/$LBL.json" ] && continue
      while [ "$(jobs -rp | wc -l)" -ge "$MAXPAR" ]; do sleep 5; done
      ( OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 12 $PY $SC --experiment confound_vc00_s4_$reg \
          --scale-from confound_vc00_s4 --ckpt "$CK" --label "$LBL" --out-json "$OUT/$LBL.json" \
          > "logs/eval_$LBL.log" 2>&1 && echo "ok $LBL" || echo "FAIL $LBL" ) &
    done
  done
done
wait; echo "OOD_EVAL_DONE $ROOT $EP"
