#!/bin/bash
# MAP-BE with the combined1 residual-error form (SD = a + b*f, as Monolix estimates it; SIGMA_MODE=c1) for one
# 3-scenario run, from the SAME Monolix fit as Table 1 (published init; found in the slot worktree w*/results/<ID>).
# Output tacro_mapbayest_auc_sigc1.csv + DONE_c1 in the run dir. REGRESS=1 also reruns SIGMA_MODE=var with this
# script and checks it reproduces the run's existing sigvar CSV.   run_c1.sh <scenario> <seed>
# FIT=definit (27/09): use the DEFAULT-INIT fit of the redo instead (w*/results/<ID>_definit) -> tacro_mapbayest_auc_definit_sigc1.csv,
# DONE_definit_c1. Needs that fit to exist: DONE_definit, or a FAILED_definit_mapbe_* run whose Monolix fit completed.
set -u
SC=$1; SEED=$2; ROOT=/Users/benjaminmaurel/Documents/PharmaNODE
ID=$((SC * 10000 + SEED)); OUT=$ROOT/results/scen_rerun/runs/s${SC}_seed$(printf %03d $SEED)
MLX=/Applications/monolixSuite2024R1.app/Contents/Resources/monolixSuite; R=$ROOT/scripts/scen_rerun/all_run_tacro_rerun_c1.r
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MLX_THREADS=1
say () { echo "[$(date '+%F %T')] c1 s$SC seed $SEED id $ID: $*"; }
fail () { say "FAILED at $1"; touch $OUT/FAILED_c1_$1; exit 1; }
[ -f $OUT/DONE ] || { say "base run not DONE, skipped"; exit 1; }
FIT=${FIT:-published}; if [ $FIT = definit ]; then P=definit_; else P=; fi       # P = prefix of markers / output tag
fail () { say "FAILED at $1 (FIT=$FIT)"; touch $OUT/FAILED_${P}c1_$1; exit 1; }
[ -f $OUT/DONE_${P}c1 ] && [ "${REGRESS:-0}" != 1 ] && { say "already done (FIT=$FIT)"; exit 0; }
rm -f $OUT/FAILED_${P}c1_*
SRC=$(ls -d $ROOT/results/paper_repro/w*/results/$ID 2>/dev/null); [ "$(echo $SRC | wc -w | tr -d ' ')" = 1 ] || fail locate
W=$(dirname $(dirname $SRC)); D=$W/results/${ID}_${P}c1; rm -rf $D; mkdir -p $D/2_test_tacro
if [ $FIT = definit ]; then
  FS=$W/results/${ID}_definit/2_test_tacro/populationParameters.txt; [ -f $FS ] || fail no_definit_fit
  [ -f $OUT/DONE_definit ] && { cmp -s $FS $OUT/populationParameters_definit.txt || fail fitcheck; }
else FS=$SRC/2_test_tacro/populationParameters.txt; cmp -s $FS $OUT/populationParameters.txt || fail fitcheck; fi
cp $FS $D/2_test_tacro/ && cp $SRC/virtual_cohort_test.csv $D/ || fail copy_data
cmp -s $D/virtual_cohort_test.csv $OUT/virtual_cohort_test.csv || fail datacheck
cd $W || exit 1
if [ "${REGRESS:-0}" = 1 ]; then
  REUSE_FIT=1 SIGMA_MODE=var OUT_TAG=regress_var nice -n 5 Rscript $R --virtual_cohort $D/virtual_cohort_test.csv --output_dir $D \
    --experiment $ID --cores 1 --monolix_path "$MLX" > $OUT/r_regress_var.log 2>&1 || fail regress
  /opt/miniconda3/bin/python3.12 -c "
import pandas as pd; a=pd.read_csv('$D/tacro_mapbayest_auc_regress_var.csv'); b=pd.read_csv('$OUT/tacro_mapbayest_auc_sigvar.csv'); m=a.merge(b,on='ID')
d=float((m.auc_ipred_x-m.auc_ipred_y).abs().max()); print(f'REGRESSION sigvar via the c1 script: n={len(m)}/{len(b)}, max |dAUC| = {d:.3g}'); raise SystemExit(0 if len(m)==len(b) and d<1e-3 else 1)" || fail regress_mismatch
fi
say "MAP-BE (combined1 error form), FIT=$FIT"
REUSE_FIT=1 SIGMA_MODE=c1 OUT_TAG=${P}sigc1 nice -n 5 Rscript $R --virtual_cohort $D/virtual_cohort_test.csv --output_dir $D \
  --experiment $ID --cores 1 --monolix_path "$MLX" > $OUT/r_${P}sigc1.log 2>&1 || fail mapbe_c1
cp $D/tacro_mapbayest_auc_${P}sigc1.csv $OUT/ || fail copy_out
touch $OUT/DONE_${P}c1; say "DONE (FIT=$FIT)"
