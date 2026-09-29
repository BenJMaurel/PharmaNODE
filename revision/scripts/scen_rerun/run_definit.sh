#!/bin/bash
# Redo of the MAP-BE arm of one 3-scenario run with Monolix DEFAULT initial values (all = 1), exactly as the paper's
# all_run_tacro.r did (Benjamin, 2026-09-26). Reuses the run's data (in the slot worktree that generated it; the test
# file is checked byte-identical to the one the base run scored) and its latent ODE results, so the comparison stays
# paired. From ONE fit: MAP-BE with SIGMA as variances (definit_sigvar) and as SDs (definit_sigsd = the paper's MAP-BE).
#   run_definit.sh <worktree used as cwd, for test_model.txt> <scenario> <seed>
set -u
SC=$2; SEED=$3; ROOT=/Users/benjaminmaurel/Documents/PharmaNODE
W=$(cd "$ROOT" && cd "$1" && pwd) || exit 1   # absolute: the script cd-s into it later (a relative W broke every path, 27/09)
ID=$((SC * 10000 + SEED)); OUT=$ROOT/results/scen_rerun/runs/s${SC}_seed$(printf %03d $SEED)
MLX=/Applications/monolixSuite2024R1.app/Contents/Resources/monolixSuite; R=$ROOT/scripts/scen_rerun/all_run_tacro_rerun_init.r
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MLX_THREADS=1
say () { echo "[$(date '+%F %T')] definit s$SC seed $SEED id $ID ($(basename $W)): $*"; }
fail () { say "FAILED at $1"; touch $OUT/FAILED_definit_$1; exit 1; }
[ -f $OUT/DONE ] || { say "base run not DONE, skipped"; exit 1; }
[ -f $OUT/DONE_definit ] && { say "already done"; exit 0; }
rm -f $OUT/FAILED_definit_*
SRC=$(ls -d $ROOT/results/paper_repro/w*/results/$ID 2>/dev/null); [ "$(echo $SRC | wc -w | tr -d ' ')" = 1 ] || fail locate
D=$W/results/${ID}_definit; rm -rf $D; mkdir -p $D
cp $SRC/virtual_cohort_train.csv $SRC/virtual_cohort_test.csv $D/ || fail copy_data
cmp -s $D/virtual_cohort_test.csv $OUT/virtual_cohort_test.csv || fail datacheck
cd $W || exit 1
say "Monolix (default init) + MAP-BE (SIGMA variances)"
INIT_MODE=default REUSE_FIT=0 SIGMA_MODE=var OUT_TAG=definit_sigvar nice -n 5 Rscript $R --virtual_cohort $D/virtual_cohort_test.csv \
  --output_dir $D --experiment $ID --cores 1 --monolix_path "$MLX" > $OUT/r_definit_sigvar.log 2>&1 || fail mapbe_var
say "MAP-BE (SIGMA as SDs = paper pipeline), same fit"
INIT_MODE=default REUSE_FIT=1 SIGMA_MODE=sd OUT_TAG=definit_sigsd nice -n 5 Rscript $R --virtual_cohort $D/virtual_cohort_test.csv \
  --output_dir $D --experiment $ID --cores 1 --monolix_path "$MLX" > $OUT/r_definit_sigsd.log 2>&1 || fail mapbe_sd
cp $D/tacro_mapbayest_auc_definit_sigvar.csv $D/tacro_mapbayest_auc_definit_sigsd.csv $OUT/ || fail copy_out
cp $D/2_test_tacro/populationParameters.txt $OUT/populationParameters_definit.txt || fail copy_out
touch $OUT/DONE_definit; say "DONE"
