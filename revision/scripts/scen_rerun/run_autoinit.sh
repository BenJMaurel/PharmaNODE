#!/bin/bash
# MAP-BE arm of one 3-scenario run with Monolix AUTO-INIT (getFixedEffectsByAutoInit, the GUI's "Auto-init" button)
# and the combined1 (form-fixed) residual error (Benjamin, 2026-09-28): third initialisation next to published init
# (Table 1) and default init (all = 1, the committed paper script). Same data as the base run (test file cmp-checked),
# so every comparison stays paired. Fit + MAP-BE in ONE R call (INIT_MODE=auto SIGMA_MODE=c1).
# Outputs in the run dir: tacro_mapbayest_auc_autoinit_sigc1.csv, populationParameters_autoinit.txt,
# autoinit_values.csv (the auto-init starting values), r_autoinit_sigc1.log, DONE_autoinit (or FAILED_autoinit_<step>).
# AUTOINIT_IDS=n restricts auto-init to the first n individuals (lixoftConnectors ids=; default all).
#   run_autoinit.sh <slot worktree, cwd for test_model.txt> <scenario> <seed>
set -u
SC=$2; SEED=$3; ROOT=/Users/benjaminmaurel/Documents/PharmaNODE
W=$(cd "$ROOT" && cd "$1" && pwd) || exit 1
ID=$((SC * 10000 + SEED)); OUT=$ROOT/results/scen_rerun/runs/s${SC}_seed$(printf %03d $SEED)
MLX=/Applications/monolixSuite2024R1.app/Contents/Resources/monolixSuite; R=$ROOT/scripts/scen_rerun/all_run_tacro_rerun_c1.r
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MLX_THREADS=1
say () { echo "[$(date '+%F %T')] autoinit s$SC seed $SEED id $ID ($(basename $W)): $*"; }
fail () { say "FAILED at $1"; touch $OUT/FAILED_autoinit_$1; exit 1; }
[ -f $OUT/DONE ] || { say "base run not DONE, skipped"; exit 1; }
[ -f $OUT/DONE_autoinit ] && { say "already done"; exit 0; }
rm -f $OUT/FAILED_autoinit_*
SRC=$(ls -d $ROOT/results/paper_repro/w*/results/$ID 2>/dev/null); [ "$(echo $SRC | wc -w | tr -d ' ')" = 1 ] || fail locate
D=$W/results/${ID}_autoinit; rm -rf $D; mkdir -p $D
cp $SRC/virtual_cohort_train.csv $SRC/virtual_cohort_test.csv $D/ || fail copy_data
cmp -s $D/virtual_cohort_test.csv $OUT/virtual_cohort_test.csv || fail datacheck
cd $W || exit 1
say "Monolix (auto-init) + MAP-BE (combined1 error form)"
INIT_MODE=auto REUSE_FIT=0 SIGMA_MODE=c1 OUT_TAG=autoinit_sigc1 nice -n 5 Rscript $R --virtual_cohort $D/virtual_cohort_test.csv \
  --output_dir $D --experiment $ID --cores 1 --monolix_path "$MLX" > $OUT/r_autoinit_sigc1.log 2>&1 || fail mapbe
cp $D/tacro_mapbayest_auc_autoinit_sigc1.csv $D/autoinit_values.csv $OUT/ || fail copy_out
cp $D/2_test_tacro/populationParameters.txt $OUT/populationParameters_autoinit.txt || fail copy_out
touch $OUT/DONE_autoinit; say "DONE"
