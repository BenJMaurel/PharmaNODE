#!/bin/bash
# Handoff 12, first checkpoint: does dose-to-dose bioavailability variability inflate the residual error NLME
# estimates, in the paper's one-visit scenario-1 setting?  kappa in {0.25, 0.5} x data seeds 1-3 (the paper rerun's
# seeds, so kappa = 0 is the existing run: same patients, etas and noise). Monolix from the published values +
# MAP-BE with the combined1 error form (all_run_tacro_rerun_c1.r, used read-only). Worktree results/paper_repro/idv.
cd /Users/benjaminmaurel/Documents/PharmaNODE
ROOT=$PWD; W=$ROOT/results/paper_repro/idv; OUT=$ROOT/results/idv/check; PY=/opt/miniconda3/bin/python3.12
MLX=/Applications/monolixSuite2024R1.app/Contents/Resources/monolixSuite; R=$ROOT/scripts/scen_rerun/all_run_tacro_rerun_c1.r
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MLX_THREADS=1
say() { echo "[$(date +%H:%M:%S)] $*"; }
mkdir -p $OUT
JOBS=()
for K in 25 50; do for SEED in 1 2 3; do
  ID=idv${K}_$((10000 + SEED)); JOBS+=("$K $SEED $ID")
  [ -f $W/results/$ID/virtual_cohort_train.csv ] && continue
  ( cd $W && rm -rf results/$ID && nice -n 5 $PY $ROOT/scripts/idv/gen_seeded_idv.py $((1000 + SEED)) 0.$K \
      --exp $ID --num_patients 200 --first_at 1 --scenario 1 > $OUT/gen_$ID.log 2>&1 ) || { say "GEN FAILED $ID"; exit 1; }
  say "data $ID"
done; done
fit() {
  set -- $1; K=$1; SEED=$2; ID=$3; D=$W/results/$ID
  [ -f $OUT/DONE_$ID ] && return
  ( cd $W && SIGMA_MODE=c1 OUT_TAG=sigc1 nice -n 5 Rscript $R --virtual_cohort $D/virtual_cohort_test.csv \
      --output_dir results/$ID --experiment $ID --cores 1 --monolix_path "$MLX" > $OUT/r_$ID.log 2>&1 ) \
    && cp $D/2_test_tacro/populationParameters.txt $OUT/popparams_$ID.txt \
    && cp $D/tacro_mapbayest_auc_sigc1.csv $OUT/mapbe_sigc1_$ID.csv && cp $D/idv_truth.csv $OUT/idv_truth_$ID.csv \
    && touch $OUT/DONE_$ID && say "fit $ID done" || say "FIT FAILED $ID"
}
i=0
for j in "${JOBS[@]}"; do fit "$j" & i=$((i+1)); [ $((i % 4)) -eq 0 ] && wait; done
wait
$PY scripts/idv/idv_check_table.py | tee $OUT/idv_check_table.txt
say "IDV_CHECK_DONE"
