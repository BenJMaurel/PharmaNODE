#!/bin/bash
# Handoff 12.3 (launched 28/09 night): (A) clearance change in the steady-state history, paper scenario 1, one visit:
# s in {0.3, 0.5} with the change at the observed dose (DRIFT_SWITCH=6) + s = 0.5 one interval earlier (DRIFT_SWITCH=5),
# data seeds 1-3 (paired with the paper rerun's kappa=0 fits). (B) sigma sweep: MAP-BE re-run from EXISTING fits with the
# residual SD scaled by lambda in {0.5, 0.7, 1.4, 2.0} (lambda = 1 = the existing c1 CSVs): paper scenarios 1-3 seeds 1-10,
# the dose-to-dose fits (12.2) and the clearance-change fits. 4 jobs at a time x 1 thread. Restart-safe (skips outputs).
cd /Users/benjaminmaurel/Documents/PharmaNODE
ROOT=$PWD; W=$ROOT/results/paper_repro/idv; OUT=$ROOT/results/idv; PY=/opt/miniconda3/bin/python3.12
MLX=/Applications/monolixSuite2024R1.app/Contents/Resources/monolixSuite
RC1=$ROOT/scripts/scen_rerun/all_run_tacro_rerun_c1.r; RS=$ROOT/scripts/idv/all_run_tacro_sigscale.r
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MLX_THREADS=1
say() { echo "[$(date '+%F %T')] $*"; }
mkdir -p $OUT/drift $OUT/sigsweep
# ---- (A) data, sequential (the paper generator also writes into the worktree root) ----
DRIFT=("30 6" "50 6" "50 5")
for d in "${DRIFT[@]}"; do set -- $d; for SEED in 1 2 3; do
  ID=drift$1_sw$2_$((10000 + SEED))
  [ -f $W/results/$ID/drift_truth.csv ] && continue
  ( cd $W && rm -rf results/$ID && DRIFT_SWITCH=$2 nice -n 5 $PY $ROOT/scripts/idv/gen_seeded_drift.py $((1000 + SEED)) 0.$1 \
      --exp $ID --num_patients 200 --first_at 1 --scenario 1 > $OUT/drift/gen_$ID.log 2>&1 ) || { say "GEN FAILED $ID"; exit 1; }
  say "data $ID"
done; done
# ---- (B) regression gate: lambda = 1 through the sweep copy must reproduce the c1 CSV of s1 seed 1 ----
sweep_job() {   # <tag> <popparams> <test csv> <lambda>
  local D=$OUT/sigsweep/$1/lam$4
  [ -f $D/tacro_mapbayest_auc_lam$4.csv ] && return 0
  rm -rf $D; mkdir -p $D/2_test_tacro && cp $2 $D/2_test_tacro/populationParameters.txt && cp $3 $D/virtual_cohort_test.csv || return 1
  ( cd $W && REUSE_FIT=1 SIGMA_MODE=c1 SIGMA_SCALE=$4 OUT_TAG=lam$4 nice -n 5 Rscript $RS --virtual_cohort $D/virtual_cohort_test.csv \
      --output_dir $D --experiment $1 --cores 1 --monolix_path "$MLX" > $D/r.log 2>&1 )
}
export -f sweep_job; export OUT W RS MLX
S1=$ROOT/results/scen_rerun/runs/s1_seed001
if [ ! -f $OUT/sigsweep/REGRESSION_OK ]; then
  sweep_job regress_s1_seed001 $S1/populationParameters.txt $S1/virtual_cohort_test.csv 1 || { say "REGRESSION RUN FAILED"; exit 1; }
  $PY -c "
import pandas as pd; a=pd.read_csv('$OUT/sigsweep/regress_s1_seed001/lam1/tacro_mapbayest_auc_lam1.csv'); b=pd.read_csv('$S1/tacro_mapbayest_auc_sigc1.csv'); m=a.merge(b,on='ID')
d=float((m.auc_ipred_x-m.auc_ipred_y).abs().max()); print(f'REGRESSION lambda=1 vs sigc1: n={len(m)}/{len(b)}, max |dAUC| = {d:.3g}'); raise SystemExit(0 if len(m)==len(b) and d<1e-3 else 1)" \
    && touch $OUT/sigsweep/REGRESSION_OK || { say "REGRESSION MISMATCH -- sweep not run"; exit 1; }
fi
say "regression gate passed"
# ---- job list 1: clearance-change fits + sweeps of the existing fits ----
J1=$OUT/jobs_night1.txt; : > $J1
for d in "${DRIFT[@]}"; do set -- $d; for SEED in 1 2 3; do
  ID=drift$1_sw$2_$((10000 + SEED)); D=$W/results/$ID
  [ -f $OUT/drift/DONE_$ID ] || echo "cd $W && SIGMA_MODE=c1 OUT_TAG=sigc1 nice -n 5 Rscript $RC1 --virtual_cohort $D/virtual_cohort_test.csv --output_dir results/$ID --experiment $ID --cores 1 --monolix_path '$MLX' > $OUT/drift/r_$ID.log 2>&1 && cp $D/2_test_tacro/populationParameters.txt $OUT/drift/popparams_$ID.txt && cp $D/tacro_mapbayest_auc_sigc1.csv $OUT/drift/mapbe_sigc1_$ID.csv && cp $D/drift_truth.csv $OUT/drift/drift_truth_$ID.csv && touch $OUT/drift/DONE_$ID" >> $J1
done; done
LAMS="0.5 0.7 1.4 2.0"
for sc in 1 2 3; do for SEED in 1 2 3 4 5 6 7 8 9 10; do
  R=$ROOT/results/scen_rerun/runs/s${sc}_seed$(printf %03d $SEED)
  for l in $LAMS; do echo "sweep_job s${sc}_seed$(printf %03d $SEED) $R/populationParameters.txt $R/virtual_cohort_test.csv $l" >> $J1; done
done; done
for K in 25 50; do for SEED in 1 2 3; do ID=idv${K}_$((10000 + SEED))
  for l in $LAMS; do echo "sweep_job $ID $OUT/check/popparams_$ID.txt $W/results/$ID/virtual_cohort_test.csv $l" >> $J1; done
done; done
say "job list 1: $(wc -l < $J1) jobs"; bash scripts/s4/run_jobs.sh $J1 4
# ---- job list 2: sweeps of the clearance-change fits ----
J2=$OUT/jobs_night2.txt; : > $J2
for d in "${DRIFT[@]}"; do set -- $d; for SEED in 1 2 3; do ID=drift$1_sw$2_$((10000 + SEED))
  [ -f $OUT/drift/DONE_$ID ] || { say "fit $ID missing, no sweep"; continue; }
  for l in $LAMS; do echo "sweep_job $ID $OUT/drift/popparams_$ID.txt $W/results/$ID/virtual_cohort_test.csv $l" >> $J2; done
done; done
say "job list 2: $(wc -l < $J2) jobs"; bash scripts/s4/run_jobs.sh $J2 4
$PY scripts/idv/drift_table.py | tee $OUT/drift/drift_table.txt
$PY scripts/idv/sigsweep_table.py | tee $OUT/sigsweep/sigsweep_table.txt
say "NIGHT_IDV_DONE"
