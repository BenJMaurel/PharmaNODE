#!/bin/bash
# One run of the NMI 3-scenario study (paper pipeline, HEAD 2eaadf6 worktree) with the MAP-BE fixes.
#   run_one.sh <worktree> <scenario> <seed>      (NITERS env overrides the 6000 epochs, for smoke tests only)
# Steps: seeded data (200 patients -> 160/40) -> Monolix SAEM from published values + MAP-BE with SIGMA as variances
# (sigvar, the fix) -> MAP-BE again from the SAME fit with SIGMA as SDs (sigsd, as published) -> plain latent ODE,
# paper flags -> per-patient AUC of the final (epoch 6000) and _best (paper's test-MSE selection) checkpoints.
# Outputs collected in results/scen_rerun/runs/s<sc>_seed<NNN>/ ; DONE or FAILED_<step> marker there.
set -u
W=$1; SC=$2; SEED=$3
ROOT=/Users/benjaminmaurel/Documents/PharmaNODE
ID=$((SC * 10000 + SEED)); NIT=${NITERS:-6000}
OUT=$ROOT/results/scen_rerun/runs/s${SC}_seed$(printf %03d $SEED); [ "$NIT" != 6000 ] && OUT=${OUT}_smoke
PY=/opt/miniconda3/bin/python3.12; MLX=/Applications/monolixSuite2024R1.app/Contents/Resources/monolixSuite
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MLX_THREADS=1
say () { echo "[$(date '+%F %T')] s$SC seed $SEED id $ID ($(basename $W)): $*"; }
fail () { say "FAILED at $1"; touch $OUT/FAILED_$1; exit 1; }
[ -f $OUT/DONE ] && { say "already done"; exit 0; }
mkdir -p $OUT; rm -f $OUT/FAILED_*
cd $W || exit 1; mkdir -p results logs; rm -rf results/$ID
say "data"
nice -n 5 $PY $ROOT/scripts/scen_rerun/gen_seeded.py $((1000 * SC + SEED)) --exp $ID --num_patients 200 --first_at 1 \
  --scenario $SC > $OUT/gen.log 2>&1 || fail gen
cp results/$ID/virtual_cohort_test.csv $OUT/
say "Monolix + MAP-BE (SIGMA variances)"
REUSE_FIT=0 SIGMA_MODE=var OUT_TAG=sigvar nice -n 5 Rscript $ROOT/scripts/scen_rerun/all_run_tacro_rerun.r \
  --virtual_cohort virtual_cohort_test.csv --output_dir results/$ID --experiment $ID --cores 1 --monolix_path "$MLX" \
  > $OUT/r_sigvar.log 2>&1 || fail mapbe_var
say "MAP-BE (SIGMA as SDs, as published), same fit"
REUSE_FIT=1 SIGMA_MODE=sd OUT_TAG=sigsd nice -n 5 Rscript $ROOT/scripts/scen_rerun/all_run_tacro_rerun.r \
  --virtual_cohort virtual_cohort_test.csv --output_dir results/$ID --experiment $ID --cores 1 --monolix_path "$MLX" \
  > $OUT/r_sigsd.log 2>&1 || fail mapbe_sd
cp results/$ID/tacro_mapbayest_auc_sig*.csv results/$ID/2_test_tacro/populationParameters.txt $OUT/ || fail copy_mapbe
say "latent ODE ($NIT epochs)"
nice -n 5 $PY run_models.py --niters $NIT -n 200 -s 40 -l 10 --dataset PK_Tacro --latent-ode --noise-weight 0.01 \
  --max-t 5. -b 512 --seed $SEED --experiment $ID > $OUT/train.log 2>&1 || fail train
for c in final best; do
  f=results/$ID/experiment_$ID.ckpt; [ $c = best ] && f=results/$ID/experiment_${ID}_best.ckpt
  nice -n 5 $PY -W ignore $ROOT/scripts/repro/eval_lode.py $f $OUT/lode_$c.json > $OUT/eval_$c.log 2>&1 || fail eval_$c
done
touch $OUT/DONE; say "DONE"
