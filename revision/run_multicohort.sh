#!/bin/bash
# ---------------------------------------------------------------------------
# Many-cohort sweep for the paper revision.
#
# Runs the whole comparison over N independently generated cohorts, so the
# unit of analysis becomes the cohort rather than the 40 held-out patients of
# a single draw. With ~50 cohorts a paired across-cohort test is far more
# informative than a within-cohort bootstrap.
#
# Throughput note: these models are small, and torch's own threading actively
# hurts them (0.75 s/epoch with default threads vs 0.45 s/epoch pinned to one).
# Every job is therefore pinned to a single thread and many jobs run at once,
# which measured ~7x the throughput of running them one at a time.
#
#   nohup bash run_multicohort.sh --n-cohorts 50 --niters 2000 --jobs 8 \
#        > multicohort.out 2>&1 &
# ---------------------------------------------------------------------------
set -uo pipefail

N_COHORTS=50
FIRST_ID=20000
COHORTS=""
NUM_PATIENTS=200
SCENARIO=2
GENERATOR="gen_tacro_bimodal.py"
GEN_EXTRA=""
NITERS=2000
SEED=101
PATIENCE=0
LR_PER_EPOCH=0
LATENTS=10
BATCH=512
JOBS=8
DATA_DIR="./results/exp_film_run"
OUT_ROOT="results/revision"
LOG_DIR="logs/multicohort"
ARMS="film dosecond lupk_fair"
DRY_RUN=0
FORCE=0

# measured single-thread seconds per epoch on this machine, for the ETA
declare -A COST=( [film]=0.51 [dosecond]=0.85 [dosecond_zero]=0.85 [lupk]=1.14 [lupk_fair]=0.45 )

usage() {
  cat <<EOF
Usage: bash run_multicohort.sh [options]

  --n-cohorts N       Number of cohorts to run                 (default: $N_COHORTS)
  --first-id N        First cohort ID; IDs are consecutive     (default: $FIRST_ID)
  --cohorts "A B C"   Explicit IDs, overrides --n-cohorts/--first-id
  --num-patients N    Patients per cohort                      (default: $NUM_PATIENTS)
  --scenario N        Generator scenario                       (default: $SCENARIO)
  --generator FILE    Data generator script. Use gen_tacro_bimodal.py for the
                      two-widely-separated-doses design, which also writes
                      dose_sweep_truth.csv for the held-out patients.
                                                               (default: $GENERATOR)
  --gen-extra "..."   Extra args passed verbatim to the generator.
  --niters N          Training epochs per model                (default: $NITERS)
  --jobs N            Concurrent single-threaded jobs          (default: $JOBS)
  --seed N            Training seed                            (default: $SEED)
  --batch-size N      Minibatch size for every arm. The cohort has ~160 training
                      patients, so -b 512 means ONE gradient step per epoch --
                      full-batch descent, which is what makes these runs need so
                      many epochs. -b 32 gives 5 steps per pass at nearly the same
                      cost per pass and measured ~36% (dose-cond) / ~16% (FiLM)
                      lower held-out MSE at equal wall-clock.  (default: $BATCH)
  --lr-per-epoch      Decay the LR once per epoch rather than once per gradient
                      step, so the schedule does not depend on --batch-size.
                      Strongly recommended with a small batch: otherwise the LR
                      hits its floor ~5x sooner in epoch terms and early stopping
                      fires on a dead schedule rather than on convergence.
  --patience N        Stop after N consecutive evaluations (every 10 epochs)
                      without improvement. 0 = never stop early, i.e. train for
                      exactly --niters. Use a real value (e.g. 50) with a large
                      --niters to compare models AT CONVERGENCE rather than at a
                      fixed budget -- a fixed budget favours whichever model
                      converges fastest.                       (default: $PATIENCE)
  --arms "a b"        Subset of: film film_notransport dosecond dosecond_zero
                      lupk lupk_fair                           (default: $ARMS)
                                                               (default: $ARMS)
  --force             Retrain even where a checkpoint exists
  --dry-run           Print the plan and the ETA, run nothing
  -h, --help          This message

Arms:
  film           the paper's OT-FiLM model
  film_notransport  the identity-transport ablation (only worth running once;
                 it loses every cohort by ~30 points, so omit it from repeats)
  dosecond       dose-conditioned baseline, encoder unchanged
  dosecond_zero  dose-conditioned baseline, z0 dose-free by construction
  lupk           Lu et al., faithful (--init history)
  lupk_fair      Lu et al., fair (--init encoder --use-static --train-mode counterfactual)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --n-cohorts) N_COHORTS="$2"; shift 2;;
    --first-id) FIRST_ID="$2"; shift 2;;
    --cohorts) COHORTS="$2"; shift 2;;
    --num-patients) NUM_PATIENTS="$2"; shift 2;;
    --scenario) SCENARIO="$2"; shift 2;;
    --generator) GENERATOR="$2"; shift 2;;
    --gen-extra) GEN_EXTRA="$2"; shift 2;;
    --niters) NITERS="$2"; shift 2;;
    --jobs) JOBS="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --patience) PATIENCE="$2"; shift 2;;
    --lr-per-epoch) LR_PER_EPOCH=1; shift;;
    --batch-size|-b) BATCH="$2"; shift 2;;
    --arms) ARMS="$2"; shift 2;;
    --force) FORCE=1; shift;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1"; usage; exit 1;;
  esac
done

if [[ -z "$COHORTS" ]]; then
  COHORTS=$(seq "$FIRST_ID" $(( FIRST_ID + N_COHORTS - 1 )))
fi
N_ACTUAL=$(echo $COHORTS | wc -w | tr -d ' ')

mkdir -p "$LOG_DIR" "$OUT_ROOT"
STAMP=$(date +%Y%m%d_%H%M%S)
JOBDIR="$LOG_DIR/jobs_${STAMP}"
SUMMARY="$LOG_DIR/summary_${STAMP}.txt"

has_arm() { [[ " $ARMS " == *" $1 "* ]]; }

# 0 means "no early stopping"; all three trainers take --patience in evaluations
PAT_ARG=""
if [[ "$PATIENCE" != "0" ]]; then PAT_ARG=" --patience $PATIENCE"; fi
LRE_ARG=""
if [[ "$LR_PER_EPOCH" == "1" ]]; then LRE_ARG=" --lr-per-epoch"; fi

# ---------------- ETA ----------------
GEN_SEC=$(python3 -c "print(1.5 * $NUM_PATIENTS)")
TRAIN_SEC=0
for a in $ARMS; do
  c=${COST[$a]:-1.0}
  TRAIN_SEC=$(python3 -c "print($TRAIN_SEC + $c * $NITERS)")
done
TOTAL=$(python3 -c "print(($GEN_SEC + $TRAIN_SEC) * $N_ACTUAL / $JOBS / 3600)")
{
  echo "multi-cohort sweep"
  echo "  cohorts      : $N_ACTUAL  ($(echo $COHORTS | cut -d' ' -f1) ... $(echo $COHORTS | rev | cut -d' ' -f1 | rev))"
  echo "  arms         : $ARMS"
  echo "  niters       : $NITERS  (early stopping patience: $PATIENCE)"
  echo "  batch size   : $BATCH   (lr-per-epoch: $LR_PER_EPOCH)"
  echo "  generator    : $GENERATOR  scenario $SCENARIO $GEN_EXTRA"
  echo "  jobs         : $JOBS concurrent, 1 thread each"
  echo "  ETA          : ~$(LC_NUMERIC=C printf '%.1f' $TOTAL) h  (generation + training; evaluation is minor)"
  echo ""
} | tee "$SUMMARY"

# ---------------- build the job scripts ----------------
# One self-contained script per cohort: xargs -I caps constructed command lines,
# and a per-cohort script is also re-runnable by hand for debugging.
mkdir -p "$JOBDIR"

need() {  # need <checkpoint>  -> 0 if the job should run
  [[ $FORCE -eq 1 ]] && return 0
  [[ -f "$1" ]] && return 1
  return 0
}

for EXP in $COHORTS; do
  mkdir -p "$OUT_ROOT/$EXP"
  TRAIN_CSV="$DATA_DIR/$EXP/virtual_cohort_film_train.csv"
  PRE=""
  if [[ ! -f "$TRAIN_CSV" ]]; then
    PRE="python3 $GENERATOR --exp $EXP --num_patients $NUM_PATIENTS --scenario $SCENARIO $GEN_EXTRA > $LOG_DIR/gen_${EXP}.log 2>&1 && "
  fi

  # Each line is one cohort's full pipeline, so generation always precedes the
  # models that read it and the cohorts still spread across the job pool.
  JOB="$JOBDIR/cohort_${EXP}.sh"
  {
    echo "#!/bin/bash"
    echo "set -uo pipefail"
    echo "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1"
    echo "cd \"$(pwd)\""
  } > "$JOB"
  LINE="${PRE}true"

  if has_arm film; then
    CK="results/exp_film_run/$EXP/experiment_film_${EXP}_best.ckpt"
    need "$CK" && LINE="$LINE && python3 run_models.py --niters $NITERS -n 200 -s 40 -l $LATENTS --dataset PK_Tacro --latent-ode --use_film --noise-weight 0.01 --max-t 5. -b $BATCH --seed $SEED --experiment $EXP$PAT_ARG$LRE_ARG > $LOG_DIR/film_${EXP}.log 2>&1"
    LINE="$LINE && python3 test_film_matched.py --experiment $EXP --data-dir $DATA_DIR --eval-split test > $LOG_DIR/film_test_${EXP}.log 2>&1"
  fi
  if has_arm film_notransport; then
    LINE="$LINE ; python3 test_film_matched.py --experiment $EXP --data-dir $DATA_DIR --eval-split test --transport none --label 'OT-FiLM (no transport)' > $LOG_DIR/film_none_${EXP}.log 2>&1"
  fi
  if has_arm dosecond; then
    CK="results/exp_dosecond_run/$EXP/experiment_dosecond_${EXP}_best.ckpt"
    LINE="$LINE ; true"
    need "$CK" && LINE="$LINE && python3 train_dose_cond.py --experiment $EXP --data-dir $DATA_DIR --niters $NITERS -b $BATCH -l $LATENTS --lr 1e-2 --seed $SEED$PAT_ARG$LRE_ARG > $LOG_DIR/dosecond_${EXP}.log 2>&1"
    LINE="$LINE && python3 test_dose_cond.py --experiment $EXP --data-dir $DATA_DIR --eval-split test > $LOG_DIR/dosecond_test_${EXP}.log 2>&1"
  fi
  if has_arm dosecond_zero; then
    CK="results/exp_dosecond_run/$EXP/experiment_dosecond_${EXP}__encdose-zero_best.ckpt"
    LINE="$LINE ; true"
    LINE="$LINE ; true"
    need "$CK" && LINE="$LINE && python3 train_dose_cond.py --experiment $EXP --data-dir $DATA_DIR --niters $NITERS -b $BATCH -l $LATENTS --lr 1e-2 --seed $SEED --encoder-dose zero$PAT_ARG$LRE_ARG > $LOG_DIR/dosecond_zero_${EXP}.log 2>&1"
    LINE="$LINE && python3 test_dose_cond.py --experiment $EXP --data-dir $DATA_DIR --eval-split test --tag '__encdose-zero' --label 'dose-conditioned (z0 dose-free)' > $LOG_DIR/dosecond_zero_test_${EXP}.log 2>&1"
  fi
  if has_arm lupk; then
    CK="results/exp_lupk_run/$EXP/experiment_lupk_${EXP}_best.ckpt"
    LINE="$LINE ; true"
    need "$CK" && LINE="$LINE && python3 train_lu_pk.py --experiment $EXP --data-dir $DATA_DIR --niters $NITERS -b $BATCH --lr 1e-3 --seed $SEED$PAT_ARG > $LOG_DIR/lupk_${EXP}.log 2>&1"
    LINE="$LINE && python3 test_lu_pk.py --experiment $EXP --data-dir $DATA_DIR --eval-split test --label 'Lu et al. neural-PK (faithful)' > $LOG_DIR/lupk_test_${EXP}.log 2>&1"
  fi
  if has_arm lupk_fair; then
    TAG="__mode-counterfactual_static_init-encoder"
    CK="results/exp_lupk_run/$EXP/experiment_lupk_${EXP}${TAG}_best.ckpt"
    LINE="$LINE ; true"
    need "$CK" && LINE="$LINE && python3 train_lu_pk.py --experiment $EXP --data-dir $DATA_DIR --niters $NITERS -b $BATCH --lr 1e-3 --seed $SEED --init encoder --use-static --train-mode counterfactual$PAT_ARG > $LOG_DIR/lupk_fair_${EXP}.log 2>&1"
    LINE="$LINE && python3 test_lu_pk.py --experiment $EXP --data-dir $DATA_DIR --eval-split test --tag '$TAG' --label 'Lu et al. neural-PK (fair)' > $LOG_DIR/lupk_fair_test_${EXP}.log 2>&1"
  fi

  LINE="$LINE || echo \"COHORT $EXP FAILED\" >> $LOG_DIR/failures_${STAMP}.txt"
  echo "$LINE" >> "$JOB"
  chmod +x "$JOB"
done

N_JOBS=$(ls "$JOBDIR"/cohort_*.sh 2>/dev/null | wc -l | tr -d ' ')
echo "$N_JOBS cohort pipelines queued -> $JOBDIR" | tee -a "$SUMMARY"

if [[ $DRY_RUN -eq 1 ]]; then
  echo ""
  echo "--- first queued pipeline ---"
  sed 's/ && /\n  /g' "$(ls "$JOBDIR"/cohort_*.sh | head -1)" | head -24
  exit 0
fi

# ---------------- run ----------------
START=$SECONDS
ls "$JOBDIR"/cohort_*.sh | xargs -P "$JOBS" -I JOBSCRIPT bash JOBSCRIPT
ELAPSED=$(( SECONDS - START ))

{
  echo ""
  echo "sweep finished in $(( ELAPSED / 3600 ))h $(( (ELAPSED % 3600) / 60 ))m"
  if [[ -f "$LOG_DIR/failures_${STAMP}.txt" ]]; then
    echo "FAILURES:"; cat "$LOG_DIR/failures_${STAMP}.txt"
  else
    echo "no cohort pipeline reported a failure"
  fi
} | tee -a "$SUMMARY"

# ---------------- aggregate ----------------
echo "" | tee -a "$SUMMARY"
python3 compare_revision_results.py --across-cohorts "$OUT_ROOT" \
  --cohorts "$(echo $COHORTS | tr '\n' ' ')" 2>&1 | tee -a "$SUMMARY"

echo ""
echo "Summary: $SUMMARY"
