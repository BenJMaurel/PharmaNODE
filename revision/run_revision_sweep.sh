#!/bin/bash
# ---------------------------------------------------------------------------
# Overnight sweep for the paper revision.
#
# Generates one or more cohorts and trains + evaluates every arm of the
# comparison on each of them, writing one JSON per arm per cohort into
# results/revision/<COHORT>/ and a combined markdown table at the end.
#
# Safe to re-run: --skip-existing (on by default) skips any arm whose best
# checkpoint is already on disk, so an interrupted sweep resumes where it
# stopped. Individual failures are logged and do not abort the sweep.
#
#   nohup bash run_revision_sweep.sh --cohorts "12345 12346 12347" > sweep.out 2>&1 &
# ---------------------------------------------------------------------------
set -uo pipefail

COHORTS="12345"
NUM_PATIENTS=200
SCENARIO=2
NITERS=1500
SEED=101
LATENTS=10
BATCH=512
DATA_DIR="./results/exp_film_run"
OUT_ROOT="results/revision"
LOG_DIR="logs/sweep"
ARMS="film dosecond dosecond_zero lupk lupk_fair"
SKIP_EXISTING=1
DRY_RUN=0

usage() {
  cat <<EOF
Usage: bash run_revision_sweep.sh [options]

  --cohorts "A B C"   Cohort / experiment IDs to run          (default: $COHORTS)
  --num-patients N    Patients per generated cohort           (default: $NUM_PATIENTS)
  --scenario N        Generator scenario                      (default: $SCENARIO)
  --niters N          Training epochs per model               (default: $NITERS)
  --seed N            Training seed                           (default: $SEED)
  --arms "a b"        Subset of: film dosecond dosecond_zero lupk lupk_fair
                                                              (default: all)
  --no-skip-existing  Retrain even if a checkpoint exists
  --dry-run           Print the commands without running them
  -h, --help          This message

Rough cost per cohort at --niters 6000 on CPU: film ~3.5 h, dosecond ~3 h,
lupk ~3 h, lupk_fair ~5 h. Budget accordingly, or lower --niters for a
first pass across several cohorts.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cohorts) COHORTS="$2"; shift 2;;
    --num-patients) NUM_PATIENTS="$2"; shift 2;;
    --scenario) SCENARIO="$3"; shift 2;;
    --niters) NITERS="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --arms) ARMS="$2"; shift 2;;
    --no-skip-existing) SKIP_EXISTING=0; shift;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1"; usage; exit 1;;
  esac
done

mkdir -p "$LOG_DIR" "$OUT_ROOT"
STAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY="$LOG_DIR/summary_${STAMP}.txt"
echo "sweep started $(date)" | tee "$SUMMARY"
echo "cohorts=[$COHORTS] arms=[$ARMS] niters=$NITERS seed=$SEED" | tee -a "$SUMMARY"

has_arm() { [[ " $ARMS " == *" $1 "* ]]; }

# run <label> <logfile> <command...>
run() {
  local label="$1"; shift
  local log="$1"; shift
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry-run] $label: $*"
    return 0
  fi
  echo "  -> $label  ($(date +%H:%M:%S))" | tee -a "$SUMMARY"
  local t0=$SECONDS
  if "$@" > "$log" 2>&1; then
    echo "     ok in $(( SECONDS - t0 ))s   log: $log" | tee -a "$SUMMARY"
    return 0
  else
    echo "     FAILED after $(( SECONDS - t0 ))s   see $log" | tee -a "$SUMMARY"
    tail -5 "$log" | sed 's/^/       /' | tee -a "$SUMMARY"
    return 1
  fi
}

# skip if the best checkpoint already exists AND is not stale.
# A Lu checkpoint saved before the dosing-history fix has no `init` in its args;
# those were trained from a zero initial state at t=0, which is mis-specified for
# this steady-state dataset, so they are retrained rather than reused.
have_ckpt() {
  [[ $SKIP_EXISTING -eq 1 && -f "$1" ]] || return 1
  if [[ "$1" == *lupk* ]]; then
    if ! python3 -c "
import torch,sys
c=torch.load('$1',map_location='cpu',weights_only=False)
sys.exit(0 if hasattr(c['args'],'init') else 1)" 2>/dev/null; then
      echo "  -- checkpoint predates the dosing-history fix, retraining: $1" | tee -a "$SUMMARY"
      return 1
    fi
  fi
  echo "  -- skipping, checkpoint exists: $1" | tee -a "$SUMMARY"
  return 0
}

for EXP in $COHORTS; do
  echo "" | tee -a "$SUMMARY"
  echo "=================== cohort $EXP ===================" | tee -a "$SUMMARY"
  OUT_DIR="$OUT_ROOT/$EXP"
  mkdir -p "$OUT_DIR"

  # ---- data ----
  if [[ -f "$DATA_DIR/$EXP/virtual_cohort_film_train.csv" ]]; then
    echo "  -- cohort data already present" | tee -a "$SUMMARY"
  else
    run "generate cohort" "$LOG_DIR/gen_${EXP}.log" \
      python3 gen_tacro_film.py --exp "$EXP" --num_patients "$NUM_PATIENTS" --scenario "$SCENARIO" \
      || { echo "  !! generation failed, skipping cohort $EXP" | tee -a "$SUMMARY"; continue; }
  fi

  # ---- OT-FiLM (the paper's model) ----
  if has_arm film; then
    CK="results/exp_film_run/$EXP/experiment_film_${EXP}_best.ckpt"
    have_ckpt "$CK" || run "train film" "$LOG_DIR/film_${EXP}.log" \
      python3 run_models.py --niters "$NITERS" -n 200 -s 40 -l "$LATENTS" \
        --dataset PK_Tacro --latent-ode --use_film --noise-weight 0.01 --max-t 5. \
        -b "$BATCH" --seed "$SEED" --experiment "$EXP"
    run "test film" "$LOG_DIR/film_test_${EXP}.log" \
      python3 test_film_matched.py --experiment "$EXP" --data-dir "$DATA_DIR" \
        --eval-split test --out-json "$OUT_DIR/film.json"
    # the transport ablation: cheap, and the strongest single figure
    for TR in none oracle; do
      run "test film transport=$TR" "$LOG_DIR/film_test_${TR}_${EXP}.log" \
        python3 test_film_matched.py --experiment "$EXP" --data-dir "$DATA_DIR" \
          --eval-split test --transport "$TR" --label "OT-FiLM (transport=$TR)" \
          --out-json "$OUT_DIR/film_transport_${TR}.json"
    done
  fi

  # ---- dose-conditioned baseline, encoder unchanged ----
  if has_arm dosecond; then
    CK="results/exp_dosecond_run/$EXP/experiment_dosecond_${EXP}_best.ckpt"
    have_ckpt "$CK" || run "train dosecond" "$LOG_DIR/dosecond_${EXP}.log" \
      python3 train_dose_cond.py --experiment "$EXP" --data-dir "$DATA_DIR" \
        --niters "$NITERS" -b "$BATCH" -l "$LATENTS" --lr 1e-2 --seed "$SEED"
    run "test dosecond" "$LOG_DIR/dosecond_test_${EXP}.log" \
      python3 test_dose_cond.py --experiment "$EXP" --data-dir "$DATA_DIR" \
        --eval-split test --out-json "$OUT_DIR/dosecond.json"
  fi

  # ---- dose-conditioned baseline, z0 dose-free by construction ----
  if has_arm dosecond_zero; then
    TAG="__encdose-zero"
    CK="results/exp_dosecond_run/$EXP/experiment_dosecond_${EXP}${TAG}_best.ckpt"
    have_ckpt "$CK" || run "train dosecond encdose=zero" "$LOG_DIR/dosecond_zero_${EXP}.log" \
      python3 train_dose_cond.py --experiment "$EXP" --data-dir "$DATA_DIR" \
        --niters "$NITERS" -b "$BATCH" -l "$LATENTS" --lr 1e-2 --seed "$SEED" \
        --encoder-dose zero
    run "test dosecond encdose=zero" "$LOG_DIR/dosecond_zero_test_${EXP}.log" \
      python3 test_dose_cond.py --experiment "$EXP" --data-dir "$DATA_DIR" \
        --eval-split test --tag "$TAG" --label "dose-conditioned (z0 dose-free)" \
        --out-json "$OUT_DIR/dosecond_zero.json"
  fi

  # ---- Lu et al., faithful ----
  if has_arm lupk; then
    CK="results/exp_lupk_run/$EXP/experiment_lupk_${EXP}_best.ckpt"
    have_ckpt "$CK" || run "train lupk (faithful)" "$LOG_DIR/lupk_${EXP}.log" \
      python3 train_lu_pk.py --experiment "$EXP" --data-dir "$DATA_DIR" \
        --niters "$NITERS" -b "$BATCH" --lr 1e-3 --seed "$SEED"
    run "test lupk (faithful)" "$LOG_DIR/lupk_test_${EXP}.log" \
      python3 test_lu_pk.py --experiment "$EXP" --data-dir "$DATA_DIR" \
        --eval-split test --out-json "$OUT_DIR/lupk.json"
  fi

  # ---- Lu et al., fairness check: same covariates, trained on the target task ----
  if has_arm lupk_fair; then
    TAG="__mode-counterfactual_static"
    CK="results/exp_lupk_run/$EXP/experiment_lupk_${EXP}${TAG}_best.ckpt"
    have_ckpt "$CK" || run "train lupk (fair)" "$LOG_DIR/lupk_fair_${EXP}.log" \
      python3 train_lu_pk.py --experiment "$EXP" --data-dir "$DATA_DIR" \
        --niters "$NITERS" -b "$BATCH" --lr 1e-3 --seed "$SEED" \
        --use-static --train-mode counterfactual
    run "test lupk (fair)" "$LOG_DIR/lupk_fair_test_${EXP}.log" \
      python3 test_lu_pk.py --experiment "$EXP" --data-dir "$DATA_DIR" \
        --eval-split test --tag "$TAG" --label "Lu et al. neural-PK (fair)" \
        --out-json "$OUT_DIR/lupk_fair.json"
  fi

  if [[ $DRY_RUN -eq 0 ]]; then
    echo "  --- table for cohort $EXP ---" | tee -a "$SUMMARY"
    python3 compare_revision_results.py "$OUT_DIR"/*.json \
      --out "$OUT_DIR/table.md" 2>/dev/null | tee -a "$SUMMARY"
  fi
done

echo "" | tee -a "$SUMMARY"
echo "sweep finished $(date)" | tee -a "$SUMMARY"
echo "Per-cohort tables: $OUT_ROOT/<cohort>/table.md" | tee -a "$SUMMARY"
echo "Summary: $SUMMARY"
echo ""
echo "To test whether two arms actually differ, use the paired bootstrap, e.g.:"
echo "  python3 compare_revision_results.py --paired \\"
echo "    $OUT_ROOT/<cohort>/film.json $OUT_ROOT/<cohort>/dosecond.json"
