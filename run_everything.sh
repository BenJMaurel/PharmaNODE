#!/bin/bash

set -euo pipefail

# -----------------------------
# Default configuration values
# -----------------------------
START_SEED=101
END_SEED=201
SCENARIO=2
CORES=10
MAIN_OUTPUT_DIR="exp_run_all"
MONOLIX_PATH=""

# -----------------------------
# Parse CLI arguments
# -----------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --start-seed)
      START_SEED="$2"
      shift 2
      ;;
    --end-seed)
      END_SEED="$2"
      shift 2
      ;;
    --scenario)
      SCENARIO="$2"
      shift 2
      ;;
    --cores)
      CORES="$2"
      shift 2
      ;;
    --main-output-dir)
      MAIN_OUTPUT_DIR="$2"
      shift 2
      ;;
    --monolix-path)
      MONOLIX_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Supported flags: --start-seed --end-seed --scenario --cores --main-output-dir --monolix-path"
      exit 1
      ;;
  esac
done

# -----------------------------
# Script / command definitions
# -----------------------------
GENERATE_DATA="gen_tacro_film.py"
R_SCRIPT_TEST="all_run_tacro.r"
PYTHON_SCRIPT_TRAIN="run_models.py"
PYTHON_SCRIPT_TEST="test_model.py"
PYTHON_SCRIPT_ANALYZE="analyse_std.py" # Analysis script

BASE_COMMAND_R="$R_SCRIPT_TEST"
BASE_COMMAND_TRAIN="$PYTHON_SCRIPT_TRAIN --niters 6000 -n 200 -s 40 -l 10 --dataset PK_Tacro --latent-ode --noise-weight 0.01 --max-t 5. --use_film"
BASE_COMMAND_TEST="$PYTHON_SCRIPT_TEST -n 200 -s 40 -l 10 --dataset PK_Tacro --latent-ode --noise-weight 0.01 --max-t 5."

# Ensure main output directory exists
mkdir -p "$MAIN_OUTPUT_DIR"

# Create a file to store all test results for later analysis
ALL_TEST_RESULTS_FILE="$MAIN_OUTPUT_DIR/test_gen_tacro_corrected.txt"
> "$ALL_TEST_RESULTS_FILE" # Clear the file if it already exists

echo "Starting multiple experiment runs and collecting test results..."
echo "Seed range: $START_SEED to $END_SEED"
echo "Scenario: $SCENARIO"
echo "Output directory: $MAIN_OUTPUT_DIR"

for SEED in $(seq "$START_SEED" "$END_SEED"); do
    # Generate a random experiment ID for each run
    EXPERIMENT_ID=$(( RANDOM % 90000 + 10000 )) # Generates a 5-digit random number
    NUM_PATIENTS=1000
    FIRST_AT=1

    EXP_DIR="$MAIN_OUTPUT_DIR/$EXPERIMENT_ID"
    mkdir -p "$EXP_DIR"

    echo "----------------------------------------------------"
    echo "Running experiment with Seed: $SEED, Experiment ID: $EXPERIMENT_ID"
    echo "Generating data..."
    python3 "$GENERATE_DATA" --exp "$EXPERIMENT_ID" --num_patients "$NUM_PATIENTS" --first_at "$FIRST_AT" --scenario "$SCENARIO"

    # --- Step 1: Monolix / mapbayr analysis via R ---
    echo "Running R + Monolix analysis..."
    R_CMD=(Rscript "$BASE_COMMAND_R"
           --virtual_cohort "virtual_cohort_test.csv"
           --output_dir "$EXP_DIR"
           --experiment "$EXPERIMENT_ID"
           --cores "$CORES")
    if [[ -n "$MONOLIX_PATH" ]]; then
      R_CMD+=(--monolix_path "$MONOLIX_PATH")
    fi
    "${R_CMD[@]}"

    # --- Step 2: Train the NODE model ---
    echo "Training model..."
    FULL_COMMAND_TRAIN="$BASE_COMMAND_TRAIN --seed $SEED --experiment $EXPERIMENT_ID"
    LOG_FILE_TRAIN="logs/train_${PYTHON_SCRIPT_TRAIN%.py}_${EXPERIMENT_ID}.log"
    mkdir -p logs # Ensure logs directory exists
    echo "Training started for Seed $SEED, ID $EXPERIMENT_ID."
    python3 $FULL_COMMAND_TRAIN > "$LOG_FILE_TRAIN" 2>&1 || {
      echo "Error: Training for Seed $SEED, ID $EXPERIMENT_ID failed. Check $LOG_FILE_TRAIN"
      continue
    }
    echo "Training complete for Seed $SEED, ID $EXPERIMENT_ID."

    # --- Step 3: Test the trained model ---
    echo "Testing model..."
    echo "--- Results for Seed $SEED (Experiment ID: $EXPERIMENT_ID) ---" >> "$ALL_TEST_RESULTS_FILE"
    python3 $BASE_COMMAND_TEST --seed "$SEED" --load "$EXPERIMENT_ID" --experiment "$EXPERIMENT_ID" | \
        grep -E "Error comparison:|RMSE comparison:|- error.*=|- error_be.*=|- rmse.*=|- rmse_be.*=" >> "$ALL_TEST_RESULTS_FILE" || {
      echo "Error: Testing for Seed $SEED, ID $EXPERIMENT_ID failed."
    }
    echo "Testing complete for Seed $SEED, ID $EXPERIMENT_ID."

    # Optional: Add a small delay between runs
    sleep 1
done

echo "----------------------------------------------------"
echo "All experiment runs initiated and test results collected in $ALL_TEST_RESULTS_FILE."

# --- Step 4: Analyze the collected results ---
echo "Running analysis script..."
python3 "$PYTHON_SCRIPT_ANALYZE" "$ALL_TEST_RESULTS_FILE"

echo "----------------------------------------------------"
echo "Analysis complete. Check the output above and in the logs directory."
echo "To monitor progress, you can use 'tail -f logs/train_run_models_*.log' or 'tensorboard --logdir experiments/runs'"