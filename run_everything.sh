#!/bin/bash

# Define the range of seeds you want to use
START_SEED=100
END_SEED=201 # Run for seeds from 1 to 10

# --- Path Configuration ---
# Default values
DEFAULT_MAIN_OUTPUT_DIR="exp_run_all"
DEFAULT_MONOLIX_PATH="/Applications/MonolixSuite2024R1.app/Contents/Resources/monolixSuite"
DEFAULT_MODEL_FILE="path/to/your/model.txt"

# Parse command-line arguments
MAIN_OUTPUT_DIR="$DEFAULT_MAIN_OUTPUT_DIR"
MONOLIX_PATH="$DEFAULT_MONOLIX_PATH"
MODEL_FILE="$DEFAULT_MODEL_FILE"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --main-output-dir) MAIN_OUTPUT_DIR="$2"; shift ;;
        --monolix-path) MONOLIX_PATH="$2"; shift ;;
        --model-file) MODEL_FILE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Define the directory where generated data will be stored
DATA_DIR="$MAIN_OUTPUT_DIR/data"


# Define your Python script names
GENERATE_DATA='gen_tacro.py'
R_SCRIPT_TEST='all_run_tacro.R'
PYTHON_SCRIPT_TRAIN="run_models.py"
PYTHON_SCRIPT_TEST="test_model.py"
PYTHON_SCRIPT_ANALYZE="analyse_std.py" # New script for analysis

# Define your base commands
BASE_COMMAND_TRAIN="$PYTHON_SCRIPT_TRAIN --niters 3000 -n 200 -s 40 -l 10 --dataset PK_Tacro --latent-ode --noise-weight 0.01 --max-t 5."
BASE_COMMAND_TEST="$PYTHON_SCRIPT_TEST -n 200 -s 40 -l 10 --dataset PK_Tacro --latent-ode --noise-weight 0.01 --max-t 5."

# Create a file to store all test results for later analysis
ALL_TEST_RESULTS_FILE="$MAIN_OUTPUT_DIR/test_gen_tacro_corrected.txt"
> "$ALL_TEST_RESULTS_FILE" # Clear the file if it already exists

echo "Starting multiple experiment runs and collecting test results..."

for SEED in $(seq $START_SEED $END_SEED)
do
    # Generate a random experiment ID for each run
    EXPERIMENT_ID=$(( RANDOM % 90000 + 10000 )) # Generates a 5-digit random number

    # Define experiment-specific output directory
    EXPERIMENT_DIR="$MAIN_OUTPUT_DIR/$EXPERIMENT_ID"
    mkdir -p "$EXPERIMENT_DIR"

    if [ "$SEED" -eq 1 ]; then
        NUM_PATIENTS=1000
        FIRST_AT=1
    else
        NUM_PATIENTS=200
        FIRST_AT=0
    fi

    echo "----------------------------------------------------"
    echo "Running experiment with Seed: $SEED, Experiment ID: $EXPERIMENT_ID"

    echo "Generating data..."
    python3 $GENERATE_DATA --num_patients $NUM_PATIENTS --output-dir "$DATA_DIR" --first_at $FIRST_AT

    # --- Step 1: Train the model ---
    echo "Training model..."
    Rscript $R_SCRIPT_TEST --virtual_cohort "$DATA_DIR/virtual_cohort_train.csv" \
                          --output_dir "$EXPERIMENT_DIR" \
                          --cores 10 \
                          --monolix_path "$MONOLIX_PATH" \
                          --model_file "$MODEL_FILE"

    FULL_COMMAND_TRAIN="$BASE_COMMAND_TRAIN --seed $SEED --save \"$EXPERIMENT_DIR\" --experiment $EXPERIMENT_ID"

    LOG_FILE_TRAIN="$EXPERIMENT_DIR/train_${PYTHON_SCRIPT_TRAIN%.py}.log"
    echo "Training started for Seed $SEED, ID $EXPERIMENT_ID."
    python3 $FULL_COMMAND_TRAIN > "$LOG_FILE_TRAIN" 2>&1
    TRAIN_EXIT_CODE=$?

    if [ $TRAIN_EXIT_CODE -ne 0 ]; then
        echo "Error: Training for Seed $SEED, ID $EXPERIMENT_ID failed. Check $LOG_FILE_TRAIN"
        continue # Skip to the next seed if training failed
    fi
    echo "Training complete for Seed $SEED, ID $EXPERIMENT_ID."

    # --- Step 2: Test the trained model ---
    echo "Testing model..."
    echo "--- Results for Seed $SEED (Experiment ID: $EXPERIMENT_ID) ---" >> "$ALL_TEST_RESULTS_FILE"
    python3 $BASE_COMMAND_TEST --seed $SEED --load "$EXPERIMENT_DIR/experiment_${EXPERIMENT_ID}.ckpt" --save "$EXPERIMENT_DIR" --experiment $EXPERIMENT_ID | \
        grep -E "Error comparison:|RMSE comparison:|- error.*=|- error_be.*=|- rmse.*=|- rmse_be.*=" >> "$ALL_TEST_RESULTS_FILE"
    TEST_EXIT_CODE=$?
    
    if [ $TEST_EXIT_CODE -ne 0 ]; then
        echo "Error: Testing for Seed $SEED, ID $EXPERIMENT_ID failed."
    fi
    echo "Testing complete for Seed $SEED, ID $EXPERIMENT_ID."

    # Optional: Add a small delay between runs
    sleep 1
done

echo "----------------------------------------------------"
echo "All experiment runs initiated and test results collected in $ALL_TEST_RESULTS_FILE."

# --- Step 3: Analyze the collected results ---
echo "Running analysis script..."
python3 "$PYTHON_SCRIPT_ANALYZE" "$ALL_TEST_RESULTS_FILE"

echo "----------------------------------------------------"
echo "Analysis complete. Check the output above and in the logs directory."
echo "To monitor progress, you can use 'tail -f $MAIN_OUTPUT_DIR/*/train_run_models.log' or 'tensorboard --logdir $MAIN_OUTPUT_DIR/runs'"