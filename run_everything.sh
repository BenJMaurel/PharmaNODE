#!/bin/bash

# Define the range of seeds you want to use
START_SEED=101
END_SEED=201 

# Define your Python script names
GENERATE_DATA='gen_tacro.py'
R_SCRIPT_TEST='all_run_tacro.R'
PYTHON_SCRIPT_TRAIN="run_models.py"
PYTHON_SCRIPT_TEST="test_model.py"
PYTHON_SCRIPT_ANALYZE="analyse_std.py" # New script for analysis

# Define your base commands
BASE_COMMAND_R="$R_SCRIPT_TEST" 
BASE_COMMAND_TRAIN="$PYTHON_SCRIPT_TRAIN --niters 6000 -n 200 -s 40 -l 10 --dataset PK_Tacro --latent-ode --noise-weight 0.01 --max-t 5."
BASE_COMMAND_TEST="$PYTHON_SCRIPT_TEST -n 200 -s 40 -l 10 --dataset PK_Tacro --latent-ode --noise-weight 0.01 --max-t 5."
# Create a file to store all test results for later analysis
ALL_TEST_RESULTS_FILE="test_gen_tacro_corrected.txt"
> "$ALL_TEST_RESULTS_FILE" # Clear the file if it already exists

echo "Starting multiple experiment runs and collecting test results..."

for SEED in $(seq $START_SEED $END_SEED)
do
    # Generate a random experiment ID for each run
    EXPERIMENT_ID=$(( RANDOM % 90000 + 10000 )) # Generates a 5-digit random number
    if [ "$SEED" -eq 1 ]; then
        NUM_PATIENTS=100
        FIRST_AT=1
    else
        NUM_PATIENTS=100
        FIRST_AT=1
    fi
    mkdir -p "exp_run_all/$EXPERIMENT_ID"
    echo "----------------------------------------------------"
    echo "Running experiment with Seed: $SEED, Experiment ID: $EXPERIMENT_ID"
    echo "Generating data..."
    python3 $GENERATE_DATA --exp $EXPERIMENT_ID --num_patients $NUM_PATIENTS --first_at $FIRST_AT --scenario 3
    # --- Step 1: Train the model ---
    echo "Training model..."
    Rscript $BASE_COMMAND_R --virtual_cohort virtual_cohort_test.csv --output_dir exp_run_all/$EXPERIMENT_ID --experiment $EXPERIMENT_ID --cores 10
    FULL_COMMAND_TRAIN="$BASE_COMMAND_TRAIN --seed $SEED --experiment $EXPERIMENT_ID"
    # It's generally better to run training sequentially if it's resource-intensive
    # or if you want to ensure a model finishes training before testing it.
    # We redirect training output to its own log file.
    LOG_FILE_TRAIN="logs/train_${PYTHON_SCRIPT_TRAIN%.py}_${EXPERIMENT_ID}.log"
    mkdir -p logs # Ensure logs directory exists
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
    # We use 'grep' to filter only the lines containing "Error comparison" and "RMSE comparison"
    # and the lines with the actual error/rmse values. This keeps the results file clean.
    python3 $BASE_COMMAND_TEST --seed $SEED --load $EXPERIMENT_ID --experiment $EXPERIMENT_ID| \
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
echo "To monitor progress, you can use 'tail -f logs/train_run_models_*.log' or 'tensorboard --logdir experiments/runs'"