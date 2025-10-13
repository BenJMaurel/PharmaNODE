#!/bin/bash

# Script to perform cross-validation on a given dataset.

# --- Configuration ---
set -e

# --- Parameters ---
MAIN_OUTPUT_DIR="exp_cross_validation"
DATASET_FILE="mr4_tls_pccp.csv"
N_ITERATIONS=50
RESULTS_FILE="$MAIN_OUTPUT_DIR/cross_validation_results.txt"

# --- Command-Line Argument Parsing ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset_file) DATASET_FILE="$2"; shift ;;
        --output_dir) MAIN_OUTPUT_DIR="$2"; shift ;;
        --iterations) N_ITERATIONS="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# --- Setup ---
mkdir -p "$MAIN_OUTPUT_DIR"
> "$RESULTS_FILE"

# --- Cross-Validation Loop ---
echo "--- Starting Cross-Validation ---"
for i in $(seq 1 $N_ITERATIONS)
do
    echo "--- Iteration $i of $N_ITERATIONS ---"

    ITERATION_DIR="$MAIN_OUTPUT_DIR/iteration_$i"
    SPLIT_DIR="$ITERATION_DIR/split"
    mkdir -p "$SPLIT_DIR"

    # Generate a new random 80/20 split for each iteration
    python3 split_dataset.py \
        --file_path "$DATASET_FILE" \
        --output_dir "$SPLIT_DIR" \
        --test_size 0.2 \
        --random_seed $i

    TRAIN_FILE="$SPLIT_DIR/train_split.csv"
    TEST_FILE="$SPLIT_DIR/test_split.csv"

    # --- Training ---
    EXPERIMENT_ID=$(date +%s)_$i
    MODEL_PATH="$ITERATION_DIR/model.ckpt"

    python3 run_models.py \
        --dataset PK_Tacro \
        --train_datasets pccp \
        --train_data_path "$TRAIN_FILE" \
        --latent-ode \
        --niters 1000 \
        --save "$ITERATION_DIR" \
        --experiment "$EXPERIMENT_ID"

    # --- Testing ---
    echo "--- Results for Iteration $i ---" >> "$RESULTS_FILE"
    python3 test_model.py \
        --dataset PK_Tacro \
        --test_datasets pccp \
        --test_data_path "$TEST_FILE" \
        --load "$MODEL_PATH" \
        >> "$RESULTS_FILE"

    echo "--- Iteration $i Complete ---"
done

echo "--- Cross-Validation Complete ---"
echo "Results saved in $RESULTS_FILE"