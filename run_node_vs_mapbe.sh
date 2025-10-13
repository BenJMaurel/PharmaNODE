#!/bin/bash

# Script to compare Latent ODE (NODE) with MAP-BE (Monolix).

# --- Configuration ---
set -e

# --- Parameters ---
MAIN_OUTPUT_DIR="exp_node_vs_mapbe"
DATA_DIR="$MAIN_OUTPUT_DIR/data"
MONOLIX_PATH="/Applications/MonolixSuite2024R1.app/Contents/Resources/monolixSuite"
MODEL_FILE="oral1_2cpt_kaClV1QV2.txt"

# --- Command-Line Argument Parsing ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --monolix_path) MONOLIX_PATH="$2"; shift ;;
        --model_file) MODEL_FILE="$2"; shift ;;
        --output_dir) MAIN_OUTPUT_DIR="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# --- Setup ---
mkdir -p "$MAIN_OUTPUT_DIR"
mkdir -p "$DATA_DIR"

# --- Data Generation ---
echo "--- Generating Synthetic Data ---"
python3 gen_tacro.py --num_patients 200 --output-dir "$DATA_DIR"
echo "--- Data Generation Complete ---"

# --- MAP-BE (Monolix) Analysis ---
echo "--- Running MAP-BE Analysis ---"
Rscript all_run_tacro.R \
    --virtual_cohort "$DATA_DIR/virtual_cohort_train.csv" \
    --output_dir "$MAIN_OUTPUT_DIR" \
    --cores 10 \
    --monolix_path "$MONOLIX_PATH" \
    --model_file "$MODEL_FILE"
echo "--- MAP-BE Analysis Complete ---"

# --- Latent ODE (NODE) Training and Testing ---
echo "--- Running Latent ODE Model ---"
EXPERIMENT_ID=$(date +%s)
NODE_OUTPUT_DIR="$MAIN_OUTPUT_DIR/node_results"
mkdir -p "$NODE_OUTPUT_DIR"

# --- Training ---
python3 run_models.py \
    --dataset PK_Tacro \
    --train_datasets gen_tac \
    --train_data_path "$DATA_DIR/virtual_cohort_train.csv" \
    --latent-ode \
    --niters 3000 \
    --save "$NODE_OUTPUT_DIR" \
    --experiment "$EXPERIMENT_ID"

# --- Testing ---
MODEL_PATH="$NODE_OUTPUT_DIR/experiment_$EXPERIMENT_ID.ckpt"
python3 test_model.py \
    --dataset PK_Tacro \
    --test_datasets gen_tac \
    --test_data_path "$DATA_DIR/virtual_cohort_test.csv" \
    --load "$MODEL_PATH" \
    > "$NODE_OUTPUT_DIR/test_results.txt"

echo "--- Latent ODE Model Complete ---"
echo "Results saved in $NODE_OUTPUT_DIR"

# --- Final Comparison ---
echo "--- Comparison ---"
echo "MAP-BE results are in $MAIN_OUTPUT_DIR"
echo "Latent ODE results are in $NODE_OUTPUT_DIR"
echo "You can now compare the results from both models."