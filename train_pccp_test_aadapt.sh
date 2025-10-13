#!/bin/bash

# Script to train on the PCCP dataset and test on the AADAPT dataset.

# --- Configuration ---
set -e

# --- Paths and Parameters ---
MAIN_OUTPUT_DIR="exp_train_pccp_test_aadapt"
TRAIN_DATA_PATH="mr4_tls_pccp.csv"
TEST_DATA_PATH="AADAPT_proadv_281016.csv"
PHARMAC_PATH="pharmac_data.csv" # Default path, can be overridden
AUC_PATH="auc_data.csv"       # Default path, can be overridden

# --- Command-Line Argument Parsing ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --train_data_path) TRAIN_DATA_PATH="$2"; shift ;;
        --test_data_path) TEST_DATA_PATH="$2"; shift ;;
        --pharmac_path) PHARMAC_PATH="$2"; shift ;;
        --auc_path) AUC_PATH="$2"; shift ;;
        --output_dir) MAIN_OUTPUT_DIR="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# --- Setup ---
mkdir -p "$MAIN_OUTPUT_DIR"
EXPERIMENT_ID=$(date +%s)
MODEL_PATH="$MAIN_OUTPUT_DIR/experiment_$EXPERIMENT_ID.ckpt"


# --- Model Training ---
echo "--- Starting Training on PCCP Dataset ---"
python3 run_models.py \
    --dataset PK_Tacro \
    --train_datasets pccp \
    --train_data_path "$TRAIN_DATA_PATH" \
    --pharmac_path "$PHARMAC_PATH" \
    --auc_path "$AUC_PATH" \
    --latent-ode \
    --niters 3000 \
    --batch-size 50 \
    --lr 0.01 \
    --save "$MAIN_OUTPUT_DIR" \
    --experiment "$EXPERIMENT_ID"

echo "--- Training Complete ---"
echo "Model saved in $MODEL_PATH"


# --- Model Testing ---
echo "--- Starting Testing on AADAPT Dataset ---"
python3 test_model.py \
    --dataset PK_Tacro \
    --test_datasets aadapt \
    --test_data_path "$TEST_DATA_PATH" \
    --pharmac_path "$PHARMAC_PATH" \
    --auc_path "$AUC_PATH" \
    --latent-ode \
    --load "$MODEL_PATH" \
    > "$MAIN_OUTPUT_DIR/test_results.txt"

echo "--- Testing Complete ---"
echo "Test results saved in $MAIN_OUTPUT_DIR/test_results.txt"