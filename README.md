# Improved Tacrolimus AUC Prediction Using Neural Ordinary Differential Equations

This repository contains the official code for the paper "Improved Tacrolimus AUC Prediction Using Neural Ordinary Differential Equations" (2025). It provides a comprehensive framework for predicting Tacrolimus Area Under the Curve (AUC) by leveraging Neural Ordinary Differential Equations (NODEs).

## Project Overview

The primary goal of this project is to demonstrate the effectiveness of NODEs in predicting drug concentration profiles and to compare their performance against traditional pharmacometric modeling approaches, specifically Monolix. This repository includes all necessary scripts to:

- **Generate synthetic patient data** for model training and evaluation.
- **Train NODE-based models** on pharmacokinetic (PK) data.
- **Evaluate model performance** by comparing predicted AUC values against established benchmarks.
- **Run a full experimental pipeline**, from data generation to results analysis.

The codebase is designed to be modular, allowing for experimentation with different model configurations and datasets.

<p align="center">
<img align="middle" src="./assets/viz.gif" width="800" />
</p>

## Prerequisites

Before running the experiments, ensure you have the required dependencies installed.

### Python Dependencies

Install the necessary Python packages using pip:

```bash
pip install -r requirements.txt
```

Additionally, `torchdiffeq` must be installed from its source:

```bash
pip install git+https://github.com/rtqichen/torchdiffeq.git
```

### R Dependencies

The project also requires R and several packages for data processing and modeling. Ensure you have the following R packages installed:

- `argparse`
- `tidyverse`
- `mrgsolve`
- `mapbayr`
- `MESS`
- `lixoftConnectors`
- `glue`
- `furrr`

You can install them in R using the following command:

```R
install.packages(c("argparse", "tidyverse", "mrgsolve", "mapbayr", "MESS", "lixoftConnectors", "glue", "furrr"))
```

### Monolix

A working installation of Monolix is required to run the comparative pharmacometric models. You can find installation instructions on the [Lixoft website](https://lixoft.com/products/monolix-suite/).

## Data Requirements

The models consume data from CSV files, which should be structured to be compatible with the `extract_gen_tac` function in `lib/read_tacro.py`. The input data is expected to have the following columns:

- **`ID`**: A unique identifier for each patient.
- **`TIME`**: The time of the observation, measured in hours.
- **`DV`**: The measured drug concentration at a given time (dependent variable).
- **`AMT`**: The dose amount administered to the patient.
- **`II`**: The dosing interval, indicating whether the treatment is immediate-release (`12`) or extended-release (`24`).
- **`CYP`**: The patient's CYP3A5 metabolizer status, where `1` typically indicates a normal metabolizer and `0` a poor metabolizer. This covariate is used to account for genetic variations in drug metabolism.
- **`AUC`**: The Area Under the Curve, representing the total drug exposure over a dosing interval. This is often the target variable for prediction.

The `gen_tacro.py` script can be used to generate synthetic data that follows this structure, which is useful for testing the models without access to clinical data.

## Output and Results

All outputs from the experimental runs are stored in the `exp_run_all` directory by default. Each execution of the `run_everything.sh` script generates a unique subdirectory within `exp_run_all`, named with a random 5-digit experiment ID.

Inside each experiment directory, you will find:

- **Log files**: `train_run_models.log` captures the output from the training process.
- **Model checkpoints**: The trained model weights are saved as `.ckpt` files (e.g., `experiment_12345.ckpt`).
- **Monolix project files**: The R script generates Monolix-compatible files for comparative modeling.
- **AUC predictions**: The predicted AUC values from the `mapbayr` R package are stored in `tacro_mapbayest_auc_YYYYMMDD.csv`.

Aggregated test results from all experiment runs are compiled into a single file: `exp_run_all/test_gen_tacro_corrected.txt`. This file contains key performance metrics, such as RMSE and prediction errors, making it easy to compare results across different runs.

## Running Experiments

The main experimental workflow is managed by the `run_everything.sh` script. This script automates data generation, model training, and evaluation.

To run the full pipeline, execute the script from the root of the repository:

```bash
bash run_everything.sh
```

### Command-Line Arguments

You can customize the behavior of the script using the following command-line arguments:

- `--main-output-dir <path>`: Specifies the main directory where all experiment outputs will be stored. Defaults to `exp_run_all`.
- `--monolix-path <path>`: Sets the path to your Monolix installation. Defaults to `/Applications/MonolixSuite2024R1.app/Contents/Resources/monolixSuite`.
- `--model-file <path>`: Defines the path to the Monolix model file to be used for the analysis.

Example of a customized run:

```bash
bash run_everything.sh --main-output-dir /path/to/my/results --monolix-path /opt/monolix/bin/Monolix --model-file /path/to/my/model.txt
```