# Improved Tacrolimus AUC Prediction Using Neural Ordinary Differential Equations

This repository contains the official code for the paper "Improved Tacrolimus AUC Prediction Using Neural Ordinary Differential Equations" (2025). It provides a comprehensive framework for predicting Tacrolimus Area Under the Curve (AUC) by leveraging Neural Ordinary Differential Equations (NODEs).

## Project Overview

The primary goal of this project is to demonstrate the effectiveness of NODEs in predicting drug concentration profiles and to compare their performance against traditional pharmacometric modeling approaches, specifically Monolix. This repository includes all necessary scripts to:

- **Generate synthetic patient data** for model training and evaluation.
- **Train NODE-based models** on pharmacokinetic (PK) data.
- **Evaluate model performance** by comparing predicted AUC values against established benchmarks.
- **Run a full experimental pipeline**, from data generation to results analysis.
- **Visualize the latent space for interpretation**

The codebase is designed to be modular, allowing for experimentation with different model configurations and datasets.

<!-- If you have a local visualization asset, you can re-enable it here. -->

## Quickstart (from a clean environment)

### 1. Create and activate a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Python dependencies

All required Python packages (including `torchdiffeq`, `scipy`, `umap-learn`, `plotly`, `dash`, etc.) are listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

> **Note:** You may prefer to install a GPU-enabled build of PyTorch following the official instructions and then run `pip install -r requirements.txt` with `torch` removed from that file.

### 3. Install R packages and Monolix (for full paper reproduction)

To reproduce the **full pipeline including the Monolix benchmark**, you also need R, the R packages listed below, and a working Monolix installation (see **Prerequisites** for details). If you only want to run the Python NODE models and plots, R/Monolix are not strictly required.

## Reproduce paper results (different scenarios / seeds)

If your goal is to **recreate the results section with the different scenarios**, you should run:

```bash
bash run_everything.sh --monolix-path "/absolute/path/to/monolixSuite"
```

This script is the **reference pipeline** used to reproduce the multi-run results: it loops over a seed range, generates data for a scenario, trains the model(s), runs evaluation, aggregates metrics, and finally runs the analysis.

To change the scenarios/seeds/hyperparameters, edit the variables at the top of [`run_everything.sh`](run_everything.sh) (e.g. `START_SEED`, `END_SEED`, the `--scenario` passed to the generator, and the `BASE_COMMAND_*` strings).

## Prerequisites

Before running the experiments, ensure you have the required dependencies installed.

### Python Dependencies

Install the necessary Python packages using pip:

```bash
pip install -r requirements.txt
```

This installs all Python dependencies needed for:

- **Core pipeline**: data generation, NODE model training and testing, and basic analysis scripts.
- **Optional extras**: UMAP-based latent visualizations, interactive Plotly/Dash dashboards, and the analysis utilities.

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

**Note:** `lixoftConnectors` is crucial for interfacing R with Monolix. Ensure it is correctly installed and compatible with your Monolix version. If your Monolix installation is not auto-detected, you may need to edit the R-side configuration (see the R scripts invoked by `run_everything.sh`, e.g. `all_run_tacro.r`).

### Monolix

A working installation of Monolix is required to run the comparative pharmacometric models and reproduce the paper’s benchmark tables.

- Install Monolix Suite from the [Lixoft website](https://lixoft.com/products/monolix-suite/).
- Locate the `monolixSuite` executable (or the directory containing it):
  - **macOS example**: `/Applications/MonolixSuite2024R1.app/Contents/Resources/monolixSuite`
  - **Linux example**: `/opt/monolixSuite2024R1/monolixSuite`

You pass this path to the pipeline via the `--monolix-path` flag of `run_everything.sh` (see **Pipeline Automation** below). Internally, it is forwarded to the R script (`all_run_tacro.r`) and used by `lixoftConnectors::initializeLixoftConnectors`.

### Guix

For a reproducible development environment, you can use Guix. Install the required dependencies by running the following command:

```bash
guix shell -m guix.scm
```

This will create a shell with all the necessary Python and R packages.

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

## Custom workflow (generate data → train → test/plots)

Use this if you want a **custom experiment** instead of the full “paper reproduction” pipeline. The safest way to get the **exact argument structure** is to copy the relevant command lines from [`run_everything.sh`](run_everything.sh).

### 1) Generate data

#### Standard Tacrolimus dataset

`gen_tacro.py` generates a single-visit dataset.

```bash
python3 gen_tacro.py --exp 12345 --num_patients 1000 --first_at 1 --scenario 3
```

#### FiLM dataset (2 visits per patient)

`gen_tacro_film.py` generates **two visits per patient** (Visit 1 and Visit 2) so you can compare them (dose extrapolation).

```bash
python3 gen_tacro_film.py --exp 12345 --num_patients 1000 --scenario 2
```

This writes FiLM CSVs under:
- `exp_run_all/exp_film_run/12345/virtual_cohort_film_train.csv`
- `exp_run_all/exp_film_run/12345/virtual_cohort_film_test.csv`

### 2) Train a model (`run_models.py`)

#### Standard (non-FiLM) training

```bash
python3 run_models.py \
  --niters 6000 -n 200 -s 40 -l 10 \
  --dataset PK_Tacro --latent-ode \
  --noise-weight 0.01 --max-t 5. \
  --seed 101 \
  --experiment 12345
```

#### FiLM training (paired visits)

Add `--use_film` and ensure `--experiment` points to the FiLM dataset folder name you generated above.

```bash
python3 run_models.py \
  --niters 6000 -n 200 -s 40 -l 10 \
  --dataset PK_Tacro --latent-ode --use_film \
  --noise-weight 0.01 --max-t 5. \
  --seed 101 \
  --experiment 12345
```

### 3) Evaluate / test

#### Test a standard model (`test_model.py`)

`test_model.py` loads a checkpoint by **experiment ID** (it expects `experiments/experiment_<ID>.ckpt`).

```bash
python3 test_model.py \
  -n 200 -s 40 -l 10 \
  --dataset PK_Tacro --latent-ode \
  --noise-weight 0.01 --max-t 5. \
  --seed 101 \
  --load 12345
```

#### Test a FiLM model (`test_film.py`)

`test_film.py` loads a checkpoint by **path** (not by numeric ID). By default, FiLM training writes `experiments/experiment_film_<ID>.ckpt`.

```bash
python3 test_film.py \
  -n 200 -s 40 -l 10 \
  --dataset PK_Tacro --latent-ode \
  --noise-weight 0.01 --max-t 5. \
  --seed 101 \
  --exp exp_film_run/12345 \
  --load experiments/experiment_film_12345.ckpt
```

### 4) Plots

Once trained/tested, you can create plots using the existing plotting scripts (entry points may vary depending on what you want to visualize):
- `plot_test.py`
- `various_plot.py`
- `plot_latent_space.py`
- `visualize_scenari.py`

## Latent space visualization / sampling

Example (copy-paste) command to visualize / generate samples from the latent space:

```bash
python3 generate_from_latent.py \
  -n 200 -s 40 -l 10 \
  --dataset PK_Tacro --latent-ode \
  --noise-weight 0.01 --max-t 5. \
  --seed 1 \
  --load 37614 \
  --experiment 37614
```

## Output and Results

All outputs from the experimental runs are stored in the `exp_run_all` directory by default. Each execution of the `run_everything.sh` script generates a unique subdirectory within `exp_run_all`, named with a random 5-digit experiment ID.

Inside each experiment directory, you will find:

- **Log files**: `train_run_models.log` captures the output from the training process.
- **Model checkpoints**: The trained model weights are saved as `.ckpt` files (e.g., `experiment_12345.ckpt`).
- **Monolix project files**: The R script generates Monolix-compatible files for comparative modeling.
- **AUC predictions**: The predicted AUC values from the `mapbayr` R package are stored in `tacro_mapbayest_auc_YYYYMMDD.csv`.

Aggregated test results from all experiment runs are compiled into a single file: `exp_run_all/test_gen_tacro_corrected.txt`. This file contains key performance metrics, such as RMSE and prediction errors, making it easy to compare results across different runs.

### Minimal smoke test (from a fresh environment)

After installing dependencies as described above, you can run a quick end‑to‑end test with reduced workload:

```bash
python3 gen_tacro_film.py --exp 12345 --num_patients 50 --scenario 2
python3 run_models.py --niters 10 -n 50 -s 40 -l 10 --dataset PK_Tacro --latent-ode --noise-weight 0.01 --max-t 5. --seed 101 --experiment 12345
python3 test_model.py -n 50 -s 40 -l 10 --dataset PK_Tacro --latent-ode --noise-weight 0.01 --max-t 5. --seed 101 --load 12345
```

This sequence:

- Generates a small synthetic FiLM cohort.
- Trains a NODE model for a few iterations.
- Runs the evaluation script on the resulting checkpoint to verify everything is wired correctly.

## Running Individual Models (`run_models.py`)

The `run_models.py` script is the core Python entry point for training and evaluating the Neural ODE models and baselines. It allows for fine-grained control over model architecture, training hyperparameters, and dataset selection.

### Key Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `-n` | int | 100 | Size of the dataset (number of training examples). |
| `--niters` | int | 1000 | Number of training iterations/epochs. |
| `--lr` | float | 1e-2 | Starting learning rate. |
| `-b`, `--batch-size` | int | 200 | Batch size for training. |
| `--dataset` | str | `periodic` | Dataset to load (`physionet`, `activity`, `hopper`, `periodic`, `PK_Tacro`, etc.). |
| `--latent-ode` | flag | False | Run the Latent ODE seq2seq model. |
| `--ode-rnn` | flag | False | Run the ODE-RNN baseline model. |
| `--rnn-vae` | flag | False | Run the RNN-VAE baseline model. |
| `--save` | str | `experiments/` | Directory path to save model checkpoints and logs. |
| `--load` | str | None | Experiment ID to load for evaluation (if `None`, starts a new experiment). |
| `--viz` | flag | False | Enable real-time plotting during training (requires display). |
| `--seed` | int | 15 | Random seed for reproducibility. |
| `--noise-weight` | float | 0.04 | Noise amplitude for generated trajectories. |

### Example Usage

To train a Latent ODE model on the `PK_Tacro` dataset with specific parameters:

```bash
python run_models.py --niters 3000 -n 200 -s 40 -l 10 --dataset PK_Tacro --latent-ode --noise-weight 0.01 --max-t 5. --save experiments/
```

To load an existing experiment (e.g., ID 12345) for evaluation:

```bash
python run_models.py --dataset PK_Tacro --load 12345 --save experiments/
```

### Generative Modeling (GMM & Flows)

In addition to the standard Latent ODE with a standard normal prior, the model supports more complex priors for the latent space, which can improve generative performance and clustering.

- **Gaussian Mixture Models (GMM)**:
  - `--use_gmm`: Initializes a Latent ODE with a GMM prior. The clusters are initialized using K-Means on the latent embeddings after a warm-up period.
  - `--use_gmm_v`: A variant of the GMM prior model that allows for additional flexibility (e.g., learnable rotations) and re-initialization during training.
  - `-nc`, `--n_components`: Specifies the number of Gaussian components (clusters) in the latent space. Default is `4`.

- **Normalizing Flows**:
  - `--use_flow`: Uses a Normalizing Flow as the prior distribution for the latent space, allowing for a more complex and flexible posterior approximation.

**Example: Training with GMM**

```bash
python run_models.py --dataset PK_Tacro --latent-ode --use_gmm --n_components 5 --save experiments/
```

**Example: Training with Normalizing Flows**

```bash
python run_models.py --dataset PK_Tacro --latent-ode --use_flow --save experiments/
```

## Pipeline Automation (`run_everything.sh`)

The main experimental workflow is managed by the `run_everything.sh` script. This script automates the entire lifecycle of an experiment, from data generation to analysis.

To run the full pipeline from the root of the repository (including the Monolix benchmark), use:

```bash
bash run_everything.sh \
  --monolix-path "/absolute/path/to/monolixSuite" \
  --start-seed 101 \
  --end-seed 201 \
  --scenario 2 \
  --cores 10 \
  --main-output-dir "exp_run_all"
```

At minimum you must provide a valid `--monolix-path` for your system; the other flags have sensible defaults.

### Workflow

For each seed in the specified range, the script executes:

1.  **Data Generation (`gen_tacro_film.py`)**: Generates synthetic FiLM datasets under `exp_run_all/exp_film_run/<EXP_ID>`.
2.  **Monolix Training (`all_run_tacro.r`)**: Uses R and the `lixoftConnectors` package to configure and run a Monolix model on the generated data, writing outputs into `<main-output-dir>/<EXP_ID>`.
3.  **NODE Training (`run_models.py`)**: Trains the Neural ODE model using the Python environment.
4.  **Testing (`test_model.py`)**: Evaluates the trained NODE model on a test set and records metrics like RMSE and prediction error.
5.  **Analysis (`analyse_std.py`)**: After all runs are complete, this script aggregates and analyzes the results from all seeds.

### Configuration

`run_everything.sh` now accepts the following CLI flags:

- `--monolix-path` (required for full pipeline): Directory containing the `monolixSuite` executable.
- `--start-seed`, `--end-seed`: Inclusive seed range for repeated experiments.
- `--scenario`: Scenario index passed to `gen_tacro_film.py` (e.g. `2` or `3`, as in the paper).
- `--cores`: Number of CPU cores to pass to the R analysis (`all_run_tacro.r`).
- `--main-output-dir`: Root directory for experiment outputs (default: `exp_run_all`).