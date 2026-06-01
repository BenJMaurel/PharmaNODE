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

## Data Requirements

The models use data from CSV files, which should be structured to be compatible with the `extract_gen_tac` function in `lib/read_tacro.py`. The input data is expected to have the following columns:

- **`ID`**: A unique identifier for each patient.
- **`TIME`**: The time of the observation, measured in hours.
- **`DV`**: The measured drug concentration at a given time (dependent variable).
- **`AMT`**: The dose amount administered to the patient.
- **`II`**: The dosing interval, indicating whether the treatment is immediate-release (`12`) or extended-release (`24`).
- **`CYP`**: The patient's CYP3A5 metabolizer status, where `1` typically indicates a normal metabolizer and `0` a poor metabolizer. This covariate is used to account for genetic variations in drug metabolism.
- **`AUC`**: The Area Under the Curve, representing the total drug exposure over a dosing interval. This is often the target variable for prediction.

The `gen_tacro.py` script can be used to generate synthetic data that follows this structure, which is useful for testing the models without access to clinical data.

## Pipeline Automation (`run_pipeline.sh`)

The main experimental workflow is managed by the `run_everything.sh` script. This script automates the entire lifecycle of an experiment, from data generation to analysis.

To run the full pipeline from the root of the repository (including the Monolix benchmark), use:

```bash
bash run_pipeline.sh \
```

It will install Monolix and lixoft connextors, and then call the `run_everything.sh` script

### Workflow

For each seed in the specified range, the `run_everything.sh` script executes:

1.  **Data Generation (`gen_tacro_film.py`)**: Generates synthetic FiLM datasets under `./results/exp_film_run/<EXP_ID>`.
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
- `--main-output-dir`: Root directory for experiment outputs (default: `./results`).

### Film Pipeline Automation (`run_film.sh`)

To run the full pipeline for the FiLM experimentation, use:

```bash
bash run_pipeline.sh \
```

This will create the data, train a model on the train part, test it and returns the metrics.

## Custom workflow (generate data → train → test/plots)

Use this if you want a **custom experiment** instead of the full “paper reproduction” pipeline. The safest way to get the **exact argument structure** is to copy the relevant command lines from [`run_everything.sh`](run_everything.sh).

### New drug tutorial (notebooks)

To adapt PharmaNODE to **another drug** (configurable compartments, absorption type, observation times, and sparse encoder inputs), we provide a 3-part step-by-step tutorial series in the `docs/` folder:

1. **Standard Pipeline:** [`docs/01_classic_pk_standard.ipynb`](docs/01_classic_pk_standard.ipynb)
   Learn how to use `DrugStudyConfig` and the generic `ClassicPKModel` to generate datasets and train a Latent ODE on single-visit data.
2. **FiLM Pipeline:** [`docs/02_classic_pk_film.ipynb`](docs/02_classic_pk_film.ipynb)
   Learn how to use the exact same config and generic ODE model to generate **paired visits** (Visit 1 and Visit 2) per patient and train a FiLM-conditioned Latent ODE.
3. **Advanced Custom ODEs:** [`docs/03_advanced_custom_ode_1_generate.ipynb`](docs/03_advanced_custom_ode_1_generate.ipynb) & [`docs/03_advanced_custom_ode_2_train.ipynb`](docs/03_advanced_custom_ode_2_train.ipynb)
   For advanced users who want to write their own custom PyTorch ODE class instead of using the generic `ClassicPKModel`.

The first two tutorials rely on [`lib/pk_drug.py`](lib/pk_drug.py) and the generic PK model in [`lib/classic_pk.py`](lib/classic_pk.py). After generating data under `./results/<exp_id>/` with `drug_config.json`, training still uses `--dataset PK_Tacro --experiment <exp_id>`; `parse_datasets` auto-loads the custom config when that file is present.

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
- `./results/exp_film_run/12345/virtual_cohort_film_train.csv`
- `./results/exp_film_run/12345/virtual_cohort_film_test.csv`

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

`test_model.py` loads a checkpoint by **experiment ID** (it expects `results/<ID>/experiment_<ID>.ckpt`).

```bash
python3 test_model.py \
  -n 200 -s 40 -l 10 \
  --dataset PK_Tacro --latent-ode \
  --noise-weight 0.01 --max-t 5. \
  --seed 101 \
  --load 12345
```

#### Test a FiLM model (`test_film.py`)

`test_film.py` loads a checkpoint by **path** (not by numeric ID). By default, FiLM training writes `results/exp_film_run/<ID>/experiment_film_<ID>.ckpt`.

```bash
python3 test_film.py \
  -n 200 -s 40 -l 10 \
  --dataset PK_Tacro --latent-ode \
  --noise-weight 0.01 --max-t 5. \
  --seed 101 \
  --exp 12345 \
  --load 12345
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

All outputs from the experimental runs are stored in the `./results` directory by default. Each execution of the `run_everything.sh` script generates a unique subdirectory within `./results`, named with a random 5-digit experiment ID.

Inside each experiment directory, you will find:

- **Log files**: `train_run_models.log` captures the output from the training process.
- **Model checkpoints**: The trained model weights are saved as `.ckpt` files (e.g., `experiment_12345.ckpt`).
- **Monolix project files**: The R script generates Monolix-compatible files for comparative modeling.
- **AUC predictions**: The predicted AUC values from the `mapbayr` R package are stored in `tacro_mapbayest_auc_YYYYMMDD.csv`.

Aggregated test results from all experiment runs are compiled into a single file: `./results/test_gen_tacro_corrected.txt`. This file contains key performance metrics, such as RMSE and prediction errors, making it easy to compare results across different runs.

### Minimal smoke test

You can also run a quick end‑to‑end test with reduced workload:

```bash
python3 gen_tacro_film.py --exp 12345 --num_patients 50 --scenario 2
python3 run_models.py --niters 10 -n 50 -s 40 -l 10 --dataset PK_Tacro --latent-ode --noise-weight 0.01 --max-t 5. --seed 101 --experiment 12345 --use_tacro
python3 test_film.py -n 50 -s 40 -l 10 --dataset PK_Tacro --latent-ode --noise-weight 0.01 --max-t 5. --seed 101 --load 12345
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
  - `--use_flow`: Uses a Normalizing Flow as the prior distribution for the latent space, allowing for a more complex and flexible posterior approximation. Not presented in the paper.

**Example: Training with GMM**

```bash
python run_models.py --dataset PK_Tacro --latent-ode --use_gmm --n_components 5 --save experiments/
```

**Example: Training with Normalizing Flows**

```bash
python run_models.py --dataset PK_Tacro --latent-ode --use_flow --save experiments/
```