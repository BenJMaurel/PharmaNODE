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

## Reproduce paper results (different scenarios / seeds)

If your goal is to **recreate the results section with the different scenarios**, you should run:

```bash
bash run_everything.sh
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

**Note:** `lixoftConnectors` is crucial for interfacing R with Monolix. Ensure it is correctly installed and compatible with your Monolix version. If your Monolix installation is not auto-detected, you may need to edit the R-side configuration (see the R scripts invoked by `run_everything.sh`, e.g. `all_run_tacro.r`).

### Monolix

A working installation of Monolix is required to run the comparative pharmacometric models. You can find installation instructions on the [Lixoft website](https://lixoft.com/products/monolix-suite/).

When running the full pipeline (`run_everything.sh`), the R scripts must be able to locate your Monolix installation (specifically the `monolixSuite` executable or the directory containing it). If it’s not detected automatically on your system, set the path in the R-side configuration used by the scripts (e.g. `all_run_tacro.r`).

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

To run the full pipeline, execute the script from the root of the repository:

```bash
bash run_everything.sh
```

### Workflow

The script executes the following steps sequentially for a range of random seeds:

1.  **Data Generation (`gen_tacro.py`)**: Generates synthetic patient data with specified parameters (e.g., number of patients, first dose time).
2.  **Monolix Training (`all_run_tacro.R`)**: Uses R and the `lixoftConnectors` package to train a Monolix model on the generated data. This step requires a valid Monolix installation path.
3.  **NODE Training (`run_models.py`)**: Trains the Neural ODE model using the Python environment.
4.  **Testing (`test_model.py`)**: Evaluates the trained NODE model on a test set and records metrics like RMSE and prediction error.
5.  **Analysis (`analyse_std.py`)**: After all runs are complete, this script aggregates and analyzes the results from all seeds.

### Configuration

`run_everything.sh` is a plain bash script (it does not currently parse CLI flags). Customize it by editing the variables and command strings inside the file (seed range, scenario number, `BASE_COMMAND_TRAIN`, `BASE_COMMAND_TEST`, etc.).