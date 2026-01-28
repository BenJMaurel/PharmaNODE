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

**Note:** `lixoftConnectors` is crucial for interfacing R with Monolix. Ensure it is correctly installed and compatible with your Monolix version. The `run_everything.sh` script will initialize this connector using the path provided via `--monolix-path`.

### Monolix

A working installation of Monolix is required to run the comparative pharmacometric models. You can find installation instructions on the [Lixoft website](https://lixoft.com/products/monolix-suite/).

When running the full pipeline (`run_everything.sh`), you must provide the path to the Monolix installation directory (specifically the `monolixSuite` executable or directory containing it) so that the R scripts can locate and utilize the Monolix engine.

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

You can customize the behavior of the script using the following command-line arguments:

- `--main-output-dir <path>`: Specifies the main directory where all experiment outputs will be stored. Defaults to `exp_run_all`.
- `--monolix-path <path>`: **Critical.** Sets the path to your Monolix installation. This is passed to the R scripts to initialize `lixoftConnectors`. Defaults to `/Applications/MonolixSuite2024R1.app/Contents/Resources/monolixSuite`. You should update this to point to your specific Monolix installation.
- `--model-file <path>`: Defines the path to the Monolix model file (`.txt`) to be used for the analysis.

### Example Usage

```bash
bash run_everything.sh \
  --main-output-dir "results_v1" \
  --monolix-path "/opt/MonolixSuite2024R1/lib/monolixSuite" \
  --model-file "models/tacrolimus_model.txt"
```

## Generative Models & Priors

The repository supports different strategies for the generative prior distribution of the latent state $z_0$. Choosing the right prior is crucial for the quality of generated time series.

### 1. Standard VAE (Standard Normal Prior)

**Usage:** Default behavior of `run_models.py`.

In the standard setting, the Latent ODE model acts as a Variational Autoencoder (VAE) where the approximate posterior $q(z_0|x)$ is regularized towards a standard normal prior $p(z_0) = \mathcal{N}(0, I)$.

- **Pros:** Simple, stable training, well-understood.
- **Cons:** May be too restrictive for complex, multi-modal latent distributions (the "posterior collapse" problem or aggregated posterior mismatch).

### 2. GMM Prior (Gaussian Mixture Model)

**Usage:** Post-hoc fitting using `fit_gaussian.py`.

To handle multi-modal latent structures (e.g., distinct patient subpopulations), you can fit a Gaussian Mixture Model (GMM) to the latent embeddings of a trained model. This approach decouples the representation learning from the density estimation.

**Workflow:**
1.  Train a standard model using `run_models.py`. Note the Experiment ID (e.g., `12345`).
2.  Run `fit_gaussian.py` to fit a GMM on the learned latent space and generate new samples.

```bash
python fit_gaussian.py --load 12345 --dataset PK_Tacro --save experiments/
```

- **Pros:** Can capture multi-modal distributions effectively; separates training stability from generation quality.
- **Cons:** Two-stage process; the GMM is not optimized end-to-end with the ODE.

### 3. Normalizing Flows

**Usage:** *Conceptual comparison / Advanced extension.*

Normalizing Flows (e.g., Continuous Normalizing Flows or CNFs) transform a simple base distribution (like a Gaussian) into a complex posterior using a sequence of invertible mappings. In the context of Latent ODEs, a Flow can be used as a flexible prior $p(z_0)$ or to model the approximate posterior $q(z_0|x)$.

- **Comparison:**
    - **vs. Standard VAE:** Flows are far more expressive, allowing the model to learn non-Gaussian latent densities end-to-end.
    - **vs. GMM:** Unlike the post-hoc GMM, Flows are typically trained jointly with the model, potentially leading to better aligned latent spaces. However, they are computationally more expensive and harder to train.

*Note: The current provided scripts (`fit_gaussian.py`) implement the GMM approach for improved generation. Flow-based priors would require modifying the `z0_prior` in the model definition.*