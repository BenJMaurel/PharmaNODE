# Installation Guide

This document provides instructions for setting up the necessary environment to run the scripts in this project.

## Python Dependencies

To install the required Python packages, you can use pip and the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

This will install all required Python packages for:

- Core training, evaluation, and analysis scripts.
- Optional visualizations (UMAP, Plotly/Dash dashboards).

Key packages include:

- torch
- torchdiffeq
- numpy
- matplotlib
- pandas
- scikit-learn
- seaborn
- tensorboard
- scipy
- umap-learn
- plotly
- dash
- chardet
- torchvision

## R Dependencies

To install the required R packages, open an R console and run the following command:

```R
install.packages(c("argparse", "tidyverse", "mrgsolve", "mapbayr", "MESS", "lixoftConnectors", "glue", "furrr"))
```

This will install the following packages:
- argparse
- tidyverse
- mrgsolve
- mapbayr
- MESS
- lixoftConnectors
- glue
- furrr

## Other Software

### Monolix

This project requires Monolix for pharmacokinetic/pharmacodynamic (PK/PD) modeling.

1.  **Download and Install Monolix:**
    - Visit the Lixoft website: [https://lixoft.com/products/monolix-suite/](https://lixoft.com/products/monolix-suite/)
    - Download and install the appropriate version of Monolix Suite for your operating system.

2.  **Set the Monolix Path:**
    - The `run_everything.sh` script requires the path to your Monolix installation.
    - You must provide this path using the `--monolix-path` command-line argument.
    - For example:
      ```bash
      bash run_everything.sh --monolix-path "/Applications/MonolixSuite2024R1.app/Contents/Resources/monolixSuite"
      ```
      or on Linux:
      ```bash
      bash run_everything.sh --monolix-path "/opt/monolixSuite2024R1/monolixSuite"
      ```