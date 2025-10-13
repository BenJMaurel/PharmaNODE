# Installation Guide

This document provides instructions for setting up the necessary environment to run the scripts in this project.

## Python Dependencies

To install the required Python packages, you can use pip and the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

This will install the following packages:
- torch
- torchdiffeq
- numpy
- matplotlib
- pandas
- scikit-learn
- seaborn
- tensorboard

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
    - You can provide this path using the `--monolix-path` command-line argument.
    - For example:
      ```bash
      ./run_everything.sh --monolix-path "/path/to/your/MonolixSuite/lixoft/monolix/bin"
      ```