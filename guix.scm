(use-modules (gnu packages)
             (gnu packages python-xyz)
             (gnu packages r-cran))

(specifications->manifest
 (list "python"
       "python-torch"
       "python-pytorch-torchdiffeq"
       "python-numpy"
       "python-matplotlib"
       "python-pandas"
       "python-scikit-learn"
       "python-seaborn"
       "python-tensorboard"
       "python-pytest"
       "python-umap-learn"
       "python-plotly"
       "python-dash"
       "python-chardet"
       "python-torchvision"
       "r"
       "r-argparse"
       "r-tidyverse"
       "r-mrgsolve"
       "r-mapbayr"
       "r-mess"
       "r-lixoftconnectors"
       "r-glue"
       "r-furrr"))