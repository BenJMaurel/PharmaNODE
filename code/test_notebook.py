import json
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

with open("docs/02_classic_pk_film.ipynb", "r") as f:
    nb = nbformat.read(f, as_version=4)

# We just want to execute cells 1, 3, 5 (indices 1, 3, 5 are the code cells)
# We can skip the `!python run_models.py` to save time.
nb.cells = nb.cells[:-2]

ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
ep.preprocess(nb, {'metadata': {'path': '.'}})
print("Notebook 02 executed successfully up to training!")
