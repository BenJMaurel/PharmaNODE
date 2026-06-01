import json
import ast

with open("docs/02_classic_pk_film.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"][:-2]:
    if cell["cell_type"] == "code":
        code = "".join(cell["source"])
        if code.startswith("!"): continue
        exec(code)
print("Notebook 02 code blocks executed successfully!")
