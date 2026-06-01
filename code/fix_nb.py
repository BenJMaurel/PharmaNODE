import json

with open("docs/02_classic_pk_film.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = cell["source"]
        new_source = []
        for line in source:
            if "from lib.pk_drug import DrugStudyConfig, generate_virtual_cohort_film, save_cohort_splits" in line:
                new_source.append("from lib.pk_drug import DrugStudyConfig, generate_virtual_cohort_film, save_cohort_splits\n")
                new_source.append("from lib.classic_pk import ClassicPKModel\n")
            elif "from lib.classic_pk import ClassicPKModel" in line:
                pass
            else:
                new_source.append(line)
        cell["source"] = new_source

with open("docs/02_classic_pk_film.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
