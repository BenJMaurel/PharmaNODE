import json

path = "docs/new_drug_tutorial.ipynb"
with open(path, "r") as f:
    nb = json.load(f)

# Find the index of the cell containing pk_factory definition to insert the pop_param tweaking cells
insert_idx_pop = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and any('def pk_factory' in line for line in cell.get('source', [])):
        insert_idx_pop = i + 1
        break

pop_md_cell = {
 "cell_type": "markdown",
 "metadata": {},
 "source": [
  "### Tweaking Population Parameters\n",
  "\n",
  "You can define custom typical values (`pop_params`) and inter-patient variabilities (`ipv`) for your PK model. The `ClassicPKModel` accepts these as dictionaries.\n",
  "If not provided, it falls back to defaults for the selected number of compartments and absorption type."
 ]
}

pop_code_cell = {
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "custom_pop_params = {\"ka\": 1.5, \"CL\": 12.0, \"Vc\": 50.0, \"Q\": 10.0, \"Vp\": 40.0}\n",
  "custom_ipv = {\"ka\": 0.2, \"CL\": 0.2, \"Vc\": 0.2, \"Q\": 0.2, \"Vp\": 0.2}\n",
  "\n",
  "model_custom = ClassicPKModel(\n",
  "    n_compartments=2, \n",
  "    absorption='oral', \n",
  "    pop_params=custom_pop_params, \n",
  "    ipv=custom_ipv, \n",
  "    cov_data=meta\n",
  ")\n",
  "\n",
  "model_custom.dose_mg = 100.0\n",
  "conc_custom = model_custom.simulate(dosing, times).squeeze(-1).detach().cpu().numpy()\n",
  "\n",
  "plt.figure(figsize=(8, 4))\n",
  "plt.plot(times.numpy() - config.n_steady_state_cycles * config.dosing_interval_h, conc_custom[:-1], label=\"Custom Params\")\n",
  "plt.xlabel(\"Time since last dose (h)\")\n",
  "plt.ylabel(\"Concentration (mg/L)\")\n",
  "plt.title(\"Example patient with Custom Population Parameters\")\n",
  "plt.legend()\n",
  "plt.tight_layout()\n",
  "plt.show()"
 ]
}

# Insert population parameter cells
if insert_idx_pop != -1:
    nb['cells'].insert(insert_idx_pop, pop_code_cell)
    nb['cells'].insert(insert_idx_pop, pop_md_cell)


# The rest of the sections can be appended to the end of the notebook
real_data_md_cell = {
 "cell_type": "markdown",
 "metadata": {},
 "source": [
  "## 6. Working with Real Data & Covariates\n",
  "\n",
  "When you have real clinical data, you do **not** need to generate a virtual cohort. Instead, format your real dataset similarly to the generated CSVs (e.g., `virtual_cohort_train.csv`) and modify the data loading pipeline to use it.\n",
  "\n",
  "### Formatting your CSV\n",
  "Your dataset should ideally be a long-format CSV containing at minimum:\n",
  "- `ID`: Patient identifier.\n",
  "- `TIME`: Time of the event (observation or dose).\n",
  "- `OUT`: The measured concentration (use `-1` or `NaN` if the row is just a dose event).\n",
  "- `DOSE`: The amount of drug administered.\n",
  "- Any covariates you wish to include (e.g., `WT` for weight, `AGE`, `SEX`).\n",
  "\n",
  "### Defining Observation Times\n",
  "In a real dataset, the time grid for the ODE solver is naturally defined by the unique times in the `TIME` column of your CSV. However, to extract specific points to feed into the Latent ODE **encoder**, you must specify the `sparse_times` argument. \n",
  "For example, if you want the encoder to only look at trough and peak concentrations taken at 0h and 2h post-dose, you would pass `sparse_times=[0.0, 2.0]` to the `extract_gen_tac` function in `lib/read_tacro.py`.\n",
  "\n",
  "### Handling Covariates\n",
  "If you have covariates like `WT` (Weight):\n",
  "1. Add them as columns to your CSV.\n",
  "2. Modify `lib/read_tacro.py` (inside `extract_gen_tac` or your custom extraction function) to read these columns from the pandas dataframe and append them to the `static` features list for each patient.\n",
  "3. In `collate_fn_tacro`, ensure these static features are properly batched into tensors.\n",
  "4. Finally, your PK Model (like `ClassicPKModel`) receives these via the `cov_data` dictionary. You can then use them to scale your parameters (e.g., allometric scaling on Clearance `CL = CL * (WT/70)**0.75`)."
 ]
}

latent_md_cell = {
 "cell_type": "markdown",
 "metadata": {},
 "source": [
  "## 7. Visualizing the Latent Space (`z0`)\n",
  "\n",
  "After training the Latent ODE model, you may want to extract the latent representations (`z0`) for each patient to perform clustering or visualization (e.g., PCA, t-SNE)."
 ]
}

latent_code_cell = {
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "# Example snippet for extracting latent representations\n",
  "# Ensure you have loaded your trained model first\n",
  "\"\"\"\n",
  "from lib.utils import get_device\n",
  "device = get_device(torch.tensor(1))\n",
  "\n",
  "# Assuming `model` is your trained Latent ODE model and `dataloader` provides your batches\n",
  "model.eval()\n",
  "all_z0_means = []\n",
  "\n",
  "with torch.no_grad():\n",
  "    for batch_dict in dataloader:\n",
  "        # The model's encoder processes the observed data to produce the latent distribution\n",
  "        z0_mean, z0_std = model.encoder_z0(\n",
  "            batch_dict[\"observed_data\"], \n",
  "            batch_dict[\"observed_tp\"], \n",
  "            mask=batch_dict.get(\"observed_mask\", None), \n",
  "            run_backwards=True\n",
  "        )\n",
  "        all_z0_means.append(z0_mean.cpu().numpy())\n",
  "\n",
  "all_z0_means = np.concatenate(all_z0_means, axis=0).squeeze()\n",
  "\n",
  "# For visualization, if z0 is multi-dimensional, you can use PCA:\n",
  "from sklearn.decomposition import PCA\n",
  "import matplotlib.pyplot as plt\n",
  "\n",
  "pca = PCA(n_components=2)\n",
  "z0_pca = pca.fit_transform(all_z0_means)\n",
  "\n",
  "plt.figure(figsize=(8, 6))\n",
  "plt.scatter(z0_pca[:, 0], z0_pca[:, 1], alpha=0.7, c='blue')\n",
  "plt.xlabel(\"Principal Component 1\")\n",
  "plt.ylabel(\"Principal Component 2\")\n",
  "plt.title(\"Latent Space (z0) Visualization\")\n",
  "plt.show()\n",
  "\"\"\"\n",
  "print(\"Uncomment and adapt the snippet above to visualize your trained model's latent space.\")"
 ]
}

nb['cells'].append(real_data_md_cell)
nb['cells'].append(latent_md_cell)
nb['cells'].append(latent_code_cell)

with open(path, "w") as f:
    json.dump(nb, f, indent=1)

print("Notebook successfully updated.")
