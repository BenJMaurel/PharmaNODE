# `revision/` — code for the follow-up work (ISCB–GMDS 2026)

This folder holds the scripts behind the work presented at ISCB–GMDS 2026 (Freiburg): the revision experiments on
**where the dose enters a latent neural-ODE PK model**, the extended simulation benchmark, and the re-run of the
three misspecification scenarios against MAP-BE. It is a snapshot of the working tree the experiments were run from,
kept in its original flat layout so every script runs unchanged.

* `../code/` is the published pipeline (paper + PAGE 2026 poster). It is **not** modified by this folder.
* The repository exactly as presented at PAGE 2026 is tagged `page-2026`.
* No data, checkpoints or results are included: every cohort is simulated by the scripts below.

## Setup

* Python 3.12, CPU only. `pip install -r ../code/requirements.txt` (torch, torchdiffeq, numpy, pandas, scipy, …).
  Tested with torch 2.6.0, numpy 2.1.3, pandas 2.2.3, scipy 1.15.2.
* For the NLME / MAP-BE references: R with `lixoftConnectors` (Monolix 2024R1), `mapbayr` 0.10.2, `mrgsolve`,
  `tidyverse`, `glue`, `argparse`, `furrr`, `MESS`.
* Run everything **from this folder**. Use one thread per job (`export OMP_NUM_THREADS=1`); the experiments are many
  small single-threaded runs.
* The shell scripts contain the absolute path of the machine they ran on. Point them at your copy once:
  ```bash
  grep -rl /Users/benjaminmaurel/Documents/PharmaNODE . | xargs sed -i.orig "s#/Users/benjaminmaurel/Documents/PharmaNODE#$PWD#g"
  ```

## Map of the folder

| what | files |
|---|---|
| Simulators | `gen_tacro_film.py` (tacrolimus PK, scenarios 1–4), `gen_tacro_confound.py` (paired two-visit cohorts: dose confounding on Km or Vc, observation noise, exposure window, out-of-range doses) |
| Models | `run_models.py` (latent ODE; OT-FiLM with `--use_film`), `train_dose_cond.py` (dose-conditioned vector field), `train_lu_pk.py` (port of Lu et al., *Nat. Mach. Intell.* 3:696, 2021), `lib/` |
| Evaluation | `test_film_matched.py`, `test_dose_cond.py`, `test_lu_pk.py`, `scripts/confound_eval/seeded_eval.py` |
| Scenario-4 benchmark (factual + counterfactual accuracy, ablations) | `scripts/s4/` (`chain_table1_window.sh` = main table), `scripts/ndm/ndm_tables.py` |
| Confounding experiments (Km and Vc, matched difference-in-differences) | `scripts/confound_eval/` (own README) |
| Three misspecification scenarios vs MAP-BE, 100 runs each | `scripts/scen_rerun/` |
| NLME / MAP-BE references on the simulated cohorts | `scripts/monolix/` |
| Patient-count sweep | `scripts/nsweep/` |
| Identifiability, information floors, true-model EBE | `scripts/ident/` |
| Theory diagnostics, baselines, figures | `scripts/theory/`, `scripts/baselines/`, `scripts/figures/`, `scripts/vc/`, `scripts/sat/`, `scripts/idv/`, `scripts/lu_diag/`, `scripts/repro/` |

## Three designs for the dose

* **OT-FiLM** (ours): the vector field is autonomous; a dose change acts only as an affine, patient-independent
  displacement of the initial latent state (`--use_film --film-no-z0-cond`).
* **Dose-conditioned field**: the dose is an input of the vector field, f(z, d) (`train_dose_cond.py`).
* **Lu et al. (2021)**: the dose is an impulse on a designated coordinate of the latent state (`train_lu_pk.py`).

## Example: the scenario-4 benchmark

```bash
# 1. cohort: scenario 4 (saturable elimination), paper-level noise, visit-1 exposure restricted to 3-25 ng/mL
python gen_tacro_confound.py --exp confound_vc00_s4_pnoise_win --num_patients 2500 --rho 0.0 --test-fraction 0.904 \
  --seed 0 --dose-seed 1234 --confound-param Vc --scenario 4 --dose-grid 1,2,3,4,5,6,7,8 \
  --prop-sd 0.113 --add-sd 0.71 --v1-cavg-window 3,25
python scripts/nsweep/make_subsets.py confound_vc00_s4_pnoise_win "100"          # 100 training patients

# 2. OT-FiLM (main configuration)
python run_models.py --niters 6000 -n 200 -s 40 -l 10 --dataset PK_Tacro --latent-ode --use_film --noise-weight 0.01 \
  --max-t 5. -b 512 --seed 1 --experiment confound_vc00_s4_pnoise_win_n100 --film-no-z0-cond --obsrv-std 0.05 \
  --film-self-consistency 141.27 --decoder-hidden 50 --decoder-residual --select-on mse_v2 --patience 1000000 \
  --static-dim 4 --save ./out/film_s1/ --eval-every 300 --ckpt-every 300

# 3. dose-conditioned field, matched likelihood and decoder
python train_dose_cond.py --experiment confound_vc00_s4_pnoise_win_n100 --data-dir ./results/exp_film_run --niters 6000 \
  -b 512 -l 10 --lr 1e-2 --seed 1 --patience 1000000 --obsrv-std 0.05 --static-dim 4 --decoder-hidden 50 \
  --decoder-residual --save ./out/dc_s1/ --eval-every 300 --ckpt-every 300

# 4. evaluation of a fixed-epoch checkpoint, normalisation pinned to the training cohort
python scripts/confound_eval/seeded_eval.py --eval-seed 0 test_film_matched.py --experiment confound_vc00_s4_pnoise_win \
  --data-dir ./results/exp_film_run --ckpt <out/film_s1/.../traj/*_ep006000.ckpt> --scale-from confound_vc00_s4_pnoise_win_n100 \
  --eval-split test --out-json film_s1.json
```

`scripts/s4/chain_table1_window.sh` runs the whole table (4 designs × 3 seeds, MAP-BE references, scoring).

## Pitfalls that silently produce wrong numbers

1. **Always pass `--scale-from <training cohort>`** to the test scripts; otherwise the normalisation is refitted on
   the evaluated cohort.
2. **Do not use `Test MSE` in training logs or the `_best` checkpoints for model selection.** The training loop refits
   the test-split scaler, so that signal is biased; read the fixed-epoch checkpoints in `traj/` instead.
3. **Do not report raw AUC RMSPE on large simulated test sets**: a handful of near-zero true AUCs dominate it. Report
   the median |relative error|, RMSPE trimmed of the worst 1 %, and the count of errors > 100 %.
4. **The default residual error of `gen_tacro_film.py` in this folder is 0.03 + 0.03·C**, much lower than the paper's
   0.71 + 0.113·C. Pass `--prop-sd 0.113 --add-sd 0.71` to `gen_tacro_confound.py` for paper-level noise.
5. **Match the observation noise (`--obsrv-std`) and the decoder (`--decoder-hidden`, `--decoder-residual`)** before
   comparing designs; the scripts' defaults differ between `run_models.py` and `train_dose_cond.py`.
6. Under Michaelis–Menten elimination (scenarios ≥ 3), `CL` is sampled and reported but never enters the ODE.
7. **Keep visit-1 exposure plausible** (`--v1-cavg-window 3,25`): with doses drawn at random, the lowest-exposure
   simulated patients dominate every relative-error metric. For out-of-range cohorts also pass `--ood-keep-v1`.

## MAP-BE rerun (`scripts/scen_rerun/`)

The 100-run comparison of the three misspecification scenarios regenerates each seeded dataset with the published
code (commit `2eaadf6`, checked out as worktrees `results/paper_repro/w01..w10`), then fits the population model in
Monolix and predicts with `mapbayr`, with three initialisations (published typical values, Monolix defaults, Monolix
auto-init; `INIT_MODE=published|default|auto`) and the residual-error variants (`SIGMA_MODE=var|sd|c1`: variances,
standard deviations as in the original script, and the combined1 form Monolix estimates). `summarise.py` prints
per-run RMSPE / MPE with paired tests.
