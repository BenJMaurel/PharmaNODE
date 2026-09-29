# Confounding experiment — evaluation and analysis

Everything needed to reproduce the numbers of the confounding experiment in the
revision (Table 4 of `main_revised.tex`, the narrow-arm paired contrasts, the
out-of-range results and the stability audit) from the trained checkpoints.

```bash
python3 scripts/confound_eval/analyse_confound.py        # every table, both arms, epochs 3000 and 6000
python3 scripts/confound_eval/analyse_confound.py --table table4 --epochs 3000 --latex
python3 scripts/confound_eval/analyse_confound.py --table paired --arms narrow --exclude 3
python3 scripts/confound_eval/audit_stability.py         # outcome-blind check of every training run
python3 scripts/confound_eval/check_reported.py          # 128 reported values; exits 1 on any mismatch
```

A snapshot of the full output is in `results/confound_eval/tables.txt`.

## Pipeline

```
gen_tacro_confound.py            cohorts   results/exp_film_run/confound_{km,kmn}{09,00}[_lo|_hi]
launch_seed.sh, launch_narrow_extra.sh
                                 checkpoints results/seeds/<seed dir>/exp_*_run/<cohort>/traj/*_epNNNNNN.ckpt
build_jobs.py  ->  run_jobs.sh  ->  eval_cell.sh  ->  seeded_eval.py  ->  test_film_matched.py | test_dose_cond.py
                                 cells table  results/confound_eval/cells.tsv   (one row per evaluated cell)
analyse_confound.py   <- cells table            audit_stability.py   <- logs/
```

| file | role |
|---|---|
| `registry.py` | where every seed's checkpoints and logs live; the only place the naming quirks are encoded |
| `seeded_eval.py` | runs a harness unchanged with random / numpy / torch seeded — the harnesses do not seed themselves |
| `eval_cell.sh` | evaluates one checkpoint on one cohort with `--scale-from <training cohort>`; fails loudly if a metric does not parse |
| `build_jobs.py` | lists every cell to evaluate; `--skip-done` drops cells already in a table |
| `run_jobs.sh` | runs a job list in parallel; one file per job, merged at the end, so rows cannot interleave |
| `import_legacy.py` | normalises the tables produced during the revision into `cells.tsv` (see below) |
| `analyse_confound.py` | every table; definitions of DiD, damage, OOD ratio and the paired tests are in its docstring |
| `audit_stability.py` | robust z-score of the logged KL per arm / model / epoch, from training logs only |
| `check_reported.py` | regression check against the reported values |

## Evaluating new seeds

Train, add the seeds to `ARMS` in `registry.py`, then:

```bash
python3 scripts/confound_eval/build_jobs.py --arms narrow --seeds 10 11 12 \
        --skip-done results/confound_eval/cells.tsv > /tmp/jobs.tsv
scripts/confound_eval/run_jobs.sh /tmp/jobs.tsv results/confound_eval/cells.tsv 8
```

New cells are evaluated with `eval_seed = 0` and are exactly reproducible.

## What `cells.tsv` holds today

The 678 cells currently in `cells.tsv` are the evaluations made during the revision,
imported by `import_legacy.py` from `results/confound_eval/raw/`. They are marked
`eval_seed = unseeded` and `source = legacy:<file>`. The wrappers and job lists
that produced them are kept in `raw/legacy_wrappers/`. `import_report.txt` records
the import in full. In short:

- **Unseeded.** Re-evaluating a legacy cell gives a slightly different number:
  137 accidental repeat evaluations spread by median 0.05, max 0.55 points, and
  the seeded pipeline re-evaluating seed 7 moved no cell by more than 0.07.
  No conclusion depends on differences of that size.
- **One row rejected** — an interleaved line in `seed_results_sweep.tsv` left by
  concurrent `>>` appends.
- **Repeat evaluations averaged** (137 cells, all in the epoch-sweep table).
- **48 cells superseded.** Where the same cell was evaluated in two passes, the
  full-metric tables win over the two early four-metric ones. The visible
  consequence: the wide epoch sweep's ep3000 row now uses the same cells as
  Table 4 (DiD +8.23 / +2.56); the sweep figures quoted during the revision,
  +8.21 / +2.68, came from a separate evaluation pass of the same checkpoints.
- **Gaps.** Wide-arm out-of-range cells exist only at epoch 3000, and the wide
  epoch-6000 cells carry four metrics (no bias). Tables name the seeds used and
  any dropped for missing data.

To replace the legacy cells with a fully seeded evaluation of every checkpoint
(576 cells, about 25 minutes at 8 in parallel):

```bash
python3 scripts/confound_eval/build_jobs.py > /tmp/jobs_all.tsv
scripts/confound_eval/run_jobs.sh /tmp/jobs_all.tsv results/confound_eval/cells_seeded.tsv 8
python3 scripts/confound_eval/analyse_confound.py --cells results/confound_eval/cells_seeded.tsv
```

`check_reported.py` would then report small mismatches, of evaluation-noise size,
because the reported values came from the unseeded pass.

## Seed registry

| arm | seeds | notes |
|---|---|---|
| wide (1–8 mg) | 1–3 | seed 2's logs use suffix `__s2w` |
| narrow (2–5 mg) | 1–9 | seed 2 is saved in `results/seeds/s2` with logs `__s2`; seeds 4–6 were launched toward 15000 epochs and stopped after 7000; the others stop at 6000. Every seed has checkpoints at 1000–6000. |

Narrow seed 3, confounded, OT-FiLM converged to a distinct, worse optimum (KL 1.373
at epoch 6000 against 0.909–0.969 for the other 17 cells; z = +26). It is kept in
every table. `--exclude 3` gives the sensitivity analysis. Re-running seed 3
reproduces the same basin exactly, because training is seeded and runs on CPU.
The dose-cond flag at wide seed 2, epoch 6000 (z = −4.37) is an artefact of a tightly
packed six-cell KL range (6.891–7.272) and is not treated as a failure.
