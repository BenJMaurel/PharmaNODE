#!/usr/bin/env python3
"""Nested training subsets of a cohort for the patient-count sweep.

Train patients are shuffled once (fixed seed) and the first N kept, so 100 c 200 c 400 c 800:
each step adds patients rather than replacing them, which keeps the curve's steps comparable.
The test split is copied unchanged -- every size is scored on the same held-out patients.
Nothing is written over an existing cohort.

    make_subsets.py <base cohort> <sizes> [seed]      e.g. confound_vc00_s4 "100 200 400"
"""
import os, sys, numpy as np, pandas as pd
base, sizes = sys.argv[1], [int(x) for x in sys.argv[2].split()]
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
R = "results/exp_film_run"
tr = pd.read_csv(f"{R}/{base}/virtual_cohort_film_train.csv")
te = pd.read_csv(f"{R}/{base}/virtual_cohort_film_test.csv")
truth = pd.read_csv(f"{R}/{base}/confound_truth.csv")
ids = np.array(sorted(tr.ID.unique())); np.random.RandomState(seed).shuffle(ids)
print(f"{base}: {len(ids)} train / {te.ID.nunique()} test patients")
for n in sizes:
    assert n <= len(ids), n
    out = f"{R}/{base}_n{n}"
    if os.path.exists(out): print(f"  {out} exists -- left untouched"); continue
    keep = set(ids[:n]); os.makedirs(out)
    tr[tr.ID.isin(keep)].to_csv(f"{out}/virtual_cohort_film_train.csv", index=False)
    te.to_csv(f"{out}/virtual_cohort_film_test.csv", index=False)
    truth[(truth.split == "test") | truth.ID.isin(keep)].to_csv(f"{out}/confound_truth.csv", index=False)
    sub = tr[tr.ID.isin(keep)]
    print(f"  {out}: {sub.ID.nunique()} train ({dict(sub.drop_duplicates('ID').DRUG.value_counts())}), "
          f"{te.ID.nunique()} test")
