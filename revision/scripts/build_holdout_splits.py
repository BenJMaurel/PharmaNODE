#!/usr/bin/env python3
"""Split every fresh 1000-patient cohort into a 500-patient selection half (A) and
a 500-patient reporting half (B).

All cohorts of an arm share one physiology seed and ID offset, so sorting by ID
puts the SAME patients in A (and in B) across the confounded/control arms and
across the in/lo/hi regimes.  That is what lets a checkpoint chosen on in-support
A be reported on lo/hi B.

The harness derives its evaluation set as "test.csv minus train.csv", so each half
is materialised with the other half standing in as train.csv.  Nothing is trained
on either file.
"""
import os, sys, glob
import pandas as pd

ROOT = "results/exp_film_run"
# prefix is an argument so the Vc cohorts can be split without touching the Km
# splits that the existing tables were built on.
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "confound_km"
bases = sorted(d for d in os.listdir(ROOT)
               if d.startswith(PREFIX) and ("_big" in d) and os.path.isdir(os.path.join(ROOT, d)))
if not bases: sys.exit(f"no {PREFIX}*_big cohorts found")
built = 0
for base in bases:
    src = os.path.join(ROOT, base, "virtual_cohort_film_test.csv")
    if not os.path.exists(src):
        print(f"  skip {base}: no test csv yet"); continue
    df = pd.read_csv(src)
    ids = sorted(df["ID"].unique())
    if len(ids) != 1000:
        print(f"  WARNING {base}: {len(ids)} patients"); continue
    A, B = set(ids[:500]), set(ids[500:])
    dfA, dfB = df[df.ID.isin(A)], df[df.ID.isin(B)]
    stem = base.replace("_big", "")          # confound_km09_biglo -> confound_km09lo
    for name, test_df, train_df in ((f"{stem}_selA", dfA, dfB), (f"{stem}_selB", dfB, dfA)):
        d = os.path.join(ROOT, name); os.makedirs(d, exist_ok=True)
        train_df.to_csv(os.path.join(d, "virtual_cohort_film_train.csv"), index=False)
        test_df.to_csv(os.path.join(d, "virtual_cohort_film_test.csv"), index=False)
    tr = os.path.join(ROOT, base, "confound_truth.csv")
    if os.path.exists(tr):
        t = pd.read_csv(tr)
        for name, s in ((f"{stem}_selA", A), (f"{stem}_selB", B)):
            t[t.ID.isin(s)].to_csv(os.path.join(ROOT, name, "confound_truth.csv"), index=False)
    built += 1
    print(f"  {base:<26} -> {stem}_selA / {stem}_selB   (500 / 500)")
print(f"{built} cohorts split")
