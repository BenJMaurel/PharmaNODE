#!/usr/bin/env python3
"""Verify the lo/hi cohorts contain the SAME test patients as confound_vc00_s4, and
that only visit 2 moved out of the training grid.  Exits non-zero on any mismatch."""
import sys, pandas as pd, numpy as np
R = "results/exp_film_run"
base = pd.read_csv(f"{R}/confound_vc00_s4/confound_truth.csv").set_index("ID")
bt = base[base.split == "test"]
ok = True
for reg, want in (("lo", {0.25, 0.5}), ("hi", {10.0, 12.0})):
    t = pd.read_csv(f"{R}/confound_vc00_s4_{reg}/confound_truth.csv").set_index("ID")
    tt = t[t.split == "test"]
    same_ids = set(tt.index) == set(bt.index)
    phys = all(np.allclose(tt.loc[bt.index, c].values, bt[c].values) for c in ("confound_par", "CL_base", "HT"))
    d2 = set(tt.d2.unique()); d1_in = set(tt.d1.unique()) <= {1, 2, 3, 4, 5, 6, 7, 8}
    print(f"  {reg}: n_test={len(tt)} same_ids={same_ids} same_physiology={phys} "
          f"d2={sorted(d2)} d1_in_grid={d1_in}")
    ok &= same_ids and phys and d2 == want and d1_in
print("PAIRING_OK" if ok else "PAIRING_FAILED"); sys.exit(0 if ok else 1)
