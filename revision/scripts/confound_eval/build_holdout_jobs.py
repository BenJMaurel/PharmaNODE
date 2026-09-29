#!/usr/bin/env python3
"""PASS A of the leakage-free read-out: score every checkpoint in the 5000-6000
window on the SELECTION half (selA) of the run's OWN training cohort.

A model trained on the confounded cohort is selected on confounded-cohort
validation data, as it would be in practice.  Nothing here touches selB -- the
reporting half is only read in pass B, after the checkpoint is fixed.

Normalisation stays pinned to the model's original training cohort (--scale-from),
as everywhere else.
"""
import os, sys, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import registry as reg
WIN = tuple(range(5000, 6001, 100))
w = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n"); n = 0
for m in ("filmd", "wr0.548", "wr2.0"):
    for s in (1, 2, 3):
        for tr in reg.cohorts("wide"):
            for ep in WIN:
                try: ck = reg.checkpoint("wide", s, m, tr, ep)
                except FileNotFoundError: continue
                w.writerow(["wide", s, ep, "in", m, tr, f"{tr}_selA", ck]); n += 1
print(f"{n} selection evaluations", file=sys.stderr)
