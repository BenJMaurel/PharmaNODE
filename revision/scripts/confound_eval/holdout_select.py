#!/usr/bin/env python3
"""Selection pass for the big-held-out-set read-out.

Every candidate checkpoint (epochs 3000-6000) scored on the IN-SUPPORT selection
half of the run's OWN training cohort.  Selection never sees selB, and never sees
the lo/hi regimes -- a checkpoint is chosen once, on in-support validation data,
exactly as a practitioner would, and then reported everywhere.

    holdout_select.py <arm>
"""
import os, sys, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import registry as reg

WIN = (3000, 4000, 5000, 6000)
ARM = sys.argv[1]
MODELS = {"wide":   [("dc", (1,2,3,4,5,6)), ("film", (1,2,3,4,5,6)),
                     ("wr0.548", (1,2,3)), ("wr2.0", (1,2,3))],
          "narrow": [("dc", tuple(range(1,10))), ("film", tuple(range(1,10)))]}[ARM]
conf, ctrl = reg.cohorts(ARM)
w = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n"); n = 0
for m, seeds in MODELS:
    for s in seeds:
        for tr in (conf, ctrl):
            for ep in WIN:
                try: ck = reg.checkpoint(ARM, s, m, tr, ep)
                except FileNotFoundError: continue
                w.writerow([ARM, s, ep, "in", m, tr, f"{tr}_selA", ck]); n += 1
print(f"{ARM}: {n} selection evaluations", file=sys.stderr)
