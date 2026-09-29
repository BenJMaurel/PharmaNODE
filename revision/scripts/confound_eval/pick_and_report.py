#!/usr/bin/env python3
"""PASS B: read pass A, pick each run's best checkpoint on the SELECTION half,
then emit the jobs that score exactly that checkpoint on the REPORTING half of
both cohorts.

Selection criterion is v2_auc on selA -- the counterfactual task, which is what
the paper reports.  The reporting half has never been looked at when this choice
is made, which is the whole point of the split.
"""
import os, sys, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import registry as reg

CELLS = "results/confound_eval/cells_holdout.tsv"
sel = {}
with open(CELLS) as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if not r["epoch"].isdigit() or not r["evaluated"].endswith("_selA"): continue
        if r["v2_auc"] in ("", "nan"): continue
        k = (r["model"], int(r["seed"]), r["trained"])
        sel.setdefault(k, []).append((float(r["v2_auc"]), int(r["epoch"]), r["source"]))
w = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n"); n = 0
for k in sorted(sel):
    best_v, best_ep, ck = min(sel[k])
    m, s, tr = k
    print(f"  {m:<9} s{s} {tr:<14} -> ep{best_ep}  (selA v2_auc {best_v:.2f}, "
          f"window {min(x[0] for x in sel[k]):.2f}-{max(x[0] for x in sel[k]):.2f})",
          file=sys.stderr)
    for base in reg.cohorts("wide"):
        w.writerow(["wide", s, best_ep, "in", m, tr, f"{base}_selB", ck]); n += 1
print(f"{n} reporting evaluations from {len(sel)} cells", file=sys.stderr)
