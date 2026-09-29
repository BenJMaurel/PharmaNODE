#!/usr/bin/env python3
"""Reporting pass: score each run's SELECTED checkpoint on the disjoint reporting
half, in all three dose regimes and on both cohorts (so the DiD can be formed).

Selection criterion is v2_nrmse on in-support selA -- nRMSE rather than AUC RMSPE,
because the latter is a per-patient relative error that a handful of large
downward extrapolations can dominate.

    holdout_report.py <arm> <cells_file>
"""
import os, sys, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import registry as reg

ARM, CELLS = sys.argv[1], sys.argv[2]
conf, ctrl = reg.cohorts(ARM)
sel = {}
with open(CELLS) as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if r["arm"] != ARM or not r["evaluated"].endswith("_selA"): continue
        if not r["epoch"].isdigit() or r["v2_nrmse"] in ("", "nan"): continue
        sel.setdefault((r["model"], int(r["seed"]), r["trained"]), []).append(
            (float(r["v2_nrmse"]), int(r["epoch"]), r["source"]))
w = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n"); n = 0
for k in sorted(sel):
    best_v, best_ep, ck = min(sel[k])
    m, s, tr = k
    spread = max(x[0] for x in sel[k]) - best_v
    print(f"  {m:<9} s{s} {tr:<15} -> ep{best_ep}  (selA v2_nRMSE {best_v:.2f}, "
          f"window spread {spread:.2f})", file=sys.stderr)
    for base in (conf, ctrl):
        for regime, suffix in (("in", ""), ("lo", "lo"), ("hi", "hi")):
            ev = f"{base}{suffix}_selB"
            if not os.path.isdir(os.path.join(reg.REPO, "results", "exp_film_run", ev)):
                continue
            w.writerow([ARM, s, best_ep, regime, m, tr, ev, ck]); n += 1
print(f"{ARM}: {n} reporting evaluations from {len(sel)} cells", file=sys.stderr)
