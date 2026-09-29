#!/usr/bin/env python3
"""Normalise the legacy evaluation tables into one cells table.

The revision's evaluations were produced by ad-hoc wrappers (kept in
results/confound_eval/raw/legacy_wrappers/), unseeded, in two schemas and with
the epoch encoded in several ways.  This reads them from
results/confound_eval/raw/ and writes results/confound_eval/cells.tsv in the
schema run_jobs.sh produces, so legacy and new cells can be analysed together.
A provenance report is written to results/confound_eval/import_report.txt.

Rules
  * A row is rejected unless every field validates: seed trained in the arm,
    model, training cohort in the arm, evaluation cohort consistent with the
    arm and regime, every metric present and finite.
  * Repeat evaluations of one cell within a file are averaged (n_evals > 1).
  * A cell present in several files is taken from the highest-precedence file
    (lowest rank in SOURCES).  Full-metric tables outrank the two early
    four-metric tables.  Superseded values are listed in the report.
"""
import csv
import math
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import registry as reg  # noqa: E402

BASE = os.path.join(reg.REPO, "results", "confound_eval")
RAW = os.path.join(BASE, "raw")
METRICS = ["v1_mpe", "v1_auc", "v1_pw", "v1_nrmse", "v2_mpe", "v2_auc", "v2_pw", "v2_nrmse"]
FOUR = ["v1_auc", "v2_auc", "v2_pw", "v2_nrmse"]
HEADER = ["seed", "epoch", "arm", "regime", "model", "trained", "evaluated"] + METRICS + \
         ["n_evals", "eval_seed", "source"]

# (file, epoch when not encoded as seed@epoch, metric columns, precedence rank)
SOURCES = [
    ("results_ep3000.tsv",     "3000", METRICS, 0),
    ("results_ood.tsv",        "3000", METRICS, 0),
    ("results_s456.tsv",       "3000", METRICS, 0),
    ("results_narrow6k.tsv",   "6000", METRICS, 0),
    ("results_s456_6k.tsv",    "6000", METRICS, 0),
    ("results_s789.tsv",       None,   METRICS, 0),
    ("seed_results.tsv",       "6000", FOUR,    1),
    ("seed_results_sweep.tsv", None,   FOUR,    2),
]


def parse_row(fields, default_epoch, cols):
    if len(fields) != 5 + len(cols):
        raise ValueError(f"{len(fields)} fields, expected {5 + len(cols)}")
    s, armf, model, tr, ev = fields[:5]
    if "@" in s:
        s, epoch = s.split("@", 1)
    elif default_epoch is None:
        raise ValueError("epoch not encoded and the file has no default")
    else:
        epoch = default_epoch
    if epoch != "best" and not epoch.isdigit():
        raise ValueError(f"bad epoch {epoch!r}")
    arm, _, regime = armf.partition("-")
    regime = regime or "in"
    if arm not in reg.ARMS or regime not in reg.REGIMES:
        raise ValueError(f"bad arm/regime field {armf!r}")
    if not s.isdigit() or int(s) not in reg.ARMS[arm][1]:
        raise ValueError(f"seed {s!r} was not trained in the {arm} arm")
    if model not in reg.MODELS:
        raise ValueError(f"bad model {model!r}")
    if tr not in reg.cohorts(arm):
        raise ValueError(f"training cohort {tr!r} is not in the {arm} arm")
    if ev not in [reg.eval_cohort(c, regime) for c in reg.cohorts(arm)]:
        raise ValueError(f"evaluation cohort {ev!r} inconsistent with {arm}/{regime}")
    vals = {}
    for c, x in zip(cols, fields[5:]):
        v = float(x)
        if not math.isfinite(v):
            raise ValueError(f"non-finite {c}")
        vals[c] = v
    return (s, epoch, arm, regime, model, tr, ev), vals


def sort_key(k):
    s, epoch, arm, regime, model, tr, ev = k
    return (list(reg.ARMS).index(arm), int(epoch) if epoch.isdigit() else 10**9, int(s),
            model, tr, reg.REGIMES.index(regime), ev)


def main():
    report, cells, superseded = [], {}, []
    for fname, dep, cols, rank in SOURCES:
        path = os.path.join(RAW, fname)
        if not os.path.exists(path):
            sys.exit(f"missing legacy table: {path}")
        acc, rejected, nread = defaultdict(list), [], 0
        with open(path) as f:
            for i, line in enumerate(f, 1):
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                nread += 1
                try:
                    key, vals = parse_row(line.split("\t"), dep, cols)
                except ValueError as e:
                    rejected.append((i, str(e), line))
                    continue
                acc[key].append(vals)
        dup = [v for v in acc.values() if len(v) > 1]
        msg = (f"{fname}: {nread} rows read, {nread - len(rejected)} accepted, {len(rejected)} rejected, "
               f"{len(acc)} distinct cells")
        if dup:
            spread = [max(x["v2_auc"] for x in v) - min(x["v2_auc"] for x in v) for v in dup]
            msg += (f"; {len(dup)} cells evaluated more than once and averaged "
                    f"(V2 AUC spread median {statistics.median(spread):.3f}, max {max(spread):.3f})")
        report.append(msg)
        report += [f"    rejected line {i}: {why}  |  {raw[:100]}" for i, why, raw in rejected]
        for key, vs in acc.items():
            mean = {c: sum(v[c] for v in vs) / len(vs) for c in cols}
            if key in cells:
                prev = cells[key]
                if prev[0] == rank:
                    sys.exit(f"cell {key} is in two sources of equal precedence: {prev[1]}, {fname}")
                superseded.append((key, prev[1], fname, prev[3]["v2_auc"], mean["v2_auc"]))
                continue
            cells[key] = (rank, fname, len(vs), mean)

    with open(os.path.join(BASE, "cells.tsv"), "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(HEADER)
        for key in sorted(cells, key=sort_key):
            _, src, n, mean = cells[key]
            w.writerow(list(key) + [f"{mean[m]:.6g}" if m in mean else "" for m in METRICS]
                       + [n, "unseeded", f"legacy:{src}"])

    report.append(f"\n{len(cells)} cells written to results/confound_eval/cells.tsv")
    if superseded:
        d = [abs(a - b) for *_, a, b in superseded]
        report.append(f"{len(superseded)} cell(s) also present in a lower-precedence file were superseded "
                      f"(|V2 AUC difference| median {statistics.median(d):.3f}, max {max(d):.3f}):")
        report += [f"    {'/'.join(k)}  kept {kept} ({a:.2f})  dropped {drop} ({b:.2f})"
                   for k, kept, drop, a, b in sorted(superseded, key=lambda r: sort_key(r[0]))]
    report.append("\nCoverage -- seeds with all 8 cells (2 models x 2 training x 2 test cohorts):")
    for arm in reg.ARMS:
        eps = sorted({k[1] for k in cells if k[2] == arm}, key=lambda e: int(e) if e.isdigit() else 10**9)
        for ep in eps:
            for regime in reg.REGIMES:
                full = [s for s in reg.ARMS[arm][1]
                        if all((str(s), ep, arm, regime, m, tr, reg.eval_cohort(ev, regime)) in cells
                               for m in reg.MODELS for tr in reg.cohorts(arm) for ev in reg.cohorts(arm))]
                if full:
                    report.append(f"    {arm:<7} epoch {ep:>5}  {reg.REGIME_NAME[regime]:<12} seeds {','.join(map(str, full))}")
    text = "\n".join(report)
    with open(os.path.join(BASE, "import_report.txt"), "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
