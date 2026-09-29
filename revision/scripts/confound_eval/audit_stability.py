#!/usr/bin/env python3
"""Outcome-blind training-stability audit of every confounding-experiment run.

Reads only the training logs.  No evaluation metric is consulted, so an
exclusion decided from this audit cannot be a selection on the result.  For
each arm, model and checkpoint epoch it reports, per cell, the test-set loss
and KL logged nearest that epoch, the run's own median loss over the preceding
1000 epochs, and a robust z-score of the KL against every cell of the same
arm, model and epoch:

    z = (KL - median) / (1.4826 * MAD)          flagged when |z| > threshold

With few, tightly packed cells the MAD is small and z inflates; the KL range
is printed with every block so a flag can be judged against the pack's spread.

    audit_stability.py [--arms narrow] [--models film dc] [--epochs 3000 6000] [--threshold 3.5]
"""
import argparse
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import registry as reg  # noqa: E402

PATTERN = {
    "film": re.compile(r"^Epoch (\d+) \[FiLM Test\] \| Total Loss ([-\d.]+).*?KL ([\d.]+)", re.M),
    "dc":   re.compile(r"^Epoch (\d+) \[DoseCond Test\] \| Loss ([-\d.]+).*?KL ([\d.]+)", re.M),
}
LABEL = {"film": "ours (OT-FiLM)", "dc": "dose-cond"}


def trace(arm, seed, model, trained):
    path = reg.train_log(arm, seed, model, trained)
    if not os.path.exists(path):
        return None, path
    with open(path) as f:
        return {int(e): (float(l), float(k)) for e, l, k in PATTERN[model].findall(f.read())}, path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=list(reg.ARMS), choices=list(reg.ARMS))
    ap.add_argument("--models", nargs="+", default=["film", "dc"], choices=list(reg.MODELS))
    ap.add_argument("--epochs", nargs="+", type=int, default=[3000, 6000])
    ap.add_argument("--threshold", type=float, default=3.5)
    a = ap.parse_args()
    flagged = []
    for arm in a.arms:
        conf, _ = reg.cohorts(arm)
        for model in a.models:
            for ep in a.epochs:
                rows = []
                for s in reg.ARMS[arm][1]:
                    for tr in reg.cohorts(arm):
                        d, path = trace(arm, s, model, tr)
                        if not d:
                            print(f"  missing or unparsable log: {path}")
                            continue
                        near = min(d, key=lambda e: abs(e - ep))
                        if abs(near - ep) > 60:
                            continue
                        win = [v[0] for e, v in d.items() if ep - 1000 <= e < ep]
                        rows.append((s, "conf" if tr == conf else "ctrl", d[near][0],
                                     float(np.median(win)) if len(win) >= 5 else math.nan, d[near][1]))
                if not rows:
                    continue
                kl = np.array([r[4] for r in rows])
                med, mad = float(np.median(kl)), float(np.median(np.abs(kl - np.median(kl))))
                print(f"\n== {arm} / {LABEL[model]} / epoch {ep} — {len(rows)} cells, KL median {med:.3f}, "
                      f"MAD {mad:.3f}, range [{kl.min():.3f}, {kl.max():.3f}] ==")
                print(f"   {'seed':>4}{'cohort':>8}{'loss':>11}{'own median':>12}{'KL':>9}{'robust z':>10}")
                scored = []
                for s, c, loss, own, k in rows:
                    z = (k - med) / (1.4826 * mad) if mad > 0 else (0.0 if k == med else math.inf)
                    scored.append((z, s, c, loss, own, k))
                for z, s, c, loss, own, k in sorted(scored, key=lambda r: -abs(r[0])):
                    flag = abs(z) > a.threshold
                    print(f"   {s:>4}{c:>8}{loss:>11.3f}{own:>12.3f}{k:>9.3f}{z:>10.2f}{'   <-- FLAG' if flag else ''}")
                    if flag:
                        flagged.append(f"{arm} / {LABEL[model]} / seed {s} {c} / epoch {ep}: "
                                       f"KL {k:.3f} (median {med:.3f}, range of the others "
                                       f"[{min(x for x in kl if x != k):.3f}, {max(x for x in kl if x != k):.3f}]), z {z:+.2f}")
    print("\nFLAGGED" + (":" if flagged else ": none"))
    for f in flagged:
        print(f"  {f}")


if __name__ == "__main__":
    main()
