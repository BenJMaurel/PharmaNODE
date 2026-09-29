#!/usr/bin/env python3
"""Regression check: recompute every confounding-experiment number that was
reported during the revision and compare it, as printed, with the reported value.

Covers the paper's Table 4 (wide and narrow arms, epoch 3000, seeds 1-3, as in
main_revised.tex), the narrow-arm paired contrasts at n=6, n=8 and n=9, the
wide-arm DiD at epoch 6000, the wide out-of-range ratios, and the DiD on nRMSE.
Exits 1 on any mismatch -- run it after touching cells.tsv or the analysis.

    check_reported.py [--cells results/confound_eval/cells.tsv]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyse_confound as A  # noqa: E402
import registry as reg        # noqa: E402

# Table 4: (V1 mean, sd, V2 conf-test mean, sd, V2 control-test mean, sd) and the DiD
TABLE4 = {
    "wide": {("dc", "conf"): (9.98, 6.63, 23.19, 6.28, 36.08, 8.21), ("dc", "ctrl"): (9.31, 5.03, 18.79, 3.26, 23.45, 4.70),
             ("film", "conf"): (6.86, 0.23, 22.48, 0.81, 27.48, 1.10), ("film", "ctrl"): (8.73, 2.26, 18.46, 2.08, 20.91, 3.53),
             "did": {"dc": (8.23, 0.67), "film": (2.56, 1.43)}},
    "narrow": {("dc", "conf"): (7.97, 1.25, 13.11, 2.08, 13.13, 2.07), ("dc", "ctrl"): (5.69, 1.75, 9.82, 0.80, 9.93, 1.30),
               ("film", "conf"): (6.89, 0.91, 13.42, 3.72, 13.46, 3.71), ("film", "ctrl"): (7.01, 0.98, 11.15, 1.21, 10.79, 0.88),
               "did": {"dc": (-0.10, 0.37), "film": (0.40, 0.35)}},
}
# narrow paired contrasts: (Δ mean, sd, CI low, CI high, t p, Wilcoxon p, #Δ>0)
PAIRED = {
    ("3000", tuple(range(1, 10))): {"did": (0.17, 1.08, -0.65, 1.00, "0.6422", "0.7344", 6),
                                    "lo": (6.04, 3.11, 3.65, 8.43, "0.0004", "0.0039", 9),
                                    "hi": (-0.23, 1.13, -1.10, 0.64, "0.5517", "0.5703", 4)},
    ("6000", tuple(range(1, 10))): {"did": (-0.59, 1.07, -1.41, 0.23, "0.1338", "0.1289", 2),
                                    "lo": (10.26, 4.06, 7.14, 13.39, "0.0001", "0.0039", 9),
                                    "hi": (0.61, 0.83, -0.03, 1.25, "0.0593", "0.0742", 7)},
    ("6000", (1, 2, 4, 5, 6, 7, 8, 9)): {"did": (-0.84, 0.83, -1.53, -0.15, "0.0239", "0.0234", 1)},
    ("6000", (1, 2, 3, 4, 5, 6)): {"lo": (12.17, 2.77, 9.26, 15.08, "0.0001", "0.0312", 6)},
}
WIDE_DID_6000 = {"dc": (8.82, 2.09), "film": (4.21, 0.35)}
WIDE_RATIO_3000 = {("lo", "dc"): (5.33, 0.06), ("lo", "film"): (8.34, 1.12),
                   ("hi", "dc"): (1.31, 0.10), ("hi", "film"): (1.43, 0.15)}
WIDE_NRMSE_DID_3000 = {"dc": (1.49, 0.32), "film": (-0.14, 0.73)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", default=A.DEFAULT_CELLS)
    C = A.Cells(ap.parse_args().cells)
    bad, n = [], 0

    def chk(label, got, exp, spec=".2f"):
        nonlocal n
        n += 1
        g, e = format(got, spec), (exp if isinstance(exp, str) else format(exp, spec))
        if g != e:
            bad.append(f"{label}: got {g}, reported {e}")

    for arm, exp in TABLE4.items():
        conf, ctrl = reg.cohorts(arm)
        S = [1, 2, 3]
        for m in reg.MODELS:
            for tr, tl in ((conf, "conf"), (ctrl, "ctrl")):
                v1 = [np.mean([C.get(s, "3000", arm, "in", m, tr, ev, "v1_auc") for ev in (conf, ctrl)]) for s in S]
                v2c = [C.get(s, "3000", arm, "in", m, tr, conf) for s in S]
                v2k = [C.get(s, "3000", arm, "in", m, tr, ctrl) for s in S]
                got = (*A.msd(v1), *A.msd(v2c), *A.msd(v2k))
                for i, (g, e) in enumerate(zip(got, exp[(m, tl)])):
                    chk(f"Table 4 {arm} {m} {tl} col{i}", g, e)
            d = A.msd([A.did(C, s, "3000", arm, m) for s in S])
            chk(f"Table 4 {arm} DiD {m} mean", d[0], exp["did"][m][0])
            chk(f"Table 4 {arm} DiD {m} sd", d[1], exp["did"][m][1])

    for (ep, S), blocks in PAIRED.items():
        for b, (dm, dsd, lo, hi, pt, pw, pos) in blocks.items():
            f = (lambda s, m: A.did(C, s, ep, "narrow", m)) if b == "did" else \
                (lambda s, m, b=b: A.ratio(C, s, ep, "narrow", m, b))
            p = A.paired([f(s, "film") for s in S], [f(s, "dc") for s in S])
            tag = f"narrow ep{ep} n={len(S)} {b}"
            chk(tag + " Δ mean", p["d"].mean(), dm)
            chk(tag + " Δ sd", p["d"].std(ddof=1), dsd)
            chk(tag + " CI low", p["ci"][0], lo)
            chk(tag + " CI high", p["ci"][1], hi)
            chk(tag + " t p", p["p_t"], pt, ".4f")
            chk(tag + " Wilcoxon p", p["p_w"], pw, ".4f")
            chk(tag + " #Δ>0", p["pos"], str(pos), "d")

    for m, (em, es) in WIDE_DID_6000.items():
        g = A.msd([A.did(C, s, "6000", "wide", m) for s in (1, 2, 3)])
        chk(f"wide ep6000 DiD {m} mean", g[0], em)
        chk(f"wide ep6000 DiD {m} sd", g[1], es)
    for (r, m), (em, es) in WIDE_RATIO_3000.items():
        g = A.msd([A.ratio(C, s, "3000", "wide", m, r) for s in (1, 2, 3)])
        chk(f"wide ep3000 ratio {r} {m} mean", g[0], em)
        chk(f"wide ep3000 ratio {r} {m} sd", g[1], es)
    for m, (em, es) in WIDE_NRMSE_DID_3000.items():
        g = A.msd([A.did(C, s, "3000", "wide", m, "v2_nrmse") for s in (1, 2, 3)])
        chk(f"wide ep3000 nRMSE DiD {m} mean", g[0], em)
        chk(f"wide ep3000 nRMSE DiD {m} sd", g[1], es)

    print(f"{n} reported values checked, {len(bad)} mismatch(es)")
    for b in bad:
        print("  MISMATCH", b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
