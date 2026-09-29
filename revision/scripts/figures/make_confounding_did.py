#!/usr/bin/env python3
"""Regenerate the DiD interaction figure for the ISCB talk.

Replaces the n=1 version (DiD +8.84 / +2.69) with the six-seed result at the
converged budget of 6000 epochs.  Points are means over seeds, bars +-1 s.d.
"""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'confound_eval'))
import registry as reg

EP, ARM, MET, SEEDS = 6000, "wide", "v2_auc", (1, 2, 3, 4, 5, 6)
RED, BLUE, MUTED = "#c8313a", "#2a5f7f", "#7a7a7a"
OUT = "/Users/benjaminmaurel/Downloads/Latent_ODEs_NMI-2/figures/confounding_did.png"

rows = {}
# resolve against the repo root, not the cwd: this script is run from the repo,
# from scripts/figures/ and from the deck's directory, and a bare relative path
# only works in the first of those.
for p in ('results/confound_eval/cells.tsv', 'results/confound_eval/cells_v1fill.tsv'):
    for r in csv.DictReader(open(os.path.join(reg.REPO, p)), delimiter='\t'):
        if not r['epoch'].isdigit(): continue
        k = (int(r['seed']), int(r['epoch']), r['arm'], r['regime'], r['model'], r['trained'], r['evaluated'])
        if k in rows:
            for f, v in r.items():
                if rows[k].get(f, '') == '' and v != '': rows[k][f] = v
        else: rows[k] = dict(r)
conf, ctrl = reg.cohorts(ARM)
def series(m, tr):
    """(mean, sd) on the confounded then the de-confounded test set."""
    out = []
    for ev in (conf, ctrl):
        v = [float(rows[(s, EP, ARM, 'in', m, tr, ev)][MET]) for s in SEEDS]
        out.append((np.mean(v), np.std(v, ddof=1)))
    return out
def did(m):
    d = []
    for s in SEEDS:
        g = lambda tr, ev: float(rows[(s, EP, ARM, 'in', m, tr, ev)][MET])
        d.append((g(conf, ctrl) - g(ctrl, ctrl)) - (g(conf, conf) - g(ctrl, conf)))
    return np.mean(d), np.std(d, ddof=1)

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), sharey=True)
fig.suptitle("Evaluated on the SAME held-out patients; only the dose assignment differs",
             fontsize=12, color=MUTED, y=0.99)
x = [0, 1]
for ax, (m, title) in zip(axes, (("dc", "dose-cond (dose in the vector field)"),
                                 ("film", "OT-FiLM (dose as latent transport)"))):
    for tr, col, ls, mk, lab in ((conf, RED, "-", "o", "trained on confounded cohort"),
                                 (ctrl, BLUE, "--", "s", "trained on control cohort")):
        pts = series(m, tr)
        y = [p[0] for p in pts]; e = [p[1] for p in pts]

        ax.errorbar(x, y, yerr=e, color=col, ls=ls, marker=mk, ms=9, lw=2.6,
                    capsize=4, capthick=1.6, elinewidth=1.6, label=lab, zorder=3)
        # the two series can coincide (both ~15.1 on the dose-cond confounded
        # test set), so push the labels apart vertically rather than let them
        # print on top of one another
        dy = 7 if tr == conf else -13
        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points",
                        xytext=(12 if xi == 1 else -36, dy), color=col,
                        fontsize=11, fontweight="medium")
    mu, sd = did(m)
    ax.set_title(title, fontsize=12.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["dose still\ncorrelated with $K_m$", "dose\nrandomised"], fontsize=11)
    ax.set_xlim(-0.42, 1.42)
    lo = min(min(p[0] - p[1] for p in series(mm, t)) for mm in ("dc",) for t in (conf, ctrl)) if False else None
    ax.grid(axis="y", color="#e2e2e2", lw=0.9)
    ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.text(0.5, 0.055, f"difference-in-differences = {mu:+.2f} $\\pm$ {sd:.2f} pp",
            transform=ax.transAxes, ha="center", fontsize=12.5, fontweight="bold",
            color=col if False else (RED if m == "dc" else "#1f6f6f"),
            bbox=dict(boxstyle="round,pad=0.42", fc="white",
                      ec=RED if m == "dc" else "#1f6f6f", lw=1.6))
ylo, yhi = axes[0].get_ylim()
axes[0].set_ylim(ylo - 0.17 * (yhi - ylo), yhi)
axes[0].set_ylabel("counterfactual AUC RMSPE (%)", fontsize=11.5)
axes[0].legend(loc="upper left", frameon=True, fontsize=10.5)
fig.text(0.5, 0.935, f"$n={len(SEEDS)}$ seeds at {EP} epochs; bars $\\pm$1 s.d.",
         ha="center", fontsize=10, color=MUTED)
fig.tight_layout(rect=[0, 0, 1, 0.915])
fig.savefig(OUT, dpi=200)
print(f"wrote {OUT}")
for m, nm in (("dc", "dose-cond"), ("film", "OT-FiLM")):
    mu, sd = did(m); print(f"  {nm:<10} DiD {mu:+.2f} +- {sd:.2f}")
