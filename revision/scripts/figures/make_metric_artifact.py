#!/usr/bin/env python3
"""Why the DiD looked unstable: the metric, not the model.

AUC RMSPE is a PER-PATIENT relative error.  Patients asked to extrapolate far
downwards (8 mg -> 1 mg) have a tiny true AUC, so a modest absolute error becomes
a 200-400% relative one, and squaring it lets a single patient carry a third of
the pooled statistic.  Which extreme patients land in a given draw then decides
the DiD.  nRMSE normalises by the cohort's mean concentration instead, and is
stable across every draw.
"""
import sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SP = sys.argv[1]
OUT = "/Users/benjaminmaurel/Downloads/Latent_ODEs_NMI-2/figures/metric_artifact.png"
df = pd.read_csv(f"{SP}/perpatient.csv")
INK, MUTED, GRID = "#16160f", "#7d7a70", "#e3e2dc"
DOWN, UP, NRM, AUC = "#c8313a", "#2a5f7f", "#1f6f6f", "#c8313a"

fig = plt.figure(figsize=(14.8, 4.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.15], wspace=0.30)

# A -- relative error explodes at small true AUC, and that means big dose cuts
ax = fig.add_subplot(gs[0, 0])
down = df.d2 < df.d1
ax.scatter(df.true[~down], 100*np.abs(df.rel[~down]), s=11, color=UP, alpha=0.45,
           label="dose increased ($d_2>d_1$)", zorder=2)
ax.scatter(df.true[down], 100*np.abs(df.rel[down]), s=11, color=DOWN, alpha=0.55,
           label="dose reduced ($d_2<d_1$)", zorder=3)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("true counterfactual AUC (ng·h/mL)", fontsize=11)
ax.set_ylabel("|relative error|  (%)", fontsize=11)
ax.set_title("A. Small true AUC → huge relative error", fontsize=12.5, loc="left")
ax.axhline(100, color=MUTED, ls=":", lw=1, zorder=1)
ax.text(df.true.max()*0.75, 118, "100%", fontsize=9, color=MUTED, ha="right")
ax.grid(color=GRID, lw=0.8, which="both"); ax.set_axisbelow(True)
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.legend(loc="lower left", fontsize=9, frameon=True)

# B -- the pooled statistic is carried by a handful of patients
ax = fig.add_subplot(gs[0, 1])
for lab, col, ls in (("orig200", "#8a6d3b", "-"), ("selA", "#2a5f7f", "--"), ("selB", "#c8313a", "-")):
    g = df[df.set == lab]
    sq = np.sort(g.rel.values**2)[::-1]
    cum = np.cumsum(sq)/sq.sum()
    x = 100*np.arange(1, len(sq)+1)/len(sq)
    ax.plot(x, 100*cum, ls, color=col, lw=2.4, label=lab, zorder=3)
ax.plot([0,100],[0,100], color=MUTED, lw=1, ls=":", zorder=1)
ax.set_xlim(0, 20); ax.set_ylim(0, 100)
ax.set_xlabel("worst patients (% of cohort)", fontsize=11)
ax.set_ylabel("share of pooled RMSPE$^2$ (%)", fontsize=11)
ax.set_title("B. One patient in 500 carries ~30%", fontsize=12.5, loc="left")
ax.grid(color=GRID, lw=0.9); ax.set_axisbelow(True)
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.legend(loc="lower right", fontsize=9.5, frameon=True, title="test set", title_fontsize=9)

# C -- consequence: one metric flips, the other does not
ax = fig.add_subplot(gs[0, 2])
sets = ["original\n200", "new\nselA", "new\nselB"]
auc  = [-3.77, 3.29, 3.44]
nrm  = [-0.48, -1.23, -1.51]
x = np.arange(3); w = 0.36
ax.bar(x-w/2, auc, w, color=AUC, label="AUC RMSPE (per-patient relative)", zorder=3)
ax.bar(x+w/2, nrm, w, color=NRM, label="nRMSE (cohort-normalised)", zorder=3)
ax.axhline(0, color=INK, lw=1.1, zorder=4)
ax.set_xticks(x); ax.set_xticklabels(sets, fontsize=10)
ax.set_ylabel("DiD contrast, FiLM $-$ dose-cond (pp)", fontsize=11)
ax.set_title("C. The metric decides the conclusion", fontsize=12.5, loc="left")
ax.grid(axis="y", color=GRID, lw=0.9); ax.set_axisbelow(True)
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.text(0.5, 0.94, "below 0 = FiLM less damaged", transform=ax.transAxes,
        ha="center", fontsize=9.5, color=MUTED)
ax.legend(loc="lower left", fontsize=8.8, frameon=True)
fig.suptitle("The DiD instability was a metric artifact, not a model effect",
             fontsize=13.5, y=1.02, color=INK)
fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"wrote {OUT}")
