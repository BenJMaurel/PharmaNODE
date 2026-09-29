#!/usr/bin/env python3
"""What the model is actually asked to do, and what it is given to do it with.

The left panel is the visit that exists: a dose was administered and three
concentrations were measured. The right panel is the question -- the same patient at a
different dose, where nothing is measured at all. The model receives the new dose and
nothing else, transports the latent state through the affine operator, and decodes.

Regenerated from scratch (the original script was lost); curves are analytic
steady-state profiles chosen to match the published figure, not simulator output.

  python3 scripts/figures/make_counterfactual_protocol.py [outfile]
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

OUT = (sys.argv[1] if len(sys.argv) > 1 else
       "/Users/benjaminmaurel/Downloads/Latent_ODEs_NMI-2/figures/counterfactual_protocol.png")

BLUE, RED, TEAL = "#2e6183", "#c8313a", "#12807b"
GREY, GRID = "#8c8c8c", "#e8e7e2"
DOSE_OBS, DOSE_CF = 5, 3

# Steady-state profile: trough decay plus a first-order absorption bolus.
def profile(t, scale, ke, ka, b0=4.30, s0=3.50):
    return scale * (b0 * np.exp(-ke * t) + s0 * (np.exp(-ke * t) - np.exp(-ka * t)))

t = np.linspace(0, 12, 600)
c_obs = profile(t, 1.00, 0.121, 1.50)            # observed visit, 5 mg
c_true = profile(t, 0.610, 0.121, 1.50)          # simulator truth at 3 mg
c_pred = profile(t, 0.645, 0.118, 1.50)          # what the transported state decodes to
t_obs = np.array([0.0, 1.0, 3.0])
y_obs = np.array([4.30, 5.72, 5.52])             # the three measurements, with noise

fig = plt.figure(figsize=(11.6, 3.89))
gs = GridSpec(1, 3, width_ratios=[1, 0.40, 1], wspace=0.10)
axL, axM, axR = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

def dress(ax):
    ax.set_xlim(-0.6, 12.6); ax.set_ylim(0, 7.3)
    ax.set_xticks(range(0, 13, 2))
    ax.set_yticks(range(0, 8))
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_xlabel("time (h)", fontsize=12.5)
    for s in ax.spines.values():
        s.set_color("#4a4a4a"); s.set_linewidth(0.9)

# ---- the visit that exists -------------------------------------------------
axL.fill_between(t, 0, c_obs, color=BLUE, alpha=0.10, zorder=1)
axL.plot(t, c_obs, color=BLUE, lw=3.0, label="model fit", zorder=3)
axL.plot(t_obs, y_obs, "o", color=RED, ms=14, label="3 observations", zorder=4)
dress(axL)
axL.set_ylabel("concentration (ng/mL)", fontsize=12.5)
axL.set_title(f"TRUE VISIT — dose {DOSE_OBS} mg",
              fontsize=13, color=BLUE, pad=10)
axL.legend(fontsize=10.5, loc="upper right", frameon=True, framealpha=0.95,
           edgecolor="#bfbfbf")

# ---- the question ----------------------------------------------------------
axR.fill_between(t, 0, c_pred, color=RED, alpha=0.08, zorder=1)
axR.plot(t, c_true, color=GREY, lw=3.0, ls=(0, (5, 2.2)),
         label="simulator ground truth", zorder=3)
axR.plot(t, c_pred, color=RED, lw=3.0, label="model prediction", zorder=4)
dress(axR)
axR.set_yticklabels([])
axR.set_title(f"COUNTERFACTUAL — dose {DOSE_CF} mg",
              fontsize=13, color=RED, pad=10)
axR.legend(fontsize=10.5, loc="upper right", frameon=True, framealpha=0.95,
           edgecolor="#bfbfbf")
axR.text(0.97, 0.625, "nothing is measured here —\nthe model is given\nonly the new dose",
         transform=axR.transAxes, ha="right", va="top", fontsize=11.5, color=RED,
         linespacing=1.35)

# ---- the operator between them ---------------------------------------------
axM.axis("off")
axM.text(0.5, 0.645, r"$\Longrightarrow$", ha="center", va="center",
         fontsize=28, color=TEAL, transform=axM.transAxes)
axM.text(0.5, 0.435, "transport", ha="center", va="center", fontsize=13,
         fontweight="bold", color=TEAL, transform=axM.transAxes)
axM.text(0.5, 0.325, r"$z_0 \mapsto \gamma \odot z_0 + \beta$", ha="center", va="center",
         fontsize=11.5, color=TEAL, transform=axM.transAxes)

fig.savefig(OUT, dpi=205, bbox_inches="tight")
print(f"wrote {OUT}")
