#!/usr/bin/env python3
"""Why Km cannot be inferred from one dose, and K_ELIM-like parameters can.

Elimination in this simulator is Michaelis-Menten, Vmax*C/(Km+C).  At a single
dose level a change in Km can be almost exactly offset by a change in Vmax: three
Km values spanning a factor of 6 give concentration curves within ~1% of each
other.  Only across dose levels does Km reveal itself, because it governs the
CURVATURE of the dose-exposure relation.

That is the whole reason the confounding experiment works: Km is not merely
withheld from the model, it is unidentifiable from the data the model is given --
so the only route to it is the dose, which is the shortcut.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SP = sys.argv[1]
OUT = "/Users/benjaminmaurel/Downloads/Latent_ODEs_NMI-2/figures/km_identifiability.png"
D = np.load(f"{SP}/km_identifiability.npz")
pairs, tw, obs = D["pairs"], D["tw"], D["obs"]
c4, c1, c8, dc = D["curves4"], D["curves1"], D["curves8"], D["dosecurve"]
INK, MUTED, GRID = "#16160f", "#7d7a70", "#e3e2dc"
COL = ["#f0c27a", "#c8313a", "#6b2020"]          # sequential: Km is a magnitude
LBL = [f"$K_m\\times{f:g}$" for f, _, _ in pairs]

fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.5))

# A -- one dose: indistinguishable
ax = axes[0]
for i in range(3):
    ax.plot(tw, c4[i], color=COL[i], lw=2.4, label=f"{LBL[i]}  ($V_{{max}}$={pairs[i,2]:.3f})", zorder=3)
for t in obs[(obs > 0) & (obs <= 24)]:
    ax.axvline(t, color=MUTED, lw=0.7, ls=":", zorder=1)
ax.plot([], [], color=MUTED, lw=0.7, ls=":", label="sampling times")
ax.set_xlabel("time since dose (h)", fontsize=11)
ax.set_ylabel("concentration (ng/mL)", fontsize=11)
ax.set_title("A. At 4 mg: a 6$\\times$ range of $K_m$, one curve", fontsize=12.5, loc="left")
ax.grid(color=GRID, lw=0.9); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(fontsize=8.6, frameon=True, loc="upper right")
dm = np.abs(c4 - c4[1]).max()
ax.text(0.97, 0.06, f"curves differ by at most {100*dm/c4[1].max():.1f}% of the peak\n"
        f"— a 6$\\times$ change in $K_m$, hidden by $V_{{max}}$",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5, color=INK,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=COL[1], lw=1.3))

# B -- other doses: they separate
ax = axes[1]
for i in range(3):
    ax.plot(tw, c1[i], color=COL[i], lw=2.2, zorder=3)
    ax.plot(tw, c8[i], color=COL[i], lw=2.2, ls="--", zorder=3)
ax.set_xlabel("time since dose (h)", fontsize=11)
ax.set_ylabel("concentration (ng/mL)", fontsize=11)
ax.set_title("B. Same parameters at 1 mg (solid) and 8 mg (dashed)", fontsize=12.5, loc="left")
ax.set_yscale("log")
ax.grid(color=GRID, lw=0.9, which="both"); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.text(0.97, 0.10, "now they are far apart\n— $K_m$ sets the curvature",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5, color=INK,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=COL[1], lw=1.3))

# C -- the dose-exposure fan
ax = axes[2]
for i in range(3):
    ax.plot(dc[:, 0], dc[:, i+1], "o-", color=COL[i], lw=2.4, ms=6, label=LBL[i], zorder=3)
ax.axvline(4.0, color=MUTED, ls=":", lw=1.2)
ax.annotate("matched here\n(the observed dose)", xy=(4.0, dc[3, 1]),
            xytext=(4.5, dc[0, 3]*3.2), fontsize=9.5, color=MUTED,
            arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.1))
ax.set_xlabel("dose (mg)", fontsize=11)
ax.set_ylabel("AUC$_{0-24}$ (ng·h/mL)", fontsize=11)
ax.set_title("C. The counterfactual question $K_m$ decides", fontsize=12.5, loc="left")
ax.grid(color=GRID, lw=0.9); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(fontsize=9, frameon=True, loc="upper left")
sp1 = 100*(dc[0,1:].max()-dc[0,1:].min())/dc[0,1:].mean()
sp8 = 100*(dc[-1,1:].max()-dc[-1,1:].min())/dc[-1,1:].mean()
ax.text(0.97, 0.06, f"AUC spread: {sp1:.0f}% at 1 mg, {sp8:.0f}% at 8 mg",
        transform=ax.transAxes, ha="right", fontsize=9.5, color=INK,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=MUTED, lw=1))
fig.suptitle("$K_m$ is invisible at the dose you observe, and decisive at the dose you are asked about",
             fontsize=13.5, y=1.02, color=INK)
fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"wrote {OUT}")
