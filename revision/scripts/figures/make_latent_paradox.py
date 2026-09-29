#!/usr/bin/env python3
"""Why d1 is readable from z_new even though z_new sits on top of z_v2.

The transported point lands within 19% of the cloud radius of its target, and the
target carries essentially no information about the administered dose (R2 = 0.02).
Yet a linear probe recovers d1 from z_new at R2 = 0.90.  The resolution is
geometric: the residual is small in NORM but points into directions where the
target cloud has almost no variance, so along those directions nothing competes
with it.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SP = sys.argv[1]
OUT = "/Users/benjaminmaurel/Downloads/Latent_ODEs_NMI-2/figures/latent_paradox.png"
D = np.load(f"{SP}/geom.npz")
z2, zn, d1, u, Vt, var = D["z2"], D["zn"], D["d1"], D["u"], D["Vt"], D["var"]
d1 = d1 * 8.0          # stored normalised by the dose maximum; back to mg
r = zn - z2

INK, MUTED, GRID = "#16160f", "#7d7a70", "#e3e2dc"
TGT, NEW = "#2a5f7f", "#c8313a"
CMAP = "cividis"                      # sequential: d1 is a magnitude, not a category

fig = plt.figure(figsize=(14.6, 4.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.05, 1.25], wspace=0.34)

# --- A: the target cloud is effectively 2-dimensional -----------------------
ax = fig.add_subplot(gs[0, 0])
k = np.arange(1, len(var) + 1)
ax.semilogy(k, var, "o-", color=TGT, lw=2.2, ms=7, zorder=3)
ax.set_xlabel("principal component of $z_{v2}$", fontsize=11)
ax.set_ylabel("variance (log scale)", fontsize=11)
ax.set_title("A. The target cloud is nearly flat", fontsize=12.5, loc="left")
ax.set_xticks(k)
ax.grid(axis="y", color=GRID, lw=0.9); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.annotate(f"PC1-2 hold {100*var[:2].sum()/var.sum():.1f}% of the variance",
            xy=(2, var[1]), xytext=(4.1, var[0]*0.55), fontsize=9.5, color=MUTED,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
ax.annotate("PC5-10: variance $\\sim10^{-4}$\nthe cloud barely occupies them",
            ha="left", va="center",
            xy=(7, var[6]), xytext=(1.15, var[6]*0.30), fontsize=9.5, color=MUTED,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))

# --- B: where the d1-readout direction lives --------------------------------
ax = fig.add_subplot(gs[0, 1])
share = (Vt @ u) ** 2
ax.bar(k, share, color=NEW, width=0.62, zorder=3)
ax.set_xlabel("principal component of $z_{v2}$", fontsize=11)
ax.set_ylabel("share of the $d_1$-readout direction $u$", fontsize=11)
ax.set_title("B. $u$ lives where the cloud does not", fontsize=12.5, loc="left")
ax.set_xticks(k)
ax.grid(axis="y", color=GRID, lw=0.9); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.text(0.97, 0.93, f"{100*share[4:].sum():.0f}% of $u$ sits in PC5-10\n"
                    f"(jointly {100*var[4:].sum()/var.sum():.2f}% of the variance)",
        transform=ax.transAxes, ha="right", va="top", fontsize=9.5, color=INK,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=NEW, lw=1.3))

# --- C: the displacement is dose-ordered; the target is not ------------------
ax = fig.add_subplot(gs[0, 2])
pr, p2 = r @ u, z2 @ u
jit = (np.random.default_rng(0).random(len(d1)) - 0.5) * 0.42
ax.scatter(d1 + jit, p2, s=11, color=TGT, alpha=0.30, zorder=2,
           label=f"target $z_{{v2}}\\cdot u$   (r = {np.corrcoef(d1,p2)[0,1]:+.2f})")
ax.scatter(d1 + jit, pr, s=11, color=NEW, alpha=0.55, zorder=3,
           label=f"displacement $(z_{{new}}\\!-\\!z_{{v2}})\\cdot u$   (r = {np.corrcoef(d1,pr)[0,1]:+.2f})")
for y, col in ((p2, TGT), (pr, NEW)):
    mu = [y[np.isclose(d1, dd)].mean() for dd in range(1, 9)]
    ax.plot(range(1, 9), mu, "-", color=col, lw=2.6, zorder=4,
            path_effects=None, marker="o", ms=6, mec="white", mew=1.0)
ax.axhline(0, color=MUTED, lw=0.9, ls=":", zorder=1)
ax.set_xlabel("administered dose $d_1$ (mg)", fontsize=11)
ax.set_ylabel("projection on $u$", fontsize=11)
ax.set_title("C. The displacement carries the dose", fontsize=12.5, loc="left")
ax.set_xticks(range(1, 9))
ax.grid(color=GRID, lw=0.9); ax.set_axisbelow(True)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.legend(loc="upper left", fontsize=9, frameon=True)

fig.suptitle("A small residual in a thin direction is still perfectly readable",
             fontsize=13.5, y=1.005, color=INK)
fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"wrote {OUT}")
print(f"  corr(d1, residual.u) = {np.corrcoef(d1, r@u)[0,1]:+.3f}")
print(f"  corr(d1, z_v2.u)     = {np.corrcoef(d1, z2@u)[0,1]:+.3f}")
