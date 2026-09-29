#!/usr/bin/env python3
"""Answers two questions about the transport residual.

A. The two bands in z_v2.u are a patient sub-population, not a dose effect:
   96% of the lower band is one formulation x genotype group (Advagraf, CYP
   expresser).  It is structure the ENCODER built, present in the target itself.

B. The dose dependence is ONE axis, not many -- 96% of the between-dose variation
   is one-dimensional -- but that axis carries a SIGNED magnitude, near zero in the
   middle of the dose range and growing with opposite sign toward each end.  That
   sign flip is why mean residual directions at 1 mg and 8 mg look ~150 deg apart.
   Two monotonic components exist: v, large but drowned in patient variation, and
   u, small but in a quiet direction -- u is the one a linear probe can read.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SP = sys.argv[1]
OUT = "/Users/benjaminmaurel/Downloads/Latent_ODEs_NMI-2/figures/direction_anatomy.png"
D = np.load(f"{SP}/clusters.npz")
z1, z2, zn, d1, d2, st, u = D["z1"], D["z2"], D["zn"], D["d1"], D["d2"], D["st"], D["u"]
r = zn - z2
unit = lambda v: v / np.linalg.norm(v)
M = np.stack([r[np.isclose(d1, dd)].mean(0) for dd in range(1, 9)])
v = unit(np.linalg.svd(M - M.mean(0), full_matrices=False)[2][0])
if (r @ v)[np.isclose(d1, 1)].mean() < 0: v = -v

INK, MUTED, GRID = "#16160f", "#7d7a70", "#e3e2dc"
C_V, C_U, C_GRP, C_OTH = "#2a5f7f", "#c8313a", "#c8313a", "#5b7f8f"

fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))

# A: the two bands
ax = axes[0]
grp = (st[:, 1] == 0) & (st[:, 2] == 1)
p2 = z2 @ u
bins = np.linspace(p2.min(), p2.max(), 45)
ax.hist(p2[~grp], bins=bins, color=C_OTH, alpha=0.85, label="all other patients")
ax.hist(p2[grp], bins=bins, color=C_GRP, alpha=0.8, label="Advagraf + CYP expresser")
ax.set_xlabel("$z_{v2}\\cdot u$", fontsize=11)
ax.set_ylabel("patients", fontsize=11)
ax.set_title("A. The two bands are a patient group", fontsize=12.5, loc="left")
ax.grid(axis="y", color=GRID, lw=0.9); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(fontsize=9, frameon=True, loc="upper left")
ax.text(0.03, 0.55, "96% of the left band is this group\n(though most of the group sits\nin the main band too -- a shift,\nnot a clean split). Target-side\nstructure, not a dose effect.",
        transform=ax.transAxes, fontsize=8.6, color=INK,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=C_GRP, lw=1.2))

# B: one axis, signed magnitude
ax = axes[1]
doses = np.arange(1, 9)
mv = np.array([(r @ v)[np.isclose(d1, dd)].mean() for dd in doses])
mu = np.array([(r @ u)[np.isclose(d1, dd)].mean() for dd in doses])
ax.plot(doses, mv / np.abs(mv).max(), "o-", color=C_V, lw=2.4, ms=7, label="$v$  (largest mean shift)")
ax.plot(doses, mu / np.abs(mu).max(), "s--", color=C_U, lw=2.4, ms=7, label="$u$  (best read-out)")
ax.axhline(0, color=INK, lw=1.0); ax.axvline(4.5, color=MUTED, ls=":", lw=1.2)
ax.text(4.62, -0.42, "sign flips\nnear 4.5 mg", fontsize=9, color=MUTED)
ax.set_xlabel("administered dose $d_1$ (mg)", fontsize=11)
ax.set_ylabel("mean projection (scaled to $\\pm1$)", fontsize=11)
ax.set_title("B. One axis, signed by dose", fontsize=12.5, loc="left")
ax.set_xticks(doses)
ax.grid(color=GRID, lw=0.9); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(fontsize=9, frameon=True, loc="lower left")
ax.text(0.97, 0.95, "96% of the between-dose\nvariation is 1-dimensional",
        transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=MUTED, lw=1))

# C: why u is the readable one
ax = axes[2]
lab, sig, noi = [], [], []
for nm, ax_ in (("$v$", v), ("$u$", u)):
    proj = r @ ax_
    mus = np.array([proj[np.isclose(d1, dd)].mean() for dd in doses])
    lab.append(nm); sig.append(mus.std())
    noi.append(np.mean([proj[np.isclose(d1, dd)].std() for dd in doses]))
x = np.arange(2); w = 0.36
ax.bar(x - w/2, sig, w, color="#1f6f6f", label="between-dose signal", zorder=3)
ax.bar(x + w/2, noi, w, color=MUTED, label="within-dose patient noise", zorder=3)
for i in range(2):
    ax.text(i, max(sig[i], noi[i]) * 1.04, f"SNR {sig[i]/noi[i]:.2f}",
            ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=INK)
ax.set_ylim(0, max(max(sig), max(noi)) * 1.22)
ax.set_xticks(x); ax.set_xticklabels([f"{lab[0]}\nlarge but noisy", f"{lab[1]}\nsmall but quiet"], fontsize=10.5)
ax.set_ylabel("magnitude along the axis", fontsize=11)
ax.set_title("C. Why the probe reads $u$, not $v$", fontsize=12.5, loc="left")
ax.grid(axis="y", color=GRID, lw=0.9); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(fontsize=9, frameon=True)
fig.suptitle("The transport residual: one dose axis, two readable scales",
             fontsize=13.5, y=1.02, color=INK)
fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"wrote {OUT}")
