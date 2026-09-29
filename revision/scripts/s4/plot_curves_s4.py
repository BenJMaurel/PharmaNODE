#!/usr/bin/env python3
"""Figure: OT-FiLM vs dose-cond Visit-2 counterfactual curves, scenario 4.

Examples are NOT hand-picked: patients sit at fixed percentiles of the per-patient
difference d = nRMSE(FiLM) - nRMSE(dose-cond), from the patient FiLM helps most to
the one it hurts most, with the median patient in the middle.
    plot_curves_s4.py <in.npz> <out.png> [title]
"""
import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

z = np.load(sys.argv[1]); out = sys.argv[2]
title = sys.argv[3] if len(sys.argv) > 3 else ""
FILM, DC, TRUTH, INK, MUTED, GRID = "#0E7C7B", "#C1272D", "#2b2b2b", "#222222", "#6b6b6b", "#e6e6e6"
nf, nd = 100 * z["film_nrmse"], 100 * z["dc_nrmse"]
d = nf - nd
pcts = [5, 18, 31, 44, 56, 69, 82, 95]
order = np.argsort(d)
pick = [order[int(round(p / 100 * (len(d) - 1)))] for p in pcts]

fig = plt.figure(figsize=(15.5, 9.2))
gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.25], hspace=0.62, wspace=0.28)
for k, (i, p) in enumerate(zip(pick, pcts)):
    ax = fig.add_subplot(gs[k // 4, k % 4])
    ax.plot(z["dense_tp"], z["dc_dense"][i], color=DC, lw=1.8, ls=(0, (5, 2)), zorder=2)
    ax.plot(z["dense_tp"], z["film_dense"][i], color=FILM, lw=1.8, zorder=3)
    ax.plot(z["tp"], z["y"][i], "o", color=TRUTH, ms=5, mec="white", mew=0.8, zorder=4)
    ax.set_title(f"{p}th pct of Δ   ({z['d1'][i]:g}→{z['d2'][i]:g} mg, "
                 f"{'Prograf' if z['prograf'][i] else 'Advagraf'})", fontsize=9.5, color=INK)
    ax.text(0.97, 0.95, f"FiLM {nf[i]:.1f}%\ndose-cond {nd[i]:.1f}%", transform=ax.transAxes,
            ha="right", va="top", fontsize=8.5, color=INK,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=GRID, lw=0.8))
    ax.set_xlim(0, 24); ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=8, colors=MUTED)
    if k % 4 == 0: ax.set_ylabel("concentration (ng/mL)", fontsize=9, color=INK)
    if k // 4 == 1: ax.set_xlabel("time after dose (h)", fontsize=9, color=INK)

# per-patient scatter
ax = fig.add_subplot(gs[2, 0:2])
lim = np.percentile(np.r_[nf, nd], 99.5) * 1.05
ax.plot([0, lim], [0, lim], color=MUTED, lw=1, ls=":", zorder=1)
ax.scatter(nf, nd, s=10, color="#4a4a4a", alpha=0.35, lw=0, zorder=2)
ax.scatter(nf[pick], nd[pick], s=34, color="none", edgecolor=INK, lw=1.2, zorder=3)
win = np.mean(d < 0) * 100
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("OT-FiLM per-patient nRMSE (%)", fontsize=9.5, color=INK)
ax.set_ylabel("dose-cond nRMSE (%)", fontsize=9.5, color=INK)
ax.set_title("per-patient error, one dot per patient", fontsize=10, color=INK)
ax.text(0.04, 0.96, f"above the diagonal: FiLM closer\nFiLM closer for {win:.0f}% of patients\n"
        f"corr of the two errors {np.corrcoef(nf, nd)[0,1]:.2f}", transform=ax.transAxes,
        ha="left", va="top", fontsize=8.5, color=INK,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID, lw=0.8))
ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)

# where along the curve the error sits
ax = fig.add_subplot(gs[2, 2:4])
scale = z["y"].mean(axis=1, keepdims=True)
for arr, col, ls, lab in ((z["film_sparse"], FILM, "-", "OT-FiLM"), (z["dc_sparse"], DC, (0, (5, 2)), "dose-cond")):
    e = np.median(np.abs(arr - z["y"]) / scale, axis=0) * 100
    ax.plot(z["tp"], e, color=col, lw=2, ls=ls, marker="o", ms=5, label=lab)
ax.set_xlabel("time after dose (h)", fontsize=9.5, color=INK)
ax.set_ylabel("median |error| / patient mean conc. (%)", fontsize=9.5, color=INK)
ax.set_title("where along the curve the two differ", fontsize=10, color=INK)
tp = z["tp"]; y = z["y"]
seg = np.diff(tp) * (y[:, 1:] + y[:, :-1]) / 2
auc_early = 100 * np.median(seg[:, tp[1:] <= 6].sum(1) / seg.sum(1))
ax.axvspan(0, 4, color="#dfeeee", zorder=0, lw=0)
# the verdict per window is computed, never assumed: it differs between regimes
def verdict(m):
    ef = np.median(np.abs(z["film_sparse"][:, m] - y[:, m]) / scale)
    ed = np.median(np.abs(z["dc_sparse"][:, m] - y[:, m]) / scale)
    if abs(ef - ed) <= 0.1 * max(ef, ed): return "the two are similar"
    return "FiLM closer" if ef < ed else "dose-cond closer"
early, late = tp <= 4, tp >= 6
ax.text(0.98, 0.96, f"shaded 0-4 h ({early.sum()} of {len(tp)} target points): {verdict(early)}\n"
        f"6-24 h (~{100 - auc_early:.0f}% of the AUC): {verdict(late)}",
        transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID, lw=0.8))
ax.set_xlim(-0.5, 24.5)
ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)

handles = [Line2D([], [], color=FILM, lw=2, label="OT-FiLM"),
           Line2D([], [], color=DC, lw=2, ls=(0, (5, 2)), label="dose-cond"),
           Line2D([], [], color=TRUTH, marker="o", ls="", ms=6, mec="white", label="true Visit-2 curve (noisy)")]
fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, 0.995))
if title: fig.text(0.5, 0.955, title, ha="center", fontsize=10, color=MUTED)
fig.savefig(out, dpi=160, bbox_inches="tight")
print(f"wrote {out}  |  mean nRMSE FiLM {nf.mean():.2f}  dose-cond {nd.mean():.2f}  |  "
      f"FiLM better for {win:.1f}%  |  median Δ {np.median(d):+.2f}  IQR {np.percentile(d,25):+.2f} to {np.percentile(d,75):+.2f}")
