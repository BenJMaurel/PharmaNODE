#!/usr/bin/env python3
"""Figure: the transport lands on the target posterior, but its residual is
oriented along the administered dose."""
import glob, os, sys
import numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib.read_tacro import extract_gen_tac_film, TacroFilmDataset, collate_fn_tacro_film  # noqa: E402
from probe_entanglement import load_film_model, encode_mu, film_transport  # noqa: E402

BEFORE, AFTER, INK, MUTED, GRID = "#eb6834", "#2a78d6", "#0b0b0b", "#898781", "#e1e0d9"
OUT = "/Users/benjaminmaurel/Downloads/Latent_ODEs_NMI-2/figures/transport_residual.png"


def states(exp, ck):
    d = f"results/exp_film_run/{exp}"
    data, _ = extract_gen_tac_film(file_path=[f"{d}/virtual_cohort_film_train.csv",
                                              f"{d}/virtual_cohort_film_test.csv"])
    b = next(iter(DataLoader(TacroFilmDataset(data), batch_size=4000, shuffle=False,
                             collate_fn=lambda x: collate_fn_tacro_film(x, torch.device("cpu")))))
    m, _ = load_film_model(ck, torch.device("cpu"))
    with torch.no_grad():
        z1 = encode_mu(m, b["observed_data_v1"], b["observed_tp_v1"], b["dose_v1"], b["static_v1"])
        z2 = encode_mu(m, b["observed_data_v2"], b["observed_tp_v2"], b["dose_v2"], b["static_v2"])
        zn = film_transport(m, z1, b["dose_v1"], b["dose_v2"], b["delta_t"], b["t_v1"])
    return (z1.cpu().numpy(), z2.cpu().numpy(), zn.cpu().numpy(),
            b["dose_v1"].cpu().numpy().ravel())


probe = lambda X, y: float(r2_score(y, cross_val_predict(
    RidgeCV(alphas=np.logspace(-3, 3, 13)), X, y, cv=5)))

z1, z2, zn, d1 = states("confound_km00",
                        glob.glob("results/seeds/s1wide/exp_film_run/confound_km00/traj/*_ep006000.ckpt")[0])
_, c2, cn, cd1 = states("confound_km09",
                        glob.glob("results/seeds/s1wide/exp_film_run/confound_km09/traj/*_ep006000.ckpt")[0])

before, after = np.linalg.norm(z1 - z2, axis=1), np.linalg.norm(zn - z2, axis=1)
radius = np.median(np.linalg.norm(z2 - z2.mean(0), axis=1))
res = zn - z2
u = np.linalg.svd(res, full_matrices=False)[2][0]
proj = res @ u
r2_res, r2_conf = probe(res, d1), probe(cn - c2, cd1)

fig, ax = plt.subplots(1, 3, figsize=(13.4, 4.3))

# (a) ECDF: how far each state sits from the real visit-2 posterior
for v, c, lab, xy in ((before, BEFORE, "before transport\n$z_{base}$", (1.35, .45)),
                      (after, AFTER, "after transport\n$z_{new}$", (.30, .86))):
    xs = np.sort(v); ys = np.arange(1, len(xs) + 1) / len(xs)
    ax[0].plot(xs, ys, color=c, lw=2)
    ax[0].annotate(lab, xy=xy, fontsize=9.5, color=c, linespacing=1.4)
ax[0].axvline(radius, color=MUTED, lw=1.1, ls="--")
ax[0].text(radius + .05, .06, "radius of the\n$z_{v2}$ cloud", fontsize=8.5, color=MUTED, linespacing=1.4)
ax[0].set_xlim(0, 2.6); ax[0].set_ylim(0, 1.02)
ax[0].set_xlabel("distance to the real visit-2 posterior")
ax[0].set_ylabel("fraction of patients below")
ax[0].set_title(f"(a)  the transport closes {100*(1-np.median(after)/np.median(before)):.0f}% of the gap",
                fontsize=11, color=INK, pad=9)

# (b) the residual predicts the administered dose -- shown as the probe's own prediction
pred = cross_val_predict(RidgeCV(alphas=np.logspace(-3, 3, 13)), res, d1, cv=5)
rng = np.random.default_rng(0)
ax[1].scatter(d1 + rng.normal(0, .012, len(d1)), pred, s=11, color=AFTER, alpha=.45, lw=0)
lim = [min(d1.min(), pred.min()) - .05, max(d1.max(), pred.max()) + .05]
ax[1].plot(lim, lim, color=MUTED, lw=1.1, ls="--")
ax[1].set_xlim(lim); ax[1].set_ylim(lim)
ax[1].set_xlabel("administered dose $d_1$ (normalised)")
ax[1].set_ylabel("$d_1$ predicted from the residual alone")
ax[1].set_title("(b)  the residual is not noise", fontsize=11, color=INK, pad=9)
ax[1].text(.04, .96, f"randomised-dose cohort   $R^2={r2_res:.2f}$\n"
                     f"confounded cohort        $R^2={r2_conf:.2f}$",
           transform=ax[1].transAxes, va="top", fontsize=9, color=MUTED, linespacing=1.6)

# (c) consequence: how much dose each state carries
labels = ["$z_{base}$ (visit 1)", "$z_{v2}$ (real visit 2)", "$z_{new}$ (transported)", "the residual alone"]
vals = [probe(z1, d1), probe(z2, d1), probe(zn, d1), r2_res]
cols = [BEFORE, MUTED, AFTER, "#6da7ec"]
y = np.arange(4)[::-1]
ax[2].barh(y, vals, height=.6, color=cols, lw=0)
for yi, v in zip(y, vals):
    ax[2].text(max(v, 0) + .025, yi, f"{v:.2f}", va="center", fontsize=9.5, color=INK)
ax[2].set_yticks(y); ax[2].set_yticklabels(labels, fontsize=9.5)
ax[2].set_xlim(0, 1.15); ax[2].set_xlabel("$R^2$ decoding the administered dose $d_1$")
ax[2].set_title("(c)  a small residual is enough", fontsize=11, color=INK, pad=9)

for a in ax:
    a.spines[["top", "right"]].set_visible(False)
    a.spines[["left", "bottom"]].set_color("#c3c2b7")
    a.tick_params(colors=MUTED, labelsize=9)
    a.xaxis.label.set_color(MUTED); a.yaxis.label.set_color(MUTED)
    a.grid(axis="x", color=GRID, lw=.8, alpha=.7); a.set_axisbelow(True)
ax[2].grid(axis="y", lw=0)
fig.suptitle("The transport lands on the target posterior; what it leaves behind points along the dose",
             fontsize=12.5, color=INK, y=1.0)
fig.tight_layout(rect=[0, 0, 1, .94])
fig.savefig(OUT, dpi=190, bbox_inches="tight", facecolor="#fcfcfb")
print("wrote", OUT)
