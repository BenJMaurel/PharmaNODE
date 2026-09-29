#!/usr/bin/env python3
"""How much does the dose rewrite the vector field?

The talk figure asserts that in dose-cond the dose enters f, so two doses give two
different fields, whereas OT-FiLM's field is autonomous and the dose only moves z0.
That is an architectural claim, and it is checkable rather than merely illustrative:
for FiLM the field has no dose argument at all, so the effect must be EXACTLY zero.
For dose-cond we measure how far f(z,t,d) moves as d sweeps the clinical grid, at
latent states the model actually visits.

Three read-outs, all at states drawn from the real posterior:
  1. relative displacement  ||f(z,d_hi) - f(z,d_lo)|| / ||f_bar(z)||
  2. turning angle between f(z,d_lo) and f(z,d_hi)
  3. variance decomposition: what share of the field's variability is driven by the
     dose rather than by where you are in latent space
"""
import argparse, glob, sys
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, '.')
from lib.read_tacro import extract_gen_tac_film, TacroFilmDataset, collate_fn_tacro_film
from probe_entanglement import load_film_model, load_dosecond_model
dev = torch.device("cpu")
DOSE_MAX = 8.0


def states(ckpt, exp, family, n=400):
    """Posterior-mean latents for real patients, plus states visited along the solve."""
    d = f"results/exp_film_run/{exp}"
    ev, _ = extract_gen_tac_film(file_path=[f"{d}/virtual_cohort_film_test.csv"])
    b = next(iter(DataLoader(TacroFilmDataset(ev), batch_size=n, shuffle=False,
                             collate_fn=lambda x: collate_fn_tacro_film(x, dev))))
    m, _ = (load_film_model if family == "film" else load_dosecond_model)(ckpt, dev)
    with torch.no_grad():
        if family == "film":
            from probe_entanglement import encode_mu
            z = encode_mu(m, b["observed_data_v1"], b["observed_tp_v1"], b["dose_v1"], b["static_v1"])
        else:
            z = m.encode(b["observed_data_v1"], b["observed_tp_v1"], b["dose_v1"], b["static_v1"])[0].squeeze(0)
    return m, z, b


def field(m, z, dose=None, family="dc", t=0.0):
    """f(z, t, d) evaluated pointwise. dose is in model units (mg / DOSE_MAX)."""
    f = m.diffeq_solver.ode_func
    with torch.no_grad():
        if family == "dc":
            f.set_dose(torch.full((z.size(0), 1), float(dose)))
            if getattr(f, "n_occ", 0): f.set_occ(None)
        return f.forward(t, z).numpy()


def report(family, ckpt, exp, grid=(1, 2, 4, 8)):
    m, z, b = states(ckpt, exp, family)
    F = np.stack([field(m, z, d / DOSE_MAX, family) for d in grid])      # [D, N, L]
    lo, hi = F[0], F[-1]
    fbar = F.mean(axis=0)
    nrm = np.linalg.norm(fbar, axis=-1)
    rel = np.linalg.norm(hi - lo, axis=-1) / np.maximum(nrm, 1e-12)
    cos = (lo * hi).sum(-1) / np.maximum(np.linalg.norm(lo, axis=-1) * np.linalg.norm(hi, axis=-1), 1e-12)
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    # variance decomposition over the (state, dose) grid
    tot = F.reshape(-1, F.shape[-1]).var(axis=0).sum()
    within_state = F.var(axis=0).mean(axis=0).sum()        # varying dose, state fixed
    print(f"\n  {family:5s}  {exp}")
    print(f"    ||f(d=8) - f(d=1)|| / ||f||   median {np.median(rel):7.4f}   "
          f"IQR {np.percentile(rel,25):.4f}-{np.percentile(rel,75):.4f}   max {rel.max():.4f}")
    print(f"    turning angle (degrees)        median {np.median(ang):7.3f}   max {ang.max():.3f}")
    print(f"    share of field variance from the dose: {100*within_state/max(tot,1e-12):6.3f}%")
    return rel, ang


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--exp", default="confound_vc00")
    a = ap.parse_args()
    print(f"Vector-field dose sensitivity, evaluated at {400} real posterior states")
    for fam, pat in (("dc",   f"results/vc/dc_s1/exp_dosecond_run/{a.exp}/traj/*_ep006000.ckpt"),
                     ("film", f"results/vc/film_s1/exp_film_run/{a.exp}/traj/*_ep006000.ckpt")):
        h = sorted(glob.glob(pat))
        if not h: print(f"  {fam}: no checkpoint at {pat}"); continue
        report(fam, h[0], a.exp)
