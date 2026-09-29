#!/usr/bin/env python3
"""AIC/BIC table for the hidden-saturation calibration (handoff 11): linear vs Michaelis-Menten per k."""
import json, os
OUT = "results/sat/calib"
def get(k, s):
    f = f"{OUT}/sat{k}_{s}/estimates.json"
    if not os.path.exists(f): return None
    return json.load(open(f))
def ll_block(e):
    L = e["loglik"]; L = L.get("importanceSampling", L)
    if isinstance(L, list):   # write_json drops the names: OFV, AIC, BIC, BICc, chosenDegree[, standardError]
        return dict(zip(("ofv", "aic", "bic", "bicc", "degree", "se"), (float(v) for v in L)))
    return {kk.lower(): float(v) for kk, v in L.items() if isinstance(v, (int, float))}
print("truth: a 0.71, b 0.113 (paper noise); omega_Vmax 0.283")
print(f"{'k':>3} {'model':>4} {'-2LL':>10} {'AIC':>10} {'BIC':>10} {'BICc':>10} | Km_pop (true 0.01k)  Vmax_pop (0.212k)  CL_pop  min")
for k in (10, 20, 30):
    rows = {}
    for s in ("lin", "mm"):
        e = get(k, s)
        if e is None: print(f"{k:>3} {s:>4}  (missing)"); continue
        L = ll_block(e); rows[s] = L; est = e["estimates"]
        ofv = L.get("ofv", L.get("-2ll", float("nan")))
        extra = (f"Km {est.get('Km_pop', float('nan')):.4g}  Vmax {est.get('Vmax_pop', float('nan')):.4g}" if s == "mm"
                 else f"CL {est.get('CL_pop', float('nan')):.4g}")
        # residual error (truth a 0.71, b 0.113) and elimination IIV: where did the misfit go?
        om = est.get('omega_CL', est.get('omega_Vmax', float('nan')))
        extra += f"  | a {est.get('a', float('nan')):.3f} b {est.get('b', float('nan')):.3f} omega_elim {om:.3f}"
        print(f"{k:>3} {s:>4} {ofv:10.1f} {L.get('aic', float('nan')):10.1f} {L.get('bic', float('nan')):10.1f} "
              f"{L.get('bicc', float('nan')):10.1f} | {extra}  ({e['minutes']:.0f} min)")
    if len(rows) == 2:
        d = {m: rows["mm"].get(m, float("nan")) - rows["lin"].get(m, float("nan")) for m in ("aic", "bic", "bicc")}
        verdict = "linear preferred/tie" if d["bic"] >= -2 else "MM preferred"
        print(f"    delta (MM - lin): AIC {d['aic']:+.1f}  BIC {d['bic']:+.1f}  BICc {d['bicc']:+.1f}  -> {verdict} on BIC")
