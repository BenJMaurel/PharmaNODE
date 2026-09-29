#!/usr/bin/env python3
"""Training-loss trajectories of the matched Km runs (chain_km_matched.sh), from their logs.
Per run, every 300 epochs: 'Train loss (one batch)' and the in-training test loss ('Total Loss' for FiLM, 'Loss' for
dose-cond). The in-training test loss uses a test scaler refit on the test split (handoff landmine 1.1), so only its
TREND within a run is meaningful. Reported: value at ep1500/2100/2400/3000 and the least-squares slope over
ep2100-3000 (per 1000 epochs), next to the typical point-to-point noise (median |diff| over ep1500-3000)."""
import re, glob, os, numpy as np
rows = []
for f in sorted(glob.glob("logs/train_kmm_*_s[123].log")):
    arch, coh, seed = re.search(r"train_kmm_(film|dc|lu)_(km0[09])_s(\d)", f).groups()
    txt = open(f).read()
    pat = r"Epoch (\d+) \[(?:FiLM|DoseCond|LuPK) Test\] \| (?:Total Loss|Loss|L2) ([-\d.eE+]+)"
    te = {int(e): float(v) for e, v in re.findall(pat, txt)}
    tr = [float(v) for v in re.findall(r"Train loss \(one batch\): ([-\d.eE+]+)", txt)]
    eps = sorted(te)
    trd = dict(zip(eps, tr[:len(eps)]))
    rows.append((arch, coh, seed, te, trd))
def slope(d, lo=2100, hi=3000):
    x = np.array([e for e in sorted(d) if lo <= e <= hi], float); y = np.array([d[e] for e in x])
    return np.polyfit(x / 1000, y, 1)[0] if len(x) >= 3 else np.nan
def noise(d):
    x = [d[e] for e in sorted(d) if e >= 1500]
    return np.median(np.abs(np.diff(x))) if len(x) > 2 else np.nan
for name, idx in (("TRAIN loss (one batch)", 4), ("in-training TEST loss (trend only)", 3)):
    print(f"\n{name}: value at ep 1500 / 2100 / 2400 / 3000 | slope ep2100-3000 per 1000 ep | point-to-point noise")
    for r in rows:
        d = r[idx]
        v = " / ".join(f"{d.get(e, np.nan):8.3f}" for e in (1500, 2100, 2400, 3000))
        print(f"  {r[0]:4s} {r[1]} s{r[2]}  {v}  | slope {slope(d):+8.3f} | noise {noise(d):6.3f}")
