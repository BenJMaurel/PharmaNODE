#!/usr/bin/env python3
"""Monolix dataset from a scenario-4 training cohort: both visits, all 12 observations,
the 7 steady-state doses of each visit -- exactly the data the networks train on.

Matching the simulator (gen_tacro_film.py):
  * each visit is simulated from a ZERO state -> visit 2 is offset by 1000 h (washout);
  * visit-1 times are shifted by +200 h so every dose time is >= 0;
  * the t=0 sample is the PRE-dose trough (taken before the dose at the same instant)
    -> moved 0.001 h earlier, so no software-specific same-time ordering can apply the dose first.
Columns: ID TIME DV AMT CYP ST HT (ST = 1 Prograf, CYP = 1 expresser); '.' = missing.

    build_monolix_data.py <cohort> <out.csv>
"""
import sys, numpy as np, pandas as pd
cohort, out = sys.argv[1], sys.argv[2]
df = pd.read_csv(f"results/exp_film_run/{cohort}/virtual_cohort_film_train.csv")
df["TIMEn"] = pd.to_numeric(df.TIME, errors="coerce")
df["AMTn"] = pd.to_numeric(df.AMT, errors="coerce"); df["DVn"] = pd.to_numeric(df.DV, errors="coerce")
rows = []
for pid, g in df.groupby("ID"):
    for v, gv in g.groupby("VISIT"):
        off = 200.0 + (1000.0 if v == 2 else 0.0)
        cov = dict(CYP=int(gv.CYP.iloc[0]), ST=int(gv.ST.iloc[0]), HT=float(gv.HT.iloc[0]))
        for r in gv.itertuples():
            if not np.isnan(r.AMTn):
                rows.append(dict(ID=pid, TIME=r.TIMEn + off, DV=".", AMT=r.AMTn, **cov))
            elif not np.isnan(r.DVn):
                t = r.TIMEn - (1e-3 if r.TIMEn == 0 else 0.0)
                rows.append(dict(ID=pid, TIME=t + off, DV=r.DVn, AMT=".", **cov))
m = pd.DataFrame(rows)
m["_o"] = pd.to_numeric(m.TIME); m = m.sort_values(["ID", "_o"]).drop(columns="_o")
m.to_csv(out, index=False)
n_obs = (m.DV != ".").sum(); n_dose = (m.AMT != ".").sum(); n_id = m.ID.nunique()
print(f"{cohort}: {n_id} patients, {n_obs} observations ({n_obs / n_id:.0f}/patient), {n_dose} doses ({n_dose / n_id:.0f}/patient)")
assert n_obs == 24 * n_id and n_dose == 14 * n_id, "expected 2 visits x (12 obs + 7 doses) per patient"
