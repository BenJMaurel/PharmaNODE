#!/usr/bin/env python3
"""Linear-elimination Monolix estimates (fit_s4_select.R, struct lin) -> an estimates.json that ebe_popmodel.py can use.
Linear elimination CL*C is written as Michaelis-Menten with Km = 1000 mg/L (>= 10^4 x any concentration here) and
Vmax = CL * Km, so Vmax*C/(Km+C) = CL*C to < 1e-4 relative; omega_Km = 1e-3 pins eta_Km at 0. Covariates on CL carry
over to Vmax.      lin_to_mm_spec.py <lin estimates.json> <out.json>"""
import sys, json
e = json.load(open(sys.argv[1])); x = e["estimates"]; KM = 1000.0
m = {k: x[k] for k in ("Ktr_pop", "beta_Ktr_ST_1", "Vc_pop", "beta_Vc_ST_1", "Q_pop", "Vp_pop",
                       "omega_Ktr", "omega_Q", "omega_Vc", "omega_Vp", "a", "b")}
m.update(Vmax_pop=x["CL_pop"] * KM, Km_pop=KM, beta_Vmax_tHT=x["beta_CL_tHT"], beta_Vmax_CYP_1=x["beta_CL_CYP_1"],
         omega_Vmax=x["omega_CL"], omega_Km=1e-3)
json.dump(dict(source=sys.argv[1], note="linear model as MM with Km=1000", estimates=m), open(sys.argv[2], "w"), indent=1)
print("wrote", sys.argv[2])
