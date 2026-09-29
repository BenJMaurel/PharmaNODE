#!/usr/bin/env python3
"""The paper's gen_tacro.py (committed, one visit per patient) with fixed seeds, as scripts/scen_rerun/gen_seeded.py,
PLUS dose-to-dose variability in bioavailability (handoff 12): each of the 7 simulated doses is absorbed as
dose * exp(kappa * e), e ~ N(0, 1) drawn from its OWN stream (RandomState(data_seed + 99991)), so every other draw
(covariates, etas, residual noise) is identical to gen_seeded.py with the same seed. The recorded AMT stays nominal.
kappa = 0 reproduces gen_seeded.py byte for byte. The per-dose factors are written to results/<exp>/idv_truth.csv
(ID, F1..F7; F7 = the observed dose, F1..F6 = the steady-state history).
Must be run from inside a worktree of the paper commit.
    gen_seeded_idv.py <data_seed> <kappa> <gen_tacro args...>
"""
import sys, random, os, textwrap, importlib.util
import numpy as np, torch
sys.path.insert(0, ".")
seed, kappa = int(sys.argv[1]), float(sys.argv[2])
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
spec = importlib.util.spec_from_file_location("gen_tacro_paper", "gen_tacro.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)   # defines everything, main block not run

rng = np.random.RandomState(seed + 99991)
F_log = []                                   # one list of factors per simulate() call = per patient, in ID order
_orig_simulate = mod.TacrolimusPK.simulate
def simulate(self, dosing_times, time_points):
    F_log.append([])
    return _orig_simulate(self, dosing_times, time_points)
def state_update(self, state):
    A_depot, A_gut1, A_gut2, A_gut3, A_central, A_peripheral = state
    f = float(np.exp(kappa * rng.standard_normal()))
    F_log[-1].append(f)
    amt = self.dose_mg if kappa == 0.0 else self.dose_mg * f
    return A_depot + amt, A_gut1, A_gut2, A_gut3, A_central, A_peripheral
mod.TacrolimusPK.simulate = simulate
mod.TacrolimusPK.state_update = state_update

src = open("gen_tacro.py").read()
main = src.split("if __name__ == '__main__':", 1)[1]
sys.argv = ["gen_tacro.py"] + sys.argv[3:]
exec(compile(textwrap.dedent(main), "gen_tacro.py:__main__", "exec"), mod.__dict__)

exp = mod.__dict__["args"].exp
import pandas as pd
rows = [dict(ID=i + 1, **{f"F{j + 1}": f for j, f in enumerate(fs)}) for i, fs in enumerate(F_log)]
pd.DataFrame(rows).to_csv(f"./results/{exp}/idv_truth.csv", index=False)
print(f"kappa={kappa}: per-dose factors for {len(rows)} patients -> results/{exp}/idv_truth.csv")
