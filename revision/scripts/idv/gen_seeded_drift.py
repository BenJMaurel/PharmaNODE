#!/usr/bin/env python3
"""The paper's gen_tacro.py (one visit per patient) with fixed seeds, as scripts/scen_rerun/gen_seeded.py, PLUS a
clearance change in the unobserved steady-state history (handoff 12.3): before dose number DRIFT_SWITCH (env, 0-based,
default 6 = the observed dose, i.e. the change happened at the start of the observed interval; 5 = one interval before)
the patient's clearance is CL_today * exp(delta), delta ~ N(0, s) per patient from its OWN stream
(RandomState(data_seed + 77773)); from that dose on it is CL_today. (A switch 2 intervals before barely moves the
trough: ~2 half-lives of re-equilibration, checked 28/09.)
The model / MAP-BE assume steady state with today's parameters. Every other draw is identical to gen_seeded.py with the
same seed; s = 0 reproduces it byte for byte. Per-patient delta -> results/<exp>/drift_truth.csv.
Must be run from inside a worktree of the paper commit.      gen_seeded_drift.py <data_seed> <s> <gen_tacro args...>
"""
import sys, os, random, textwrap, importlib.util
import numpy as np, torch
sys.path.insert(0, ".")
seed, s = int(sys.argv[1]), float(sys.argv[2])
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
spec = importlib.util.spec_from_file_location("gen_tacro_paper", "gen_tacro.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

rng = np.random.RandomState(seed + 77773)
SWITCH = int(os.environ.get('DRIFT_SWITCH', '6'))   # doses before SWITCH use the old clearance
log = []
_orig_simulate, _orig_update = mod.TacrolimusPK.simulate, mod.TacrolimusPK.state_update
def simulate(self, dosing_times, time_points):
    self._drift_delta = float(s * rng.standard_normal()); self._drift_n = 0
    log.append(self._drift_delta)
    return _orig_simulate(self, dosing_times, time_points)
def state_update(self, state):
    if s != 0.0:
        if self._drift_n == 0:               # first dose: parameters were just sampled = today's values
            self._cl_today = self.individual_params['CL']
            self.individual_params['CL'] = self._cl_today * float(np.exp(self._drift_delta))
        elif self._drift_n == SWITCH:
            self.individual_params['CL'] = self._cl_today
    self._drift_n += 1
    return _orig_update(self, state)
mod.TacrolimusPK.simulate = simulate
mod.TacrolimusPK.state_update = state_update

src = open("gen_tacro.py").read()
main = src.split("if __name__ == '__main__':", 1)[1]
sys.argv = ["gen_tacro.py"] + sys.argv[3:]
exec(compile(textwrap.dedent(main), "gen_tacro.py:__main__", "exec"), mod.__dict__)
exp = mod.__dict__["args"].exp
import pandas as pd
pd.DataFrame(dict(ID=np.arange(1, len(log) + 1), delta=log)).to_csv(f"./results/{exp}/drift_truth.csv", index=False)
print(f"s={s}: CL-history deltas for {len(log)} patients -> results/{exp}/drift_truth.csv")
