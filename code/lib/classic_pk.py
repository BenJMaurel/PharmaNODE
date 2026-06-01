"""
Generic Classic PK model (1, 2, or 3 compartments) with oral or IV absorption.
"""

import random
import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint

class ClassicPKModel(nn.Module):
    """
    Generic PK model supporting 1, 2, or 3 compartments with 
    optional first-order (oral) absorption or IV bolus.
    """
    def __init__(self, n_compartments=2, absorption="oral", pop_params=None, ipv=None, cov_data=None, device=torch.device("cpu"), adjoint=False):
        super().__init__()
        if n_compartments not in [1, 2, 3]:
            raise ValueError("n_compartments must be 1, 2, or 3.")
        if absorption not in ["oral", "iv"]:
            raise ValueError("absorption must be 'oral' or 'iv'.")
        
        self.n_compartments = n_compartments
        self.absorption = absorption
        self.cov_data = cov_data or {}
        self.device = device
        self.odeint = odeint_adjoint if adjoint else odeint

        # Default parameters if not provided
        self.pop_params = pop_params or {}
        self.ipv = ipv or {}
        
        # Add defaults if empty
        if not self.pop_params:
            if self.absorption == "oral":
                self.pop_params["ka"] = 1.2
            self.pop_params["CL"] = 10.5
            self.pop_params["Vc"] = 45.0
            if self.n_compartments >= 2:
                self.pop_params["Q"] = 15.0
                self.pop_params["Vp"] = 60.0
            if self.n_compartments >= 3:
                self.pop_params["Q2"] = 10.0
                self.pop_params["Vp2"] = 40.0

        if not self.ipv:
            for k in self.pop_params:
                self.ipv[k] = 0.3

        self.individual_params = {}
        self.dose_mg = 100.0

    def _sample_individual_parameters(self):
        for p_name, tv_p in self.pop_params.items():
            eta = torch.randn(1).item() * self.ipv.get(p_name, 0.0)
            self.individual_params[p_name] = (
                torch.tensor(tv_p, device=self.device)
                * torch.exp(torch.tensor(eta, device=self.device))
            )
        return self.individual_params

    def forward(self, t, state):
        CL = self.individual_params["CL"]
        Vc = self.individual_params["Vc"]
        
        # Allometric scaling based on weight, if available
        if "WT" in self.cov_data:
            CL = CL * ((self.cov_data["WT"] / 70.0) ** 0.75)
            Vc = Vc * (self.cov_data["WT"] / 70.0)

        k_elim = CL / Vc

        if self.absorption == "oral":
            A_gut = state[0]
            A_central = state[1]
            peris = state[2:]
            ka = self.individual_params["ka"]
            dA_gut_dt = -ka * A_gut
            dA_central_in = ka * A_gut
        else:
            A_central = state[0]
            peris = state[1:]
            dA_central_in = 0.0

        dA_central_dt = dA_central_in - k_elim * A_central
        
        dA_peris = []
        if self.n_compartments >= 2:
            Q = self.individual_params["Q"]
            Vp = self.individual_params["Vp"]
            k_12 = Q / Vc
            k_21 = Q / Vp
            A_peri1 = peris[0]
            dA_central_dt += -k_12 * A_central + k_21 * A_peri1
            dA_peris.append(k_12 * A_central - k_21 * A_peri1)
            
        if self.n_compartments >= 3:
            Q2 = self.individual_params["Q2"]
            Vp2 = self.individual_params["Vp2"]
            k_13 = Q2 / Vc
            k_31 = Q2 / Vp2
            A_peri2 = peris[1]
            dA_central_dt += -k_13 * A_central + k_31 * A_peri2
            dA_peris.append(k_13 * A_central - k_31 * A_peri2)

        if self.absorption == "oral":
            return tuple([dA_gut_dt, dA_central_dt] + dA_peris)
        else:
            return tuple([dA_central_dt] + dA_peris)

    def get_initial_state(self):
        t0 = torch.tensor([0.0], device=self.device)
        z = torch.tensor([0.0], device=self.device)
        n_states = self.n_compartments
        if self.absorption == "oral":
            n_states += 1
        return t0, tuple([z for _ in range(n_states)])

    def state_update(self, state):
        state_list = list(state)
        # Dose goes into gut if oral, else directly into central
        state_list[0] = state_list[0] + self.dose_mg
        return tuple(state_list)

    def _central_index(self):
        return 1 if self.absorption == "oral" else 0

    def simulate(self, dosing_times, time_points):
        self._sample_individual_parameters()
        t0, state = self.get_initial_state()
        
        Vc = self.individual_params["Vc"]
        if "WT" in self.cov_data:
            Vc = Vc * (self.cov_data["WT"] / 70.0)

        if 0.0 in dosing_times:
            state = self.state_update(state)

        all_concentrations = [torch.zeros(1, device=self.device)]
        dosing_times = sorted(dosing_times)
        last_time = t0

        for event_t in dosing_times:
            if event_t > last_time.item():
                mask = (time_points > last_time) & (time_points <= event_t)
                ts_interval = time_points[mask]
                if len(ts_interval) > 0:
                    tt = torch.cat([last_time, ts_interval])
                    solution = self.odeint(self, state, tt, atol=1e-6, rtol=1e-6)
                    concentrations = solution[self._central_index()][1:] / Vc
                    all_concentrations.append(concentrations.flatten())
                    state = tuple(s[-1] for s in solution)
            if event_t > 0.0:
                state = self.state_update(state)
            last_time = torch.tensor([event_t], device=self.device)

        ts_final = time_points[time_points > last_time]
        if len(ts_final) > 0:
            tt = torch.cat([last_time, ts_final])
            solution = self.odeint(self, state, tt, atol=1e-6, rtol=1e-6)
            concentrations = solution[self._central_index()][1:] / Vc
            all_concentrations.append(concentrations.flatten())
        
        return torch.cat(all_concentrations)

def random_dose_mg(dose_choices):
    return float(random.choice(dose_choices))
