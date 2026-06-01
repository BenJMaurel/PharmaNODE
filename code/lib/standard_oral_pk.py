"""
Reference 2-compartment oral PK model for the new-drug tutorial.
"""

import random

import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint


class StandardOralPK(nn.Module):
    """
    First-order absorption, 2-compartment model with linear elimination.
    Optional WT covariate on CL (set via cov_data before simulate).
    """

    def __init__(self, cov_data=None, device=torch.device("cpu"), adjoint=False):
        super().__init__()
        self.cov_data = cov_data or {}
        self.device = device
        self.odeint = odeint_adjoint if adjoint else odeint

        self.pop_params = {
            "ka": 1.2,
            "CL": 10.5,
            "Vc": 45.0,
            "Q": 15.0,
            "Vp": 60.0,
        }
        self.ipv = {
            "ka": 0.4,
            "CL": 0.3,
            "Vc": 0.2,
            "Q": 0.3,
            "Vp": 0.3,
        }
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
        ka = self.individual_params["ka"]
        CL = self.individual_params["CL"]
        Vc = self.individual_params["Vc"]
        Q = self.individual_params["Q"]
        Vp = self.individual_params["Vp"]

        if "WT" in self.cov_data:
            CL = CL * ((self.cov_data["WT"] / 70.0) ** 0.75)

        A_gut, A_central, A_peri = state
        k_elim = CL / Vc
        k_12 = Q / Vc
        k_21 = Q / Vp

        dA_gut_dt = -ka * A_gut
        dA_central_dt = ka * A_gut - k_elim * A_central - k_12 * A_central + k_21 * A_peri
        dA_peri_dt = k_12 * A_central - k_21 * A_peri
        return dA_gut_dt, dA_central_dt, dA_peri_dt

    def get_initial_state(self):
        t0 = torch.tensor([0.0], device=self.device)
        z = torch.tensor([0.0], device=self.device)
        return t0, (z, z, z)

    def state_update(self, state):
        A_gut, A_central, A_peri = state
        A_gut = A_gut + self.dose_mg
        return A_gut, A_central, A_peri

    def _central_index(self):
        return 1

    def simulate(self, dosing_times, time_points):
        self._sample_individual_parameters()
        t0, state = self.get_initial_state()
        Vc = self.individual_params["Vc"]

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
                    print(concentrations)
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
