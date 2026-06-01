import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint, odeint
import numpy as np

class GeneratedStandardPK(nn.Module):
    def __init__(self, cov_data, device=torch.device("cpu"), adjoint=False):
        super().__init__()
        self.device = device
        self.odeint = odeint_adjoint if adjoint else odeint
        self.cov_data = cov_data  # Dictionary of patient covariates (e.g., {'WT': 80.0})

        # Population parameters
        self.pop_params = {'ka': 1.2, 'CL': 10.5, 'Vc': 45.0, 'Q': 15.0, 'Vp': 60.0}
        self.ipv = {'ka': 0.4, 'CL': 0.3, 'Vc': 0.2, 'Q': 0.3, 'Vp': 0.3}
        self.individual_params = {}

    def _sample_individual_parameters(self):
        for p_name, tv_p in self.pop_params.items():
            eta = torch.randn(1).item() * self.ipv.get(p_name, 0.0)
            self.individual_params[p_name] = torch.tensor(tv_p, device=self.device) * torch.exp(torch.tensor(eta, device=self.device))
        return self.individual_params

    def forward(self, t, state):
        # Extract base individual parameters
        ka = self.individual_params.get('ka', 0.0)
        CL = self.individual_params.get('CL', 0.0)
        Vc = self.individual_params.get('Vc', 0.0)
        Q = self.individual_params.get('Q', 0.0)
        Vp = self.individual_params.get('Vp', 0.0)
        Vmax = self.individual_params.get('Vmax', 0.0)
        Km = self.individual_params.get('Km', 0.0)

        # Apply Covariates (Assumes self.cov_data exists in the class)
        CL = CL * ((self.cov_data['WT'] / 70.0) ** 0.75)

        # ODE System Formulation
        A_gut, A_central = state
        C_central = A_central / Vc

        dA_gut_dt = -ka * A_gut
        dA_central_dt = ka * A_gut - (CL / Vc) * A_central

        return dA_gut_dt, dA_central_dt

    def get_initial_state(self):
        t0 = torch.tensor([0.0], device=self.device)
        state = tuple(torch.tensor([float(s)], device=self.device) for s in "0.0, 0.0".split(','))
        return t0, state
