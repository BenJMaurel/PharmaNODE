import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint, odeint
import numpy as np
import matplotlib.pyplot as plt

import random
import pandas as pd
import time
import ot # Python Optimal Transport (POT) library
from tqdm import tqdm

# Population mean parameters (Theta values)
POPULATION_PARAMS = {
    'theta1_Ktr': 3.34,         # Base absorption rate constant (h^-1)
    'theta2_Ktr_study': 1.53,   # Multiplier for Ktr for Prograf
    'theta3_CL': 21.2,
    # Covariate effects
    'theta4_CL_HT': 3.14,      # Exponent for hematocrit effect on Vmax
    'theta5_CL_CYP': 2.00,      # Multiplier for Vmax for CYP3A5 expressers

    'Q': 79.0,                  # Apparent inter-compartmental clearance (L/h)
    'theta6_Vc': 486.0,         # Base apparent central volume (L)
    'theta7_Vc_study': 0.29,    # Multiplier for Vc for Prograf
    'Vp': 271.0,                # Apparent peripheral volume (L)
}

# Inter-Patient Variability (IPV) as standard deviation of the random effect
IPV_OMEGA = {
    'CL':  np.sqrt(0.08), # 0.283
    'Vc':  np.sqrt(0.10), # 0.316
    'Q':   np.sqrt(0.29), # 0.539
    'Vp':  np.sqrt(0.36), # 0.6
    'Ktr': np.sqrt(0.06), # 0.245
}

# Inter-Occasion Variability (IOV) as standard deviation of the random effect
IOV_KAPPA = {
    'Ktr': 0.33,
    'Vmax': 0.31,
    'Vc': 0.75,
}

# Residual error
# RESIDUAL_ERROR_PROP_SD = 0.113
# RESIDUAL_ERROR_ADD_SD = 0.5
RESIDUAL_ERROR_PROP_SD = 0.0
RESIDUAL_ERROR_ADD_SD = 0.0
# ==============================================================================
# END OF PARAMETERS
# ==============================================================================


class TacrolimusPK_F(nn.Module):
    """
    MODEL FAMILY F:
    Tacrolimus PK Model WITH Hematocrit (HT) Covariate Effect.
    """
    def __init__(self, 
                 formulation='Advagraf', 
                 hematocrit=35.0, 
                 cyp_status='non_expresser',
                 device=torch.device("cpu"), 
                 adjoint=False):
        super().__init__()
            
        self.formulation = formulation
        self.hematocrit = hematocrit
        self.cyp_status = cyp_status
        dose_list =[2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        self.dose_mg = dose_list[random.randint(0, len(dose_list) - 1)]
        self.device = device
        self.odeint = odeint_adjoint if adjoint else odeint
        
        self.pop_params = POPULATION_PARAMS
        self.ipv = IPV_OMEGA
        self.iov = IOV_KAPPA
        self.individual_params = {}

    def _sample_individual_parameters(self):
        study_factor = 1.0 if self.formulation == 'Prograf' else 0.0
        cyp_factor = 1.0 if self.cyp_status == 'expresser' else 0.0
        
        tv_ktr = self.pop_params['theta1_Ktr'] * (self.pop_params['theta2_Ktr_study'] ** study_factor)

        tv_cl = self.pop_params['theta3_CL'] * ((self.hematocrit / 35.0) ** self.pop_params['theta4_CL_HT']) * (self.pop_params['theta5_CL_CYP'] ** cyp_factor)
        tv_q = self.pop_params['Q']
        tv_vc = self.pop_params['theta6_Vc'] * (self.pop_params['theta7_Vc_study'] ** study_factor)
        tv_vp = self.pop_params['Vp']
        params = {'Ktr': tv_ktr, 'CL': tv_cl, 'Q': tv_q, 'Vc': tv_vc, 'Vp': tv_vp}
        for p_name, tv_p in params.items():
            eta = torch.randn(1).item() * self.ipv.get(p_name, 0.0)
            kappa = torch.randn(1).item() * self.iov.get(p_name, 0.0)
            self.individual_params[p_name] = torch.tensor(tv_p, device=self.device) * torch.exp(torch.tensor(eta + kappa, device=self.device))

    def forward(self, t, state):
        A_depot, A_gut1, A_gut2, A_gut3, A_central, A_peripheral = state
        Ktr = self.individual_params['Ktr']
        CL_F = self.individual_params['CL']
        Q_F = self.individual_params['Q']
        Vc_F = self.individual_params['Vc']
        Vp_F = self.individual_params['Vp']

        k_elim = CL_F / Vc_F
        k_12 = Q_F / Vc_F  # Central to Peripheral
        k_21 = Q_F / Vp_F  # Peripheral to Central

        dA_depot_dt = -Ktr * A_depot
        dA_gut1_dt = Ktr * A_depot - Ktr * A_gut1
        dA_gut2_dt = Ktr * A_gut1 - Ktr * A_gut2
        dA_gut3_dt = Ktr * A_gut2 - Ktr * A_gut3
        input_to_central = Ktr * A_gut3
        dA_central_dt = input_to_central - (k_elim * A_central) - (k_12 * A_central) + (k_21 * A_peripheral)
        dA_peripheral_dt = (k_12 * A_central) - (k_21 * A_peripheral)

        return dA_depot_dt, dA_gut1_dt, dA_gut2_dt, dA_gut3_dt, dA_central_dt, dA_peripheral_dt
    def get_initial_state(self):
        t0 = torch.tensor([0.0], device=self.device)
        state = (torch.tensor([0.0], device=self.device),
                 torch.tensor([0.0], device=self.device),
                 torch.tensor([0.0], device=self.device), 
                 torch.tensor([0.0], device=self.device),
                 torch.tensor([0.0], device=self.device), 
                 torch.tensor([0.0], device=self.device))
        return t0, state
    
    def state_update(self, state):
        A_gut1, A_gut2, A_gut3, A_gut4, A_central, A_peripheral = state
        A_gut1 = A_gut1 + self.dose_mg
        return A_gut1, A_gut2, A_gut3, A_gut4, A_central, A_peripheral
     
    def simulate(self, dosing_times, time_points):
        self._sample_individual_parameters()
        t0, state = self.get_initial_state()
        
        if 0.0 in dosing_times:
             state = self.state_update(state)
        
        all_concentrations = []
        dosing_times = sorted(list(set(dosing_times)))
        last_time = t0
        
        for i, event_t in enumerate(dosing_times):
            if event_t > last_time:
                ts_interval = time_points[(time_points > last_time) & (time_points <= event_t)]
                if len(ts_interval) > 0:
                    tt = torch.cat([last_time, ts_interval])
                    solution = self.odeint(self, state, tt, atol=1e-6, rtol=1e-6)
                    
                    # Convert Amount (solution[4]) to Concentration
                    # We skip the first point (which is last_time)
                    concentrations = (solution[4][1:] / self.individual_params['Vc']) * 1000.0 # mg/L -> ng/mL
                    all_concentrations.append(concentrations)
                    
                    state = tuple(s[-1] for s in solution)
            
            if event_t > 0.0:
                 state = self.state_update(state)
                 
            last_time = torch.tensor([event_t], device=self.device)
        
        ts_final = time_points[time_points > last_time]
        if len(ts_final) > 0:
            tt = torch.cat([last_time, ts_final])
            solution = self.odeint(self, state, tt, atol=1e-6, rtol=1e-6)
            concentrations = (solution[4][1:] / self.individual_params['Vc']) * 1000.0 # mg/L -> ng/mL
            all_concentrations.append(concentrations)
        
        # --- SCRIPT FIX ---
        # The original script returned only the last segment.
        # This returns the full simulated trajectory.
        if not all_concentrations:
             # Handle case with no time points > 0
             return torch.empty(0, device=self.device)
        return torch.cat(all_concentrations)

class TacrolimusPK_G(nn.Module):
    """
    MODEL FAMILY F:
    Tacrolimus PK Model WITHOUT Hematocrit (HT) Covariate Effect.
    """
    def __init__(self, 
                 formulation='Advagraf', 
                 hematocrit=35.0, 
                 cyp_status='non_expresser',
                 device=torch.device("cpu"), 
                 adjoint=False):
        super().__init__()
            
        self.formulation = formulation
        self.hematocrit = hematocrit
        self.cyp_status = cyp_status
        dose_list =[2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        self.dose_mg = dose_list[random.randint(0, len(dose_list) - 1)]
        self.device = device
        self.odeint = odeint_adjoint if adjoint else odeint
        
        self.pop_params = POPULATION_PARAMS
        self.ipv = IPV_OMEGA
        self.iov = IOV_KAPPA
        self.individual_params = {}

    def _sample_individual_parameters(self):
        study_factor = 1.0 if self.formulation == 'Prograf' else 0.0
        cyp_factor = 1.0 if self.cyp_status == 'expresser' else 0.0
        
        tv_ktr = self.pop_params['theta1_Ktr'] * (self.pop_params['theta2_Ktr_study'] ** study_factor)
        # ----------------------------

        
        tv_cl = self.pop_params['theta3_CL'] * (self.pop_params['theta5_CL_CYP'] ** cyp_factor)
        tv_q = self.pop_params['Q']
        tv_vc = self.pop_params['theta6_Vc'] * (self.pop_params['theta7_Vc_study'] ** study_factor)
        tv_vp = self.pop_params['Vp']
        params = {'Ktr': tv_ktr, 'CL': tv_cl, 'Q': tv_q, 'Vc': tv_vc, 'Vp': tv_vp}
        for p_name, tv_p in params.items():
            eta = torch.randn(1).item() * self.ipv.get(p_name, 0.0)
            kappa = torch.randn(1).item() * self.iov.get(p_name, 0.0)
            self.individual_params[p_name] = torch.tensor(tv_p, device=self.device) * torch.exp(torch.tensor(eta + kappa, device=self.device))

    def forward(self, t, state):
        A_depot, A_gut1, A_gut2, A_gut3, A_central, A_peripheral = state
        Ktr = self.individual_params['Ktr']
        CL_F = self.individual_params['CL']
        Q_F = self.individual_params['Q']
        Vc_F = self.individual_params['Vc']
        Vp_F = self.individual_params['Vp']

        k_elim = CL_F / Vc_F
        k_12 = Q_F / Vc_F  # Central to Peripheral
        k_21 = Q_F / Vp_F  # Peripheral to Central

        dA_depot_dt = -Ktr * A_depot
        dA_gut1_dt = Ktr * A_depot - Ktr * A_gut1
        dA_gut2_dt = Ktr * A_gut1 - Ktr * A_gut2
        dA_gut3_dt = Ktr * A_gut2 - Ktr * A_gut3
        input_to_central = Ktr * A_gut3
        dA_central_dt = input_to_central - (k_elim * A_central) - (k_12 * A_central) + (k_21 * A_peripheral)
        dA_peripheral_dt = (k_12 * A_central) - (k_21 * A_peripheral)

        return dA_depot_dt, dA_gut1_dt, dA_gut2_dt, dA_gut3_dt, dA_central_dt, dA_peripheral_dt
    def get_initial_state(self):
        t0 = torch.tensor([0.0], device=self.device)
        state = (torch.tensor([0.0], device=self.device),
                 torch.tensor([0.0], device=self.device),
                 torch.tensor([0.0], device=self.device), 
                 torch.tensor([0.0], device=self.device),
                 torch.tensor([0.0], device=self.device), 
                 torch.tensor([0.0], device=self.device))
        return t0, state
    
    def state_update(self, state):
        A_gut1, A_gut2, A_gut3, A_gut4, A_central, A_peripheral = state
        A_gut1 = A_gut1 + self.dose_mg
        return A_gut1, A_gut2, A_gut3, A_gut4, A_central, A_peripheral
     
    def simulate(self, dosing_times, time_points):
        self._sample_individual_parameters()
        t0, state = self.get_initial_state()
        
        if 0.0 in dosing_times:
             state = self.state_update(state)
        
        all_concentrations = []
        dosing_times = sorted(list(set(dosing_times)))
        last_time = t0
        
        for i, event_t in enumerate(dosing_times):
            if event_t > last_time:
                ts_interval = time_points[(time_points > last_time) & (time_points <= event_t)]
                if len(ts_interval) > 0:
                    tt = torch.cat([last_time, ts_interval])
                    solution = self.odeint(self, state, tt, atol=1e-6, rtol=1e-6)
                    
                    # Convert Amount (solution[4]) to Concentration
                    # We skip the first point (which is last_time)
                    concentrations = (solution[4][1:] / self.individual_params['Vc']) * 1000.0 # mg/L -> ng/mL
                    all_concentrations.append(concentrations)
                    
                    state = tuple(s[-1] for s in solution)
            if event_t > 0.0:
                 state = self.state_update(state)
                 
            last_time = torch.tensor([event_t], device=self.device)
        
        ts_final = time_points[time_points > last_time]
        if len(ts_final) > 0:
            tt = torch.cat([last_time, ts_final])
            solution = self.odeint(self, state, tt, atol=1e-6, rtol=1e-6)
            concentrations = (solution[4][1:] / self.individual_params['Vc']) * 1000.0 # mg/L -> ng/mL
            all_concentrations.append(concentrations)
        
        # --- SCRIPT FIX ---
        # The original script returned only the last segment.
        # This returns the full simulated trajectory.
        if not all_concentrations:
             # Handle case with no time points > 0
             return torch.empty(0, device=self.device)
        return torch.cat(all_concentrations)
    


# ==============================================================================
# WASSERSTEIN DISTANCE CALCULATION
# ==============================================================================

def simulate_curves(model_class, num_patients, dosing_times, sim_times, device):
    """
    Simulates a set of concentration-time curves for a virtual cohort.
    
    Args:
        model_class: The PK model class to use (e.g., TacrolimusPK_F).
        num_patients (int): The number of patients to simulate.
        dosing_times (list): List of times for dosing.
        sim_times (torch.Tensor): Time points for simulation output.
        device: The torch device.

    Returns:
        list: A list of NumPy arrays, where each array is a concentration curve.
    """
    print(f"Simulating {num_patients} patients from {model_class.__name__}...")
    curves = []
    
    for _ in tqdm(range(num_patients), desc=f"Simulating {model_class.__name__}"):
        # Generate random covariates for each *new* patient
        formulation = random.choice(['Prograf', 'Advagraf'])
        cyp_status = random.choice(['expresser', 'non_expresser'])
        # CRITICAL: Sample hematocrit from a distribution to see the effect
        hematocrit = random.uniform(25.0, 45.0) 
        
        model = model_class(
            formulation=formulation,
            hematocrit=hematocrit,
            cyp_status=cyp_status,
            device=device
        ).to(device)
        
        # Simulate the "true" concentration (without residual error)
        # Note: We add residual error to the *functions* themselves
        with torch.no_grad():
            true_conc = model.simulate(dosing_times, sim_times)
        
            # Apply residual error
            prop_error = torch.randn_like(true_conc) * RESIDUAL_ERROR_PROP_SD
            add_error = torch.randn_like(true_conc) * RESIDUAL_ERROR_ADD_SD
            final_conc = true_conc * (1 + prop_error) + add_error
            final_conc = torch.clamp(final_conc, min=0.0) # No negative concentrations
        
        # This makes it compatible with numpy's integration functions.
        curves.append(final_conc.cpu().numpy().squeeze())
        # ---------------
    return curves, sim_times.cpu().numpy()


def compute_cost_matrix(f_curves, g_curves, times):
    """
    Computes the (N x M) cost matrix of squared L2 distances.
    
    Args:
        f_curves (list): List of N curves from Model F.
        g_curves (list): List of M curves from Model G.
        times (np.array): The time points for integration.

    Returns:
        np.array: An (N x M) matrix C, where C[i, j] is the squared L2 distance
                  between F_i and G_j.
    """
    n = len(f_curves)
    m = len(g_curves)
    cost_matrix = np.zeros((n, m))
    print(f"Computing {n}x{m} cost matrix ({n*m} integrations)...")
    
    for i in range(n):
        for j in range(m):
            # Calculate the squared difference
            # Squeeze just in case, to ensure both are 1D
            diff_sq = (f_curves[i].squeeze() - g_curves[j].squeeze()) ** 2
            
            # Use np.trapezoid instead of the deprecated np.trapz
            cost = np.trapezoid(diff_sq, x=times)
            # ---------------
            
            cost_matrix[i, j] = cost
            
    print("Cost matrix computation complete.")
    return cost_matrix


if __name__ == '__main__':
    # --- 0. Setup Simulation Parameters ---
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Simulation settings
    # WARNING: N*M integrations will be performed. 
    # Start with small numbers (e.g., 50) to test.
    # 200x200 = 40,000 integrations, which will take time.
    N_PATIENTS_F = 10  # Number of patients from Model F
    M_PATIENTS_G = 10 # Number of patients from Model G
    nbr_ss = 4
    # Simulate 1 week (168 hours) with measurements every 0.5 hours
    DOSING_TIMES = [24*nbr_ss - 12*(nbr_ss - i) for i in range(nbr_ss+1)] # Once daily
    # SIM_TIMES = torch.arange(0., 24.0 +24/100 , 24/100, device=device) # Start just after t=
    
    observation_times = torch.tensor([0.0, 0.33, 0.67, 1., 1.5, 2.0, 3.0, 4.0, 6.0, 9.0, 12.0, 24.0], device=device) + 24*nbr_ss
    sim_times = observation_times[observation_times > 0]
    start_time = 0.0
    end_time = sim_times.max().item()
    fine_grained_times = torch.arange(start_time, end_time, 0.01)
    SIM_TIMES = torch.unique(torch.cat([sim_times, fine_grained_times]))
    # --- 1. Simulate Family F (Pile F) ---
    start_time = time.time()
    
    f_curves, times_np = simulate_curves(TacrolimusPK_F, N_PATIENTS_F, DOSING_TIMES, SIM_TIMES, device)
    mask = torch.isin(SIM_TIMES[1:], observation_times)
    f_curves = np.array(f_curves)[:,mask]

    # str(11943)
    # csv_filename = 'exp_run_all/' + str(26825) +'/new_test_data.csv'
    # reloaded_df = pd.read_csv(csv_filename)
    # reloaded_time_steps = reloaded_df['time'].values
    # reloaded_new_test_2d = reloaded_df.drop('time', axis=1).values
    # f_curves, times_np = reloaded_new_test_2d.T[:,-241:], reloaded_time_steps[-241:]
   
    print(f"Model F simulation took {time.time() - start_time:.2f} seconds.")
    
    # --- 2. Simulate Family G (Pile G) ---c
    start_time = time.time()
    g_curves, times_np = simulate_curves(TacrolimusPK_G, M_PATIENTS_G, DOSING_TIMES, SIM_TIMES, device)
    print(f"Model G simulation took {time.time() - start_time:.2f} seconds.")
    mask = torch.isin(torch.tensor(times_np[1:]), observation_times)
    g_curves = np.array(g_curves)[:,mask]
    times_np = times_np[1:]
    # times_np = np.array(times_np)[mask]
    
    # --- 3. Compute the Pairwise Cost Matrix (C_ij) ---
    # csv_filename = 'exp_run_all/' + str(26825) +'/new_test_data.csv'
    # reloaded_df = pd.read_csv(csv_filename)
    # reloaded_time_steps = reloaded_df['time'].values
    # reloaded_new_test_2d = reloaded_df.drop('time', axis=1).values
    # g_curves, times_np = reloaded_new_test_2d.T[:,-241:], reloaded_time_steps[-241:]

    start_time = time.time()
    cost_matrix = compute_cost_matrix(f_curves, g_curves, observation_times)
    print(f"Cost matrix calculation took {time.time() - start_time:.2f} seconds.")
    # --- 4. Solve the Optimal Transport Problem ---
    print("Solving optimal transport problem (Earth-Mover's Distance)...")
    start_time = time.time()
    
    # Define the probability distributions (uniform weights for our samples)
    a = np.ones(N_PATIENTS_F) / N_PATIENTS_F
    b = np.ones(M_PATIENTS_G) / M_PATIENTS_G
    
    # Calculate the squared Wasserstein distance (Earth-Mover's Distance)
    # ot.emd2 computes the minimum cost, which is W_2^2
    wasserstein_squared = ot.emd2(a, b, cost_matrix)
    
    # The W_2 distance is the square root of the minimum cost
    wasserstein_dist = np.sqrt(wasserstein_squared)
    
    print(f"Optimal transport solution took {time.time() - start_time:.2f} seconds.")
    
    # --- 5. Report Results ---
    print("\n" + "="*50)
    print("      Probabilistic Distance Calculation Results")
    print("="*50)
    print(f"  Family F (N={N_PATIENTS_F}): Model without HT effect")
    print(f"  Family G (M={M_PATIENTS_G}): Model with HT effect")
    print(f"\n  Squared L2 Cost Matrix Shape: {cost_matrix.shape}")
    print(f"  Min/Max Cost in Matrix: {cost_matrix.min():.2f} / {cost_matrix.max():.2f}")
    print(f"\n  Squared W_2 Distance (Total Cost): {wasserstein_squared:.4f}")
    print(f"  W_2 Wasserstein Distance (RMS): {wasserstein_dist:.4f}")
    print("="*50)
    
    # Optional: Plot the two families
    plt.figure(figsize=(14, 7))
    # Plot 50 curves from each family to visualize
    for i in range(min(N_PATIENTS_F, 50)):
        plt.plot(observation_times, f_curves[i], 'b-', alpha=0.1, label='Family F (No HT)' if i == 0 else "")
    for i in range(min(M_PATIENTS_G, 50)):
        plt.plot(observation_times, g_curves[i], 'r--', alpha=0.1, label='Family G (With HT)' if i == 0 else "")
    
    plt.title('Comparison of Simulated Function Families')
    plt.xlabel('Time (hours)')
    plt.ylabel('Concentration (ng/mL)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()