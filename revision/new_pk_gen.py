import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint, odeint
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import argparse
import os

# ==============================================================================
# Parameters for standard 2-Compartment Model with First-Order Absorption
# ==============================================================================

POPULATION_PARAMS = {
    'CL': 10.5,    # Clearance (L/h)
    'Vc': 45.0,    # Central Volume (L)
    'Q': 15.0,     # Inter-compartmental Clearance (L/h)
    'Vp': 60.0,    # Peripheral Volume (L)
    'ka': 1.2      # Absorption rate constant (h^-1)
}

# Inter-Patient Variability (IPV) standard deviations (Normal distribution on log-scale)
IPV_OMEGA = {
    'CL': np.sqrt(0.09),  # ~30% CV
    'Vc': np.sqrt(0.04),  # ~20% CV
    'Q':  np.sqrt(0.09), 
    'Vp': np.sqrt(0.09),
    'ka': np.sqrt(0.16)   # ~40% CV
}

RESIDUAL_ERROR_PROP_SD = 0.10 # 10% proportional error
RESIDUAL_ERROR_ADD_SD = 0.5   # Additive error (e.g., 0.5 ng/mL)

class StandardTwoCptPK(nn.Module):
    """
    A class to simulate the pharmacokinetics of a standard 2-compartment 
    model with 1st-order absorption.
    """
    def __init__(self, device=torch.device("cpu"), adjoint=False):
        super().__init__()
        
        # Standardize dose for the simulation
        dose_list = [5.0, 10.0, 15.0, 20.0]
        self.dose_mg = random.choice(dose_list)
        self.device = device
        self.odeint = odeint_adjoint if adjoint else odeint
        
        self.pop_params = POPULATION_PARAMS
        self.ipv = IPV_OMEGA
        self.individual_params = {}

    def _sample_individual_parameters(self):
        """Samples log-normally distributed individual PK parameters."""
        for p_name, tv_p in self.pop_params.items():
            # Standard random effect from Normal distribution
            eta = torch.randn(1).item() * self.ipv.get(p_name, 0.0)
            self.individual_params[p_name] = torch.tensor(tv_p, device=self.device) * torch.exp(torch.tensor(eta, device=self.device))
        return self.individual_params

    def forward(self, t, state):
        """
        Defines the system of ordinary differential equations (ODEs).
        """
        A_gut, A_central, A_peripheral = state

        ka = self.individual_params['ka']
        CL = self.individual_params['CL']
        Q = self.individual_params['Q']
        Vc = self.individual_params['Vc']
        Vp = self.individual_params['Vp']

        # Calculate micro-rate constants
        k_el = CL / Vc
        k_12 = Q / Vc  
        k_21 = Q / Vp  

        # ODEs for the 3 compartments
        dA_gut_dt = -ka * A_gut
        dA_central_dt = (ka * A_gut) - (k_el * A_central) - (k_12 * A_central) + (k_21 * A_peripheral)
        dA_peripheral_dt = (k_12 * A_central) - (k_21 * A_peripheral)

        return dA_gut_dt, dA_central_dt, dA_peripheral_dt
    
    def get_initial_state(self):
        """Returns the initial state of the system (all 3 compartments empty)."""
        t0 = torch.tensor([0.0], device=self.device)
        state = (torch.tensor([0.0], device=self.device), # A_gut
                 torch.tensor([0.0], device=self.device), # A_central
                 torch.tensor([0.0], device=self.device)) # A_peripheral
        return t0, state
    
    def state_update(self, state):
        """Applies a dose to the absorption compartment."""
        A_gut, A_central, A_peripheral = state
        A_gut = A_gut + self.dose_mg
        return A_gut, A_central, A_peripheral
     
    def simulate(self, dosing_times, time_points):
        """Simulates the drug concentration over time for a given dosing regimen."""
        self._sample_individual_parameters()
        t0, state = self.get_initial_state()
        
        if 0.0 in dosing_times:
             state = self.state_update(state)
        
        all_concentrations = [torch.tensor([[0.0]])]
        dosing_times = sorted(dosing_times)
        last_time = t0
        
        for event_t in dosing_times:
            if event_t > last_time:
                ts_interval = time_points[(time_points > last_time) & (time_points <= event_t)]
                if len(ts_interval) > 0:
                    tt = torch.cat([last_time, ts_interval])
                    solution = self.odeint(self, state, tt, atol=1e-6, rtol=1e-6)
                    
                    # Calculate concentration = Amount_central (index 1) / Vc
                    concentrations = solution[1][1:] / self.individual_params['Vc']
                    all_concentrations.append(concentrations)
                    state = tuple(s[-1] for s in solution)
            
            if event_t > 0.0:
                 state = self.state_update(state) 
            last_time = torch.tensor([event_t], device=self.device)
        
        # Integrate from the last dose time to the end
        ts_final = time_points[time_points > last_time]
        if len(ts_final) > 0:
            tt = torch.cat([last_time, ts_final])
            solution = self.odeint(self, state, tt, atol=1e-6, rtol=1e-6)
            concentrations = solution[1][1:] / self.individual_params['Vc']
            all_concentrations.append(concentrations)
            
        return torch.cat(all_concentrations)

# ==============================================================================
# Visualization Function
# ==============================================================================

def plot_cohort(df, num_to_plot=10):
    """
    Plots the concentration-time trajectories for a subset of the generated cohort.
    """
    print(f"Generating plot for the first {num_to_plot} patients...")
    
    # Filter for observation records only (MDV == 0)
    plot_df = df[df['MDV'] == 0].copy()
    
    # Convert DV to numeric since non-observation rows used '.'
    plot_df['DV'] = pd.to_numeric(plot_df['DV'], errors='coerce')
    plot_df['TIME'] = pd.to_numeric(plot_df['TIME'])
    
    # Limit to the first N unique patients
    unique_ids = plot_df['ID'].unique()
    selected_ids = unique_ids[:num_to_plot]
    plot_df = plot_df[plot_df['ID'].isin(selected_ids)]
    
    # Plot setup
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Plot lines with markers
    sns.lineplot(
        data=plot_df, 
        x='TIME', 
        y='DV', 
        hue='ID', 
        palette='tab10', 
        marker='o', 
        linewidth=2,
        alpha=0.8
    )
    
    plt.title('Simulated PK Trajectories (2-Compartment, 1st-Order Abs)', fontsize=14, fontweight='bold')
    plt.xlabel('Time (hours)', fontsize=12)
    plt.ylabel('Concentration (ng/mL)', fontsize=12)
    
    # Use Log Scale for Y-axis (Standard in PK)
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    # Move legend outside the plot
    plt.legend(title='Patient ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# ==============================================================================
# Example Usage - Generating a Virtual Cohort
# ==============================================================================

def generate_virtual_cohort(num_patients=10):
    observation_times = torch.tensor([0, 0.5, 1., 2., 4., 8., 12., 24.])
    
    print(f"Generating data for {num_patients} virtual patients...")
    all_rows = [] 
    
    for patient_id in range(1, num_patients + 1):
        pk_model = StandardTwoCptPK()
        
        # Simulate over 24 hours with a single dose at time 0
        end_time = observation_times.max().item()
        fine_grained_times = torch.arange(0.0, end_time, 0.1)
        all_sim_times = torch.unique(torch.cat([observation_times[observation_times > 0], fine_grained_times]))
        
        true_concentrations_all = pk_model.simulate(dosing_times=[0.0], time_points=all_sim_times) * 1000 # Convert to ng/mL
        
        mask = torch.isin(all_sim_times, observation_times[observation_times > 0])
        true_concentrations = true_concentrations_all[mask]

        # Add residual error
        sd_error = RESIDUAL_ERROR_ADD_SD + RESIDUAL_ERROR_PROP_SD * true_concentrations
        noise = torch.randn_like(true_concentrations)
        concentrations = torch.clamp(true_concentrations + sd_error * noise, min=0.0)

        # Build data records mapping to NONMEM / Monolix format
        all_rows.append({
            'ID': patient_id, 'TIME': 0.0, 'DV': '.', 'AMT': pk_model.dose_mg, 'MDV': 1, 'EVID': 1
        })
        
        for time, conc in zip(observation_times[observation_times > 0].tolist(), concentrations.tolist()):
            all_rows.append({
                'ID': patient_id, 'TIME': time, 'DV': conc[0], 'AMT': '.', 'MDV': 0, 'EVID': 0
            })
            
    print("Generation complete.")
    return pd.DataFrame(all_rows)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generation 2-Compartment PK Data')
    parser.add_argument('--num_patients', type=int, default=50, help="Number of virtual patients to generate")
    args = parser.parse_args()
    
    cohort_df = generate_virtual_cohort(num_patients=args.num_patients)
    cohort_df.to_csv('virtual_cohort_2cpt.csv', index=False)
    
    print(f"Successfully created standard 2-compartment virtual cohort with {args.num_patients} patients.")
    print("\n--- File Head ---")
    print(cohort_df.head(10))
    
    # --- NEW: Call Plotting function ---
    plot_cohort(cohort_df, num_to_plot=10)