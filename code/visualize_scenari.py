import torch
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random

# ==============================================================================
# 1. Model Configuration (Copied & Adapted from gen_tacro.py)
# ==============================================================================

POPULATION_PARAMS = {
    'theta3_CL': 21.2,          
    'theta1_Ktr': 3.34,         
    'theta2_Ktr_study': 1.53,   
    'theta4_CL_HT': -3.14,      
    'theta5_CL_CYP': 2.00,      
    'theta6_Vc' : 486,
    'Q': 79.0,                  
    'theta7_Vc_study': 0.29,    
    'Vp': 271.0, 
    # Non-linear specific
    'Vp2' : 292.0,
    'Q2' : 75.0,
    'theta_Km': 0.01,           
    'theta_Vmax': 21.2 * 0.01 
}

IPV_OMEGA = {
    'CL':  np.sqrt(0.08), 
    'Vc':  np.sqrt(0.10), 
    'Q':   np.sqrt(0.29), 
    'Vp':  np.sqrt(0.36), 
    'Ktr': np.sqrt(0.06), 
    'Vp2': np.sqrt(0.36),
    'Q2': np.sqrt(0.29),
    'Vmax': np.sqrt(0.08),
    'Km': np.sqrt(0.10)   
}

class TacrolimusPK(nn.Module):
    def __init__(self, formulation='Advagraf', hematocrit=35.0, cyp_status='non_expresser', 
                 scenario=1, fixed_dose=None, device=torch.device("cpu")):
        super().__init__()
        self.formulation = formulation
        self.hematocrit = hematocrit
        self.cyp_status = cyp_status
        self.scenario = scenario
        self.device = device
        
        # FIX DOSE for visualization consistency (optional, but helps compare kinetics)
        if fixed_dose:
            self.dose_mg = fixed_dose
        else:
            self.dose_mg = 5.0 
            
        self.pop_params = POPULATION_PARAMS
        self.ipv = IPV_OMEGA
        self.individual_params = {}

    def _sample_individual_parameters(self):
        study_factor = 1.0 if self.formulation == 'Prograf' else 0.0
        cyp_factor = 1.0 if self.cyp_status == 'expresser' else 0.0

        tv_ktr = self.pop_params['theta1_Ktr'] * (self.pop_params['theta2_Ktr_study'] ** study_factor)
        tv_cl = self.pop_params['theta3_CL'] * ((self.hematocrit / 35.0) ** self.pop_params['theta4_CL_HT']) * (self.pop_params['theta5_CL_CYP'] ** cyp_factor)
        tv_q = self.pop_params['Q']
        tv_vc = self.pop_params['theta6_Vc'] * (self.pop_params['theta7_Vc_study'] ** study_factor)
        tv_vp = self.pop_params['Vp']

        tv_vmax = 0
        tv_km = 0
        if self.scenario == 3:
            tv_vmax = self.pop_params['theta_Vmax'] * \
                    ((self.hematocrit / 35.0) ** self.pop_params['theta4_CL_HT']) * \
                    (self.pop_params['theta5_CL_CYP'] ** cyp_factor)
            tv_km = self.pop_params['theta_Km']

        # Determine which params to sample based on scenario
        if self.scenario == 3:
            params_to_sample = {
                'Ktr': tv_ktr, 'CL': tv_cl, 'Q': tv_q, 'Vc': tv_vc, 'Vp': tv_vp, 
                'Vp2': self.pop_params['Vp2'], 'Q2': self.pop_params['Q2'],
                'Vmax': tv_vmax, 'Km': tv_km
            }
        else:
            params_to_sample = {'Ktr': tv_ktr, 'CL': tv_cl, 'Q': tv_q, 'Vc': tv_vc, 'Vp': tv_vp}

        for p_name, tv_p in params_to_sample.items():
            eta = torch.randn(1).item() * self.ipv.get(p_name, 0.0)
            self.individual_params[p_name] = torch.tensor(tv_p, device=self.device) * torch.exp(torch.tensor(eta, device=self.device))
            
        return self.individual_params

    def forward(self, t, state):
        A_depot, A_gut1, A_gut2, A_gut3, A_central, A_peripheral = state
        
        Ktr = self.individual_params['Ktr']
        Vc_F = self.individual_params['Vc']
        Vp_F = self.individual_params['Vp']
        Q_F = self.individual_params['Q']

        k_12 = Q_F / Vc_F
        k_21 = Q_F / Vp_F

        dA_depot_dt = -Ktr * A_depot
        dA_gut1_dt = Ktr * A_depot - Ktr * A_gut1
        dA_gut2_dt = Ktr * A_gut1 - Ktr * A_gut2
        dA_gut3_dt = Ktr * A_gut2 - Ktr * A_gut3
        input_to_central = Ktr * A_gut3

        if self.scenario < 3:
            CL_F = self.individual_params['CL']
            k_elim = CL_F / Vc_F
            dA_central_dt = input_to_central - (k_elim * A_central) - (k_12 * A_central) + (k_21 * A_peripheral)
        else:
            Vmax = self.individual_params['Vmax']
            Km = self.individual_params['Km']
            C_central = A_central / Vc_F
            dA_central_dt = input_to_central - (Vmax * C_central) / (Km + C_central) - (k_12 * A_central) + (k_21 * A_peripheral)

        dA_peripheral_dt = (k_12 * A_central) - (k_21 * A_peripheral)
        return dA_depot_dt, dA_gut1_dt, dA_gut2_dt, dA_gut3_dt, dA_central_dt, dA_peripheral_dt

    def simulate_profile(self, time_points):
        self._sample_individual_parameters()
        t0 = torch.tensor([0.0], device=self.device)
        # Initial state (plus dose in depot)
        state = (torch.tensor([self.dose_mg], device=self.device),
                 torch.tensor([0.0], device=self.device), torch.tensor([0.0], device=self.device),
                 torch.tensor([0.0], device=self.device), torch.tensor([0.0], device=self.device), 
                 torch.tensor([0.0], device=self.device))
        
        solution = odeint(self, state, time_points, atol=1e-6, rtol=1e-6)
        # Concentration = Amount / Volume
        concentrations = solution[4] / self.individual_params['Vc']
        return concentrations * 1000 # Convert to ng/mL

# ==============================================================================
# 2. Data Generation Function
# ==============================================================================

def generate_scenario_data(scenario_id, n_runs=50):
    sim_times = torch.linspace(0, 24, 100) # 0 to 24h profile
    data_records = []

    for i in range(n_runs):
        # 1. Randomize Covariates
        formulation = random.choice(['Prograf', 'Advagraf'])
        cyp = random.choice(['expresser', 'non_expresser'])
        
        # 2. Handle Hematocrit Logic (Crucial for Scen 2)
        # Based on gen_tacro.py lines 352-353:
        # If Scenario 2: HCT varies (25-45)
        # If Scenario 1 or 3: HCT fixed (35)
        
        raw_hct = random.uniform(25.0, 45.0)
        
        if scenario_id == 2:
            hct_used = raw_hct
        else:
            hct_used = 35.0
            
        # 3. Initialize Model
        # Note: We pass the 'scenario_id' to the class. 
        # If scenario_id == 3, the class switches to Vmax/Km ODEs.
        model = TacrolimusPK(
            formulation=formulation,
            hematocrit=hct_used,
            cyp_status=cyp,
            scenario=scenario_id,
            fixed_dose=5.0 # Fixed dose to visualize PK differences cleanly
        )
        
        # 4. Simulate
        concs = model.simulate_profile(sim_times)
        concs_np = concs.detach().numpy().flatten()
        times_np = sim_times.numpy()
        
        for t, c in zip(times_np, concs_np):
            data_records.append({
                'ID': i,
                'Time': t,
                'Conc': c,
                'Scenario': f'Scenario {scenario_id}',
                'HCT': hct_used,
                'LogConc': np.log(c) if c > 0.1 else np.nan
            })
            
    return pd.DataFrame(data_records)

# ==============================================================================
# 3. Main Execution and Plotting
# ==============================================================================

if __name__ == "__main__":
    print("Generating trajectories...")
    
    # Generate data
    df1 = generate_scenario_data(scenario_id=1, n_runs=50)
    df2 = generate_scenario_data(scenario_id=2, n_runs=50)
    df3 = generate_scenario_data(scenario_id=3, n_runs=50)
    
    # Combine for global min/max calc
    df_all = pd.concat([df1, df2, df3])
    
    # Setup Plot
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.4)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    
    # --- ROW 1: Linear Scale ---
    
    # Scenario 1: Baseline
    sns.lineplot(data=df1, x='Time', y='Conc', units='ID', estimator=None, ax=axes[0,0], 
                 color='navy', alpha=0.3, linewidth=1)
    axes[0,0].set_title("Scenario 1: Baseline\n(Fixed Hematocrit, Linear elim.)")
    axes[0,0].set_ylim(0, df_all['Conc'].max()*1.05)
    axes[0,0].set_ylabel("Concentration (ng/mL)")
    
    # Scenario 2: Covariate
    # We map Hue to Hematocrit to visualize the "Unaccounted Covariate" effect
    points = plt.scatter([], [], c=[], vmin=25, vmax=45, cmap='viridis') # Dummy for colorbar
    
    for pid in df2['ID'].unique():
        sub = df2[df2['ID']==pid]
        hct_val = sub['HCT'].iloc[0]
        color = plt.cm.viridis((hct_val - 25)/(45-25))
        axes[0,1].plot(sub['Time'], sub['Conc'], color=color, alpha=0.5, linewidth=1)
        
    axes[0,1].set_title("Scenario 2: Unaccounted Covariate\n(Variable Hematocrit)")
    axes[0,1].set_ylim(0, df_all['Conc'].max()*1.05)
    axes[0,1].set_ylabel("")
    cbar = fig.colorbar(points, ax=axes[0,1], orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Hematocrit (%)')

    # Scenario 3: Structural Misspecification
    sns.lineplot(data=df3, x='Time', y='Conc', units='ID', estimator=None, ax=axes[0,2], 
                 color='darkred', alpha=0.3, linewidth=1)
    axes[0,2].set_title("Scenario 3: Structural Misspec.\n(Saturable / Non-Linear elim.)")
    axes[0,2].set_ylim(0, df_all['Conc'].max()*1.05)
    axes[0,2].set_ylabel("")

    # --- ROW 2: Log Scale (To reveal kinetics) ---
    
    # S1 Log
    sns.lineplot(data=df1, x='Time', y='Conc', units='ID', estimator=None, ax=axes[1,0], 
                 color='navy', alpha=0.3, linewidth=1)
    axes[1,0].set_yscale('log')
    axes[1,0].set_ylim(0.1, 50) # Fixed limits for comparison
    axes[1,0].set_ylabel("Log Concentration (ng/mL)")
    axes[1,0].text(12, 10, "Linear Decay\n(Straight Line)", ha='center', color='black', fontweight='bold', bbox=dict(facecolor='white', alpha=0.7))

    # S2 Log
    for pid in df2['ID'].unique():
        sub = df2[df2['ID']==pid]
        hct_val = sub['HCT'].iloc[0]
        color = plt.cm.viridis((hct_val - 25)/(45-25))
        axes[1,1].plot(sub['Time'], sub['Conc'], color=color, alpha=0.5, linewidth=1)
    axes[1,1].set_yscale('log')
    axes[1,1].set_ylim(0.1, 50)
    axes[1,1].set_ylabel("")
    axes[1,1].text(12, 10, "High Variability\n(Driven by HCT)", ha='center', color='black', fontweight='bold', bbox=dict(facecolor='white', alpha=0.7))

    # S3 Log
    sns.lineplot(data=df3, x='Time', y='Conc', units='ID', estimator=None, ax=axes[1,2], 
                 color='darkred', alpha=0.3, linewidth=1)
    axes[1,2].set_yscale('log')
    axes[1,2].set_ylim(0.1, 50)
    axes[1,2].set_ylabel("")
    axes[1,2].text(12, 10, "Non-Linear Decay\n(Concave Curve)", ha='center', color='darkred', fontweight='bold', bbox=dict(facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.savefig("scenario_comparison.png", dpi=300)
    print("Plot saved as 'scenario_comparison.png'")
    plt.show()