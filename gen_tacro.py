import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint, odeint
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from lib.read_tacro import auc_linuplogdown
from sklearn.model_selection import train_test_split
import argparse
import torch.distributions as dist
import seaborn as sns




# ==============================================================================
# Parameters from the paper (Table 4, Final nonmixture model) [cite: 264, 265]
# ==============================================================================

# Population mean parameters (Theta values)
POPULATION_PARAMS = {
    'theta3_CL': 21.2,          # Base apparent clearance (L/h)
    'theta1_Ktr': 3.34,         # Base absorption rate constant (h^-1)
    'theta2_Ktr_study': 1.53,   # Multiplier for Ktr for Prograf
    # 'theta4_CL_HT': -1.14,      # Exponent for hematocrit effect on CL
    'theta4_CL_HT': -3.14,      # Exponent for hematocrit effect on CL
    'theta5_CL_CYP': 2.00,      # Multiplier for CL for CYP3A5 expressers
    'theta6_Vc' : 486,
    'Q': 79.0,                  # Apparent inter-compartmental clearance (L/h)
    'theta7_Vc_study': 0.29,    # Multiplier for Vc for Prograf
    'Vp': 271.0,                # Apparent peripheral volume (L)

    # --- NEW PARAMETERS FOR NON-LINEAR ELIMINATION ---
    # Assuming Vmax/Km approx equals the linear CL (21.2)
    # Km set to 20 ug/L (typical saturation point for Tacro is higher, 
    # but 20 ensures non-linearity is visible in your simulation)
    'Vp2' : 292.0,
    'Q2' : 75.0,
    'theta_Km': 0.01,           
    'theta_Vmax': 21.2 * 0.01  # Base Vmax (approx 424)
}

# Inter-Patient Variability (IPV) as standard deviation of the random effect
IPV_OMEGA = {
    'CL':  np.sqrt(0.08), # 0.283
    'Vc':  np.sqrt(0.10), # 0.316
    'Q':   np.sqrt(0.29), # 0.539
    'Vp':  np.sqrt(0.36), # 0.6
    
    'Ktr': np.sqrt(0.06), # 0.245
    # --- NEW IPV ---
    'Vp2': np.sqrt(0.36),
    'Q2': np.sqrt(0.29),
    'Vmax': np.sqrt(0.08), # Assume Vmax varies like CL
    'Km': np.sqrt(0.10)    # Assume some variability on Km
}

# Inter-Occasion Variability (IOV) as standard deviation of the random effect
IOV_KAPPA = {
    'Ktr': 0.33,
    'CL': 0.31,
    'Vc': 0.75,
}
# ==============================================================================
# Residual error is modeled as: Y = F * (1 + ε_prop) + ε_add
# where F is the true concentration and ε are normally distributed errors.
RESIDUAL_ERROR_PROP_SD = 0.113  # Proportional error standard deviation (11.3%)0.03
RESIDUAL_ERROR_ADD_SD = 0.71  # Additive error standard deviation (0.71 ng/mL)0.2

class TacrolimusPK(nn.Module):
    """
    A class to simulate the pharmacokinetics of Tacrolimus based on the 
    population model described by Woillard et al., 2011[cite: 9].
    
    This model is a two-compartment model with Erlang absorption (n=3) and
    first-order elimination. It accounts for covariates like drug formulation,
    hematocrit, and CYP3A5 genotype.
    """
    def __init__(self, 
                 formulation='Advagraf', 
                 hematocrit=35.0, 
                 cyp_status='non_expresser',
                 distribution_type = 'log_normal',
                 scenario = 1,
                 device=torch.device("cpu"), 
                 adjoint=False):
        """
        Initializes the PK model for a patient with specific characteristics.

        Args:
            formulation (str): Drug formulation, 'Prograf' or 'Advagraf'.
            hematocrit (float): Patient's hematocrit level (%).
            cyp_status (str): CYP3A5 genotype status, 'expresser' or 'non_expresser'.
            device (torch.device): The device to run the simulation on.
            adjoint (bool): Whether to use the adjoint method for odeint.
        """
        super().__init__()
        
        # Patient-specific covariates
        if formulation not in ['Prograf', 'Advagraf']:
            raise ValueError("Formulation must be 'Prograf' or 'Advagraf'.")
        if cyp_status not in ['expresser', 'non_expresser']:
            raise ValueError("cyp_status must be 'expresser' or 'non_expresser'.")
        self.distribution_type= distribution_type
        self.formulation = formulation
        self.hematocrit = hematocrit
        self.cyp_status = cyp_status
        dose_list =[2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        self.dose_mg = dose_list[random.randint(0, len(dose_list) - 1)]
        self.device = device
        self.odeint = odeint_adjoint if adjoint else odeint
        
        # Store model parameters
        self.pop_params = POPULATION_PARAMS
        self.ipv = IPV_OMEGA
        self.iov = IOV_KAPPA
        self.scenario = scenario
        # This dictionary will hold the sampled parameters for a given simulation
        self.individual_params = {}

    def _sample_individual_parameters(self, t_df=4):
        """
        Calculates individual PK parameters for a simulation run.

        This method can generate parameters based on different underlying
        distributions for the random effects to test model robustness.

        Args:
            distribution_type (str): 'log_normal' (standard) or 'log_t' (for misspecification).
            t_df (int): Degrees of freedom for the Student's t-distribution.
                        Only used if distribution_type is 'log_t'.
        """
        # --- 1. Calculate Typical Values (TV) based on covariates ---
        study_factor = 1.0 if self.formulation == 'Prograf' else 0.0
        cyp_factor = 1.0 if self.cyp_status == 'expresser' else 0.0

        tv_ktr = self.pop_params['theta1_Ktr'] * (self.pop_params['theta2_Ktr_study'] ** study_factor)
        tv_cl = self.pop_params['theta3_CL'] * ((self.hematocrit / 35.0) ** self.pop_params['theta4_CL_HT']) * (self.pop_params['theta5_CL_CYP'] ** cyp_factor)
        tv_q = self.pop_params['Q']
        tv_vc = self.pop_params['theta6_Vc'] * (self.pop_params['theta7_Vc_study'] ** study_factor)
        tv_vp = self.pop_params['Vp']

        
        # We apply the CL covariates (Hematocrit, CYP) to Vmax instead of CL
        # Because Vmax represents the enzyme capacity
        if self.scenario ==3:
            tv_vmax = self.pop_params['theta_Vmax'] * \
                    ((self.hematocrit / 35.0) ** self.pop_params['theta4_CL_HT']) * \
                    (self.pop_params['theta5_CL_CYP'] ** cyp_factor)
            
            tv_km = self.pop_params['theta_Km']

        distribution_type = self.distribution_type
        
        # --- 2. Apply IPV and IOV based on the chosen distribution ---
        params = {'Ktr': tv_ktr, 'CL': tv_cl, 'Q': tv_q, 'Vc': tv_vc, 'Vp': tv_vp}
        if self.scenario ==3:
            params = {
                'Ktr': tv_ktr,
                'CL': tv_cl,
                'Q': tv_q, 
                'Vc': tv_vc, 
                'Vp': tv_vp, 
                'Vp2': self.pop_params['Vp2'], 
                'Q2': self.pop_params['Q2'],
                'Vmax': tv_vmax,
                'Km': tv_km
            }
        if distribution_type not in ['log_normal', 'log_t']:
            raise ValueError("distribution_type must be 'log_normal' or 'log_t'")

        for p_name, tv_p in params.items():
            eta, kappa = 0.0, 0.0
            if distribution_type == 'log_normal':
                # Standard assumption: random effects are from a Normal distribution
                eta = torch.randn(1).item() * self.ipv.get(p_name, 0.0)
                # kappa = torch.randn(1).item() * self.iov.get(p_name, 0.0)

            elif distribution_type == 'log_t':
                # Misspecified "ground truth": random effects from a Student's t-distribution
                t_distribution = dist.StudentT(df=t_df)
                
                # Rescale the sample to have the same variance as the Normal equivalent
                # Variance of Student's t is df/(df-2) for df > 2
                scale_factor = np.sqrt((t_df - 2) / t_df) if t_df > 2 else 1.0
                
                # Sample eta for IPV
                if p_name in self.ipv:
                    eta = t_distribution.sample().item() * self.ipv[p_name] * scale_factor
                # Sample kappa for IOV
                # if p_name in self.iov:
                #     kappa = t_distribution.sample().item() * self.iov[p_name] * scale_factor

            # Combine to get the final individual parameter
            self.individual_params[p_name] = torch.tensor(tv_p, device=self.device) * torch.exp(torch.tensor(eta, device=self.device))
        return self.individual_params

    def forward(self, t, state):
        """
        Defines the system of ordinary differential equations (ODEs).
        
        State vector: (A_gut1, A_gut2, A_gut3, A_central, A_peripheral)
        - A_gut#: Amount of drug in the three transit absorption compartments.
        - A_central: Amount of drug in the central compartment.
        - A_peripheral: Amount of drug in the peripheral compartment.
        """
        A_depot, A_gut1, A_gut2, A_gut3, A_central, A_peripheral = state

        # Unpack individually sampled parameters
        Ktr = self.individual_params['Ktr']
        CL_F = self.individual_params['CL']
        Q_F = self.individual_params['Q']
        Vc_F = self.individual_params['Vc']
        Vp_F = self.individual_params['Vp']

        # Calculate micro-rate constants
        k_elim = CL_F / Vc_F
        k_12 = Q_F / Vc_F  # Central to Peripheral
        k_21 = Q_F / Vp_F  # Peripheral to Central

        # ODEs for the 5 compartments
        dA_depot_dt = -Ktr * A_depot
        dA_gut1_dt = Ktr * A_depot - Ktr * A_gut1
        dA_gut2_dt = Ktr * A_gut1 - Ktr * A_gut2
        dA_gut3_dt = Ktr * A_gut2 - Ktr * A_gut3
        input_to_central = Ktr * A_gut3

        if self.scenario <3:
            dA_central_dt = input_to_central - (k_elim * A_central) - (k_12 * A_central) + (k_21 * A_peripheral)
            dA_peripheral_dt = (k_12 * A_central) - (k_21 * A_peripheral)

        else:
            Vmax = self.individual_params['Vmax']
            Km = self.individual_params['Km']
            Vp2_F = self.individual_params['Vp2']
            Q2_F = self.individual_params['Q2']
            #--- NEW: Calculate Concentration & Saturable Elimination ---
            # Concentration in Central Compartment = Amount / Volume
            C_central = A_central / Vc_F
            dA_central_dt = input_to_central - (Vmax * C_central) / (Km + C_central) - (k_12 * A_central) + (k_21 * A_peripheral)
            dA_peripheral_dt = (k_12 * A_central) - (k_21 * A_peripheral)

        return dA_depot_dt, dA_gut1_dt, dA_gut2_dt, dA_gut3_dt, dA_central_dt, dA_peripheral_dt
    
    def get_initial_state(self):
        """Returns the initial state of the system (all compartments are empty)."""
        t0 = torch.tensor([0.0], device=self.device)
        # State: (A_gut1, A_gut2, A_gut3, A_central, A_peripheral)
        state = (torch.tensor([0.0], device=self.device),
                 torch.tensor([0.0], device=self.device),
                 torch.tensor([0.0], device=self.device),
                 torch.tensor([0.0], device=self.device),
                 torch.tensor([0.0], device=self.device), 
                 torch.tensor([0.0], device=self.device))
        return t0, state
    
    def state_update(self, state):
        """Applies a dose to the first absorption compartment."""
        A_depot, A_gut1, A_gut2, A_gut3, A_central, A_peripheral = state
        A_depot = A_depot + self.dose_mg
        return A_depot, A_gut1, A_gut2, A_gut3, A_central, A_peripheral
     
    def simulate(self, dosing_times, time_points):
        """
        Simulates the drug concentration over time for a given dosing regimen.

        Args:
            dose_mg (float): The amount of each dose in mg.
            dosing_times (list or np.array): A list of times at which doses are administered.
            time_points (torch.Tensor): A tensor of time points at which to evaluate the concentration.

        Returns:
            torch.Tensor: A tensor of drug concentrations at the specified time_points.
        """
        # Sample a new set of parameters for this specific patient simulation
        self._sample_individual_parameters(t_df = 3)
        t0, state = self.get_initial_state()
        
        # Apply the first dose at t=0
        if 0.0 in dosing_times:
             state = self.state_update(state)
        
        all_concentrations = [torch.tensor([[0.0]])]
        
        # Sort dosing times to process them chronologically
        dosing_times = sorted(dosing_times)
        
        last_time = t0
        
        for i, event_t in enumerate(dosing_times):
            # Integrate up to the current dosing time
            if event_t > last_time:
                ts_interval = time_points[(time_points > last_time) & (time_points <= event_t)]
                if len(ts_interval) > 0:
                    tt = torch.cat([last_time, ts_interval])
                    solution = self.odeint(self, state, tt, atol=1e-6, rtol=1e-6)
                    
                    # Calculate concentration = Amount / Volume
                    concentrations = solution[4][1:] / self.individual_params['Vc']
                    all_concentrations.append(concentrations)
                    # Update state for the next interval
                    state = tuple(s[-1] for s in solution)
            # Apply dose if it's not the first one (already applied)
            
            if event_t > 0.0:
                 state = self.state_update(state) 
            last_time = torch.tensor([event_t], device=self.device)
        
        # Integrate from the last dose time to the end of the simulation
        ts_final = time_points[time_points > last_time]
        if len(ts_final) > 0:
            tt = torch.cat([last_time, ts_final])
            solution = self.odeint(self, state, tt, atol=1e-6, rtol=1e-6)
            concentrations = solution[4][1:] / self.individual_params['Vc']
            all_concentrations.append(concentrations)
        return torch.cat(all_concentrations)
        # return torch.cat(all_concentrations)

# ==============================================================================
# Example Usage
# ==============================================================================

def compare_distributions(num_patients=5000, t_df=4):
    """
    Generates and plots the distributions of PK parameters from both
    log-normal and log-t assumptions.
    """
    print(f"Simulating {num_patients} patients for each distribution type...")
    
    # Store results
    results = []

    for i in range(num_patients):
        # Instantiate a model with typical covariates
        formulation = random.choice(['Prograf', 'Advagraf'])
        cyp_status = random.choice(['expresser', 'non_expresser'])
        model_normal = TacrolimusPK(formulation=formulation, hematocrit=35.0, distribution_type='log_normal', cyp_status=cyp_status, scenario = 1)
        model_t = TacrolimusPK(formulation=formulation, hematocrit=35.0, distribution_type='log_normal', cyp_status=cyp_status, scenario = 3)
        # 1. Sample from Log-Normal distribution
        params_ln = model_normal._sample_individual_parameters()
        for p_name, p_val in params_ln.items():
            # Apply the fix here by converting to float
            results.append({'patient_id': i, 'param': p_name, 'value': float(p_val), 'distribution': 'Log-Normal'})
            
        # 2. Sample from Log-t distribution
        params_lt = model_t._sample_individual_parameters(t_df)
        for p_name, p_val in params_lt.items():
            # Apply the fix here by converting to float
            results.append({'patient_id': i, 'param': p_name, 'value': float(p_val), 'distribution': f'Log-t (df={t_df})'})

    # Convert results to a pandas DataFrame for easy plotting
    df = pd.DataFrame(results)
    
    # You can verify the data type like this:
    # print(df.info()) 
    
    print("Simulation complete. Generating plots...")

    # --- Plotting (No changes needed here) ---
    sns.set_style("whitegrid")
    param_list = ['CL', 'Vc', 'Ktr', 'Q', 'Vp']
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten() 

    for i, p_name in enumerate(param_list):
        ax = axes[i]
        df_param = df[df['param'] == p_name]
        
        sns.kdeplot(data=df_param, x='value', hue='distribution', ax=ax, fill=True, alpha=0.1)
        
        ax.set_title(f'Distribution of {p_name}', fontsize=14)
        ax.set_xlabel('Parameter Value')
        ax.set_ylabel('Density')
        ax.legend()

    axes[5].set_visible(False)
    fig.suptitle('Comparison of Parameter Distributions', fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    # plt.show()

def compare_trajectories(num_patients=20):
    """
    Simulates and plots concentration-time trajectories for Scenario 1 (Linear)
    vs Scenario 3 (Non-Linear/Saturable) to visualize structural differences.
    
    Args:
        num_patients (int): Number of patients to simulate (keep low, e.g., 20, for clarity).
        dose_mg (float): Dose amount. MUST be high (e.g., 5.0 mg or higher) to trigger saturation.
    """
    print(f"Simulating trajectories for {num_patients} patients...")
    
    results = []
    
    # Define dense time points for smooth curves (0 to 24 hours)
    # We use a tensor for the simulation
    sim_times = torch.arange(0, 24.1, 0.1) 
    
    # We use the same seed/covariates for paired comparison
    for i in range(num_patients):
        
        # 1. Randomize Covariates (shared between both scenarios for this patient)
        formulation = random.choice(['Prograf', 'Advagraf'])
        cyp = random.choice(['expresser', 'non_expresser'])
        hct = random.uniform(25.0, 45.0)
        
        # 2. Instantiate Scenario 1 (Linear Base)
        # We assume your class logic uses 'scenario' to switch between CL and Vmax/Km
        model_linear = TacrolimusPK(
            formulation=formulation, 
            hematocrit=hct, 
            cyp_status=cyp, 
            scenario=1  # Linear
        )
        
        
        
        # 3. Instantiate Scenario 3 (Non-Linear Saturation)
        model_nonlinear = TacrolimusPK(
            formulation=formulation, 
            hematocrit=hct, 
            cyp_status=cyp, 
            scenario=3  # Non-Linear (Michaelis-Menten)
        )
        model_linear.dose_mg = model_nonlinear.dose_mg # Force a specific dose for fair comparison
        # 4. Simulate
        # Note: We assume .simulate() returns a tensor of concentrations
        # We pass dosing_times=[0] for a single dose to see the decay shape clearly
        conc_linear = model_linear.simulate(dosing_times=[0.0], time_points=sim_times)
        conc_nonlinear = model_nonlinear.simulate(dosing_times=[0.0], time_points=sim_times)
        
        # Convert to ng/mL (assuming output is mg/L, multiply by 1000)
        conc_linear = conc_linear.detach().cpu().numpy().flatten() * 1000
        conc_nonlinear = conc_nonlinear.detach().cpu().numpy().flatten() * 1000
        t_np = sim_times.detach().cpu().numpy()

        # 5. Store Data
        for t, c_lin, c_nonlin in zip(t_np, conc_linear, conc_nonlinear):
            # Store Linear
            results.append({
                'Patient': i, 
                'Time': t, 
                'Concentration': c_lin, 
                'Model': 'Scenario 1 (Linear)'
            })
            # Store Non-Linear
            results.append({
                'Patient': i, 
                'Time': t, 
                'Concentration': c_nonlin, 
                'Model': 'Scenario 3 (Saturable)'
            })

    df = pd.DataFrame(results)

    # --- Plotting ---
    print("Simulation complete. Generating comparison plots...")
    sns.set_style("whitegrid")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Standard Linear Scale
    # Good for seeing differences in Peak Concentration (Cmax)
    sns.lineplot(
        data=df, x='Time', y='Concentration', 
        hue='Model', units='Patient', estimator=None, alpha=0.5, linewidth=1, ax=axes[0]
    )
    axes[0].set_title(f'Linear Scale', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Concentration (ng/mL)')
    axes[0].set_xlabel('Time (h)')
    
    # Plot 2: Log Scale
    # CRITICAL for proving structural difference.
    # Linear model = Straight line decay.
    # Non-linear model = Curved decay.
    sns.lineplot(
        data=df, x='Time', y='Concentration', 
        hue='Model', units='Patient', estimator=None, alpha=0.5, linewidth=1, ax=axes[1]
    )
    axes[1].set_yscale('log')
    axes[1].set_title(f'Log-Linear Scale (Visualizing Saturation)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Log Concentration (ng/mL)')
    axes[1].set_xlabel('Time (h)')
    
    # Add a limit to avoid log(0) issues if concentration drops too low
    axes[1].set_ylim(bottom=1.0) 

    plt.tight_layout()
    plt.show()

# if __name__ == '__main__':
#     # --- Simulation setup ---
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # Define the time points for the simulation output
#     # Simulate for 1 week (168 hours) with measurements every 0.5 hours
#     sim_times = torch.arange(0, 168.05, 0.05, device=device)

#     # --- Define two patient profiles ---
#     patient1_advagraf_non_expresser = {
#         'formulation': 'Advagraf',  # Once-daily formulation
#         'hematocrit': 38.0,
#         'cyp_status': 'non_expresser', # Slower metabolism
#         'dosing_times': [0, 24, 48, 72, 96, 120, 144] # Every 24h
#         # 'dosing_times': [0]
#     }
    
#     patient2_advagraf_expresser = {
#         'formulation': 'Prograf', # Twice-daily formulation
#         'hematocrit': 32.0,
#         'cyp_status': 'expresser', # Faster metabolism
#         'dose_mg': 5.0, # Higher dose needed
#         'dosing_times': [0, 24, 48, 72, 96, 120, 144] # Every 24h
#         # 'dosing_times': [0]
#     }
    
#     # Instantiate models for each patient
#     pk_model1 = TacrolimusPK(**{k:v for k,v in patient1_advagraf_non_expresser.items() if k not in ['dose_mg', 'dosing_times']}).to(device)
#     pk_model2 = TacrolimusPK(**{k:v for k,v in patient2_advagraf_expresser.items() if k not in ['dose_mg', 'dosing_times']}).to(device)

#     # --- Run simulations for 5 different virtual patients from each profile ---
#     plt.figure(figsize=(14, 7))
    
#     # Advagraf Patient
#     for i in range(80):
#         traj1 = pk_model1.simulate(
#             dosing_times=patient1_advagraf_non_expresser['dosing_times'],
#             time_points=sim_times
#         )
#         plt.plot(sim_times.cpu().numpy()[:len(traj1)], traj1.cpu().detach().numpy(), 'b-', alpha=0.6, label='Advagraf (non-expresser)' if i == 0 else "")

#     # Prograf Patient
#     for i in range(20):
#         traj2 = pk_model2.simulate(
#             dosing_times=patient2_advagraf_expresser['dosing_times'],
#             time_points=sim_times
#         )
#         plt.plot(sim_times.cpu().numpy()[:len(traj2):], traj2.cpu().detach().numpy(), 'r--', alpha=0.6, label='Advagraf (expresser)' if i == 0 else "")


#     # --- Plotting ---
#     plt.title('Simulated Tacrolimus Concentration Profiles')
#     plt.xlabel('Time (hours)')
#     plt.ylabel('Concentration (ng/mL or µg/L)')
#     plt.grid(True, which='both', linestyle='--', linewidth=0.5)
#     plt.legend()
#     plt.show()

def generate_virtual_cohort(num_patients=10, scenario = 1):
    """
    Generates a CSV file for a virtual cohort of patients.

    Args:
        num_patients (int): The number of virtual patients to create.
    
    Returns:
        pandas.DataFrame: A dataframe containing the full cohort data.
    """
    all_patients_df = []
    nbr_ss = 6
    # Define the time points for concentration measurement
    observation_times = torch.tensor([0, 0.33, 0.67, 1., 1.5, 2., 3., 4., 6., 9., 12., 24.])+24*nbr_ss
    
    print(f"Generating data for {num_patients} virtual patients...")
    # --- NEW: Setup for plotting ---
    # plt.figure(figsize=(12, 7))
    # ax = plt.gca()
    plotted_patient_count = 0
    # We use a while loop to ensure we generate the requested number of *valid* patients
    generated_patients = 0
    patient_id_counter = 1
    while generated_patients < num_patients:
        
        # Randomly assign patient covariates
        formulation = random.choice(['Prograf', 'Advagraf'])
        cyp_status = random.choice(['expresser', 'non_expresser'])
        hematocrit = random.uniform(25.0, 45.0)
        if scenario != 2:
            hematocrit = 35.
        # Instantiate the model for this patient
        pk_model = TacrolimusPK(
            formulation=formulation,
            hematocrit=hematocrit,
            distribution_type='log_normal',
            cyp_status=cyp_status,
            scenario = scenario
        )
        
        
        # Simulate to get concentration values
        sim_times = observation_times[observation_times > 0]
        # start_time = sim_times.min().item()
        start_time = 0.0
        end_time = sim_times.max().item()
        fine_grained_times = torch.arange(start_time, end_time, 0.01)
        all_sim_times = torch.unique(torch.cat([sim_times, fine_grained_times]))
        if formulation == 'Advagraf':
            true_concentrations_all = pk_model.simulate(
                dosing_times=[i*24 for i in range(nbr_ss+1)], # Every 24h,
                time_points=all_sim_times
            )*1000
        else:
            true_concentrations_all = pk_model.simulate(
                dosing_times=[24*nbr_ss - 12*(nbr_ss - i) for i in range(nbr_ss+1)], # Every 12h,
                time_points=all_sim_times
            )*1000
        mask = torch.isin(all_sim_times, sim_times)
        true_concentrations = true_concentrations_all[mask]
        CL_F = pk_model.individual_params['CL'].item()
        Q_F = pk_model.individual_params['Q'].item()
        Vc_F = pk_model.individual_params['Vc'].item()
        Vp_F = pk_model.individual_params['Vp'].item()

        k_elim = CL_F / Vc_F
        k_12 = Q_F / Vc_F 
        k_21 = Q_F / Vp_F
        # --- Structure the data for this patient ---
        patient_data = []
        if formulation == 'Prograf':    
            ii = 12.
        else:
            ii = 24.

        if cyp_status == 'expresser':
            CYP = 1
        else:
            CYP = 0
        # 1. Add the dosing line
        

        # # 2. Add the observation at time 0 (pre-dose concentration is 0)
        # patient_data.append({
        #     'ID': patient_id, 'TIME': 0.0, 'DV': 0.0, 'AMT': 0.0, 'PERI': 1
        # })
        # Generate random noise for the proportional component
        # prop_error = torch.randn_like(true_concentrations) * RESIDUAL_ERROR_PROP_SD
        
        # # Generate random noise for the additive component
        # add_error = torch.randn_like(true_concentrations) * RESIDUAL_ERROR_ADD_SD
        sd_error = RESIDUAL_ERROR_ADD_SD + RESIDUAL_ERROR_PROP_SD * true_concentrations

        # 2. Generate a single vector of random noise from a standard normal distribution
        # This corresponds to the 'e' part of the formula
        noise = torch.randn_like(true_concentrations)

        # 3. Calculate the final observed values (DV)
        concentrations = true_concentrations + sd_error * noise
        # Apply the error model: Y = F * (1 + ε_prop) + ε_add
        # concentrations = true_concentrations * (1 + prop_error) + add_error
        
        # Ensure concentrations are not negative, which is physiologically impossible
        concentrations = torch.clamp(concentrations, min=0.0)
        # 3. Add the subsequent concentration measurements
        if ii ==12.:
            auc = np.trapezoid(true_concentrations_all.squeeze(-1)[(all_sim_times >= nbr_ss*24) & (all_sim_times <= (nbr_ss+1)*24 - 12)], all_sim_times[(all_sim_times >= (nbr_ss)*24) & (all_sim_times <= (nbr_ss+1)*24 - 12)] - (nbr_ss+1)*24.)
            ST = 1
        else:
            auc = np.trapezoid(true_concentrations_all.squeeze(-1)[all_sim_times >= (nbr_ss)*24], all_sim_times[all_sim_times >= nbr_ss*24] - nbr_ss*24.)
            ST = 0
        # if auc < 100 or auc > 800:
        #     continue
        # --- This is a valid patient, so we process and save them ---
        generated_patients += 1
        patient_id = generated_patients

        patient_data.append({
            'ID': patient_id, 'TIME': 0.0, 'DV': '.', 'AMT': pk_model.dose_mg, 'PERI': 1, 'CYP':CYP, 'II':ii, 'DRUG':formulation, 'nbr_ss': nbr_ss,'AUC': auc, 'mdv':1, 'ss':1, 'ST':ST, 'HT': hematocrit, 'K_ELIM': k_elim, 'K_12': k_12, 'K_21': k_21
        })
        for i in range(nbr_ss):
            patient_data.append({
            'ID': patient_id, 'TIME': -ii*(i+1), 'DV': '.', 'AMT': pk_model.dose_mg, 'PERI': 1, 'CYP':CYP, 'II':ii, 'DRUG':formulation, 'AUC': auc, 'mdv':1, 'ss':1, 'ST':ST, 'HT': hematocrit, 'K_ELIM': k_elim, 'K_12': k_12, 'K_21': k_21
        })
        for time, conc, true_conc in zip(sim_times.tolist(), concentrations.tolist(), true_concentrations.tolist()):
            # if ii <13 and time-144 > 13:
            #     pass
            # else:
            if conc[0] == 0:
                conc[0] = true_conc[0]
            patient_data.append({
                'ID': patient_id, 'TIME': time-nbr_ss*24, 'DV': conc[0], 'AMT': '.', 'PERI': 1, 'CYP':CYP, 'II':'.','nbr_ss': nbr_ss, 'DRUG':formulation, 'AUC': auc, 'mdv':0, 'ss':'.', 'ST':ST, 'HT' : hematocrit, 'K_ELIM': k_elim, 'K_12': k_12, 'K_21': k_21
            })
        # patient_data = pd.DataFrame(patient_data)
        # # plt.plot(patient_data['TIME'], patient_data['DV'])
        # # plt.show()
        patient_df = pd.DataFrame(patient_data)
        all_patients_df.append(pd.DataFrame(patient_data))
        # if plotted_patient_count < 20:
        #     obs_df = patient_df[patient_df['mdv'] == 0].copy()
        #     obs_df['DV'] = pd.to_numeric(obs_df['DV'])
        #     color = 'blue' if formulation == 'Advagraf' else 'red'
        #     linestyle = '-' if cyp_status == 'non_expresser' else '--'
        #     ax.plot(obs_df['TIME'], obs_df['DV'], color=color, linestyle=linestyle, alpha=0.6,
        #             label=f'{formulation} ({cyp_status})' if plotted_patient_count < 2 else "")
        #     plotted_patient_count += 1
    print("Generation complete.")
    # --- NEW: Finalize and show the plot ---
    # ax.set_title(f'Simulated Concentration Profiles (First {plotted_patient_count} Patients)')
    # ax.set_xlabel('Time after dose at steady state (hours)')
    # ax.set_ylabel('Concentration (ng/mL)')
    # ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    # ax.legend()
    # plt.show()

    return pd.concat(all_patients_df, ignore_index=True)


if __name__ == '__main__':
    # Define the number of patients for the virtual cohort
    parser = argparse.ArgumentParser('Generation Tacro')
    parser.add_argument('--exp', type=str, default='exp_all_run/', help="Path for save experiment")
    parser.add_argument('--num_patients', type=int, default=1000, help="Number of virtual patients to generate")
    parser.add_argument('--first_at', type=int, default=1, help="Do you want to generate test set also?")
    parser.add_argument('--scenario', type=int, default=3, help="Type of scenario you want (cf paper)")
    args = parser.parse_args()
    NUM_VIRTUAL_PATIENTS = args.num_patients
    
    # Generate the dataset
    cohort_df = generate_virtual_cohort(num_patients=NUM_VIRTUAL_PATIENTS, scenario = args.scenario)
    
    
    # Save the dataset to a CSV file
    # 1. Get a list of all unique IDs
    unique_ids = cohort_df['ID'].unique()

    # 2. Split the unique IDs into training (80%) and testing (20%) sets
    # random_state ensures the split is the same every time you run the code
    if args.first_at == 1:
        train_ids, test_ids = train_test_split(unique_ids, test_size=0.2, shuffle=False)
    elif args.first_at == 2:
        _, test_ids = train_test_split(unique_ids, test_size=0.8, shuffle=False)
    else:
        # train_ids, test_ids = train_test_split(unique_ids, test_size=0.0, shuffle=False)
        train_ids = unique_ids

    # 3. Filter the original DataFrame to create train and test sets
    if args.first_at <= 1:
        train_df = cohort_df[cohort_df['ID'].isin(train_ids)]
        train_df.to_csv('virtual_cohort_train.csv', index=False)
    if args.first_at >= 1:
        test_df = cohort_df[cohort_df['ID'].isin(test_ids)]
        test_df.to_csv('virtual_cohort_test.csv', index=False)
    # 4. Save the two new DataFrames to separate .csv files
    try:
        if args.first_at <= 1:
            train_df.to_csv(f'exp_run_all/{args.exp}/virtual_cohort_train.csv', index=False)
        if args.first_at >= 1:
            test_df.to_csv(f'exp_run_all/{args.exp}/virtual_cohort_test.csv', index=False)
    except:
        pass
    print(f"\nSuccessfully created virtual cohort with {NUM_VIRTUAL_PATIENTS} patients.")
    print(f"Data saved to virtual_cohort_train")
    
    # Display the first few rows of the generated file
    print("\n--- File Head ---")
    print(cohort_df.head(15))