import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint, odeint
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from lib.read_tacro import auc_linuplogdown
from sklearn.model_selection import train_test_split
# ==============================================================================
# Parameters from the paper (Table 4, Final nonmixture model) [cite: 264, 265]
# ==============================================================================

# Population mean parameters (Theta values)
# ==============================================================================
# PHARMACOKINETIC MODEL PARAMETERS (MICHAELIS-MENTEN ELIMINATION)
# ==============================================================================

POPULATION_PARAMS = {
    'theta1_Ktr': 3.34,         # Base absorption rate constant (h^-1)
    'theta2_Ktr_study': 1.53,   # Multiplier for Ktr for Prograf
    
    # --- CL parameters REMOVED ---
    # 'theta3_CL': 21.2,
    # 'theta4_CL_HT': -1.14,
    # 'theta5_CL_CYP': 2.00,
    
    # --- Vmax and Km parameters ADDED ---
    # Vmax is the max elimination rate (e.g., mg/h). Calculated as ~21.2 L/h * 0.15 mg/L
    'theta_Vmax': 3.18,
    # Km is the concentration at half-max rate (e.g., mg/L). 0.15 mg/L = 150 ng/mL
    'theta_Km': 0.15,
    # Covariate effects from CL are transferred to Vmax
    'theta_Vmax_HT': -1.14,      # Exponent for hematocrit effect on Vmax
    'theta_Vmax_CYP': 2.00,      # Multiplier for Vmax for CYP3A5 expressers

    'Q': 79.0,                  # Apparent inter-compartmental clearance (L/h)
    'theta6_Vc': 486.0,         # Base apparent central volume (L)
    'theta7_Vc_study': 0.29,    # Multiplier for Vc for Prograf
    'Vp': 271.0,                # Apparent peripheral volume (L)
}

# Inter-Patient Variability (IPV) as standard deviation of the random effect
IPV_OMEGA = {
    'Ktr': 0.24,
    # 'CL': 0.28, # REMOVED
    'Vmax': 0.28, # ADDED: Variability from CL transferred to Vmax
    'Km': 0.15,   # ADDED: Assumed smaller variability for Km
    'Q': 0.54,
    'Vc': 0.31,
    'Vp': 0.60,
}

# Inter-Occasion Variability (IOV) as standard deviation of the random effect
IOV_KAPPA = {
    'Ktr': 0.33,
    # 'CL': 0.31, # REMOVED
    'Vmax': 0.31, # ADDED: Variability from CL transferred to Vmax
    'Vc': 0.75,
}
# ==============================================================================
# Residual error is modeled as: Y = F * (1 + ε_prop) + ε_add
# where F is the true concentration and ε are normally distributed errors.
RESIDUAL_ERROR_PROP_SD = 0.113  # Proportional error standard deviation (11.3%)
RESIDUAL_ERROR_ADD_SD = 0.5   # Additive error standard deviation (0.71 ng/mL)

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
        
        # This dictionary will hold the sampled parameters for a given simulation
        self.individual_params = {}

    def _sample_individual_parameters(self):
        """
        Calculates individual PK parameters for a simulation run.
        MODIFIED FOR MICHAELIS-MENTEN ELIMINATION.
        """
        # --- 1. Calculate Typical Values (TV) based on covariates ---
        study_factor = 1.0 if self.formulation == 'Prograf' else 0.0
        cyp_factor = 1.0 if self.cyp_status == 'expresser' else 0.0
        
        # --- MODIFIED SECTION ---
        # Calculate typical values for the model parameters
        tv_ktr = self.pop_params['theta1_Ktr'] * (self.pop_params['theta2_Ktr_study'] ** study_factor)
        
        # REMOVED: tv_cl calculation
        # tv_cl = self.pop_params['theta3_CL'] * ((self.hematocrit / 35.0) ** self.pop_params['theta4_CL_HT']) * (self.pop_params['theta5_CL_CYP'] ** cyp_factor)
        
        # ADDED: tv_vmax and tv_km calculations
        # The covariate effects from CL are logically transferred to Vmax
        tv_vmax = self.pop_params['theta_Vmax'] * ((self.hematocrit / 35.0) ** self.pop_params['theta_Vmax_HT']) * (self.pop_params['theta_Vmax_CYP'] ** cyp_factor)
        tv_km = self.pop_params['theta_Km'] # Assuming Km has no covariates
        
        tv_q = self.pop_params['Q']
        tv_vc = self.pop_params['theta6_Vc'] * (self.pop_params['theta7_Vc_study'] ** study_factor)
        tv_vp = self.pop_params['Vp']

        # --- 2. Apply IPV and IOV ---
        # UPDATED: The params dictionary now includes 'Vmax' and 'Km' instead of 'CL'
        params = {
            'Ktr': tv_ktr, 'Vmax': tv_vmax, 'Km': tv_km, 'Q': tv_q, 'Vc': tv_vc, 'Vp': tv_vp
        }
        
        for p_name, tv_p in params.items():
            # Sample random effects (eta for IPV, kappa for IOV)
            eta = torch.randn(1).item() * self.ipv.get(p_name, 0.0)
            kappa = torch.randn(1).item() * self.iov.get(p_name, 0.0)
            
            # Combine to get individual parameter using log-normal distribution
            self.individual_params[p_name] = torch.tensor(tv_p, device=self.device) * torch.exp(torch.tensor(eta + kappa, device=self.device))

    def forward(self, t, state):
        """
        Defines the system of ordinary differential equations (ODEs)
        WITH MICHAELIS-MENTEN ELIMINATION.
        
        State vector: (A_gut1, A_gut2, A_gut3, A_gut4, A_central, A_peripheral)
        """
        A_gut1, A_gut2, A_gut3, A_gut4, A_central, A_peripheral = state

        # Unpack individually sampled parameters
        Ktr = self.individual_params['Ktr']
        # We now use Vmax and Km instead of CL
        Vmax = self.individual_params['Vmax'] # NEW: Max elimination rate
        Km = self.individual_params['Km']     # NEW: Michaelis-Menten constant
        Q_F = self.individual_params['Q']
        Vc_F = self.individual_params['Vc']
        Vp_F = self.individual_params['Vp']

        # Calculate micro-rate constants
        k_12 = Q_F / Vc_F  # Central to Peripheral
        k_21 = Q_F / Vp_F  # Peripheral to Central
        k_elem = Q_F*((Vc_F + Vp_F)/(Vc_F*Vp_F))

        # ODEs for the 5 compartments
        dA_gut1_dt = -Ktr * A_gut1
        dA_gut2_dt = Ktr * A_gut1 - Ktr * A_gut2
        dA_gut3_dt = Ktr * A_gut2 - Ktr * A_gut3
        dA_gut4_dt = Ktr * A_gut3 - Ktr * A_gut4
        input_to_central = Ktr * A_gut4
        
        # --- MODIFIED LINE ---
        # The elimination term now follows the Michaelis-Menten equation
        # for amount, not concentration.
        elimination_mm = (Vmax * A_central) / (Km * Vc_F + A_central)
        dA_central_dt = input_to_central - elimination_mm - (k_12 * A_central) + (k_21 * A_peripheral)
        
        dA_peripheral_dt = (k_12 * A_central) - (k_21 * A_peripheral)

        return dA_gut1_dt, dA_gut2_dt, dA_gut3_dt, dA_gut4_dt, dA_central_dt, dA_peripheral_dt
    
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
        A_gut1, A_gut2, A_gut3, A_gut4, A_central, A_peripheral = state
        A_gut1 = A_gut1 + self.dose_mg
        return A_gut1, A_gut2, A_gut3, A_gut4, A_central, A_peripheral
     
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
        self._sample_individual_parameters()

        t0, state = self.get_initial_state()
        
        # Apply the first dose at t=0
        if 0.0 in dosing_times:
             state = self.state_update(state)
        
        all_concentrations = []
        
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
                    concentrations = solution[3][1:] / self.individual_params['Vc']
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
        return solution[4]/ self.individual_params['Vc']
        # return torch.cat(all_concentrations)

# ==============================================================================
# Example Usage
# ==============================================================================

def generate_virtual_cohort(num_patients=10):
    """
    Generates a CSV file for a virtual cohort of patients.

    Args:
        num_patients (int): The number of virtual patients to create.
    
    Returns:
        pandas.DataFrame: A dataframe containing the full cohort data.
    """
    all_patients_df = []
    
    # Define the time points for concentration measurement
    observation_times = torch.tensor([0, 0.33, 0.67, 1., 1.5, 2., 3., 4., 6., 9., 12., 24.])+144
    
    print(f"Generating data for {num_patients} virtual patients...")

    for i in range(1, num_patients + 1):
        patient_id = i
        
        # Randomly assign patient covariates
        formulation = random.choice(['Prograf', 'Advagraf'])
        cyp_status = random.choice(['expresser', 'non_expresser'])
        hematocrit = random.uniform(30.0, 45.0)

        # Instantiate the model for this patient
        pk_model = TacrolimusPK(
            formulation=formulation,
            hematocrit=hematocrit,
            cyp_status=cyp_status
        )

        # Simulate to get concentration values
        # We simulate for times > 0, as concentration at t=0 is 0
        sim_times = observation_times[observation_times > 0]
        start_time = sim_times.min().item()
        end_time = sim_times.max().item()
        fine_grained_times = torch.arange(start_time, end_time, 0.01)
        all_sim_times = torch.unique(torch.cat([sim_times, fine_grained_times]))

        true_concentrations_all = pk_model.simulate(
            dosing_times=[0, 24, 48, 72, 96, 120, 144], # Every 24h,
            time_points=all_sim_times
        )*1000

        mask = torch.isin(all_sim_times, sim_times)
        true_concentrations = true_concentrations_all[mask]
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
        prop_error = torch.randn_like(true_concentrations) * RESIDUAL_ERROR_PROP_SD
        
        # Generate random noise for the additive component
        add_error = torch.randn_like(true_concentrations) * RESIDUAL_ERROR_ADD_SD
        
        # Apply the error model: Y = F * (1 + ε_prop) + ε_add
        concentrations = true_concentrations * (1 + prop_error) + add_error
        
        # Ensure concentrations are not negative, which is physiologically impossible
        concentrations = torch.clamp(concentrations, min=0.0)
        # 3. Add the subsequent concentration measurements
        if ii ==12.:
            auc = np.trapz(true_concentrations_all.squeeze(-1)[all_sim_times <= 144 + 12.], all_sim_times[all_sim_times <= 144+12.] - 144.)
            ST = 1
        else:
            auc = np.trapz(true_concentrations_all.squeeze(-1), all_sim_times - 144.)
            ST = 0
        if auc < 100 or auc > 800:
            continue 
        patient_data.append({
            'ID': patient_id, 'TIME': 0.0, 'DV': '.', 'AMT': pk_model.dose_mg, 'PERI': 1, 'CYP':CYP, 'II':ii, 'DRUG':formulation, 'AUC': auc, 'mdv':1, 'ss':1, 'ST':ST, 'HT': 35
        })
        # patient_data.append({'ID': patient_id, 'TIME': 0.0, 'DV': '.', 'AMT': pk_model.dose_mg, 'ss': 1, 'II':ii})
        for time, conc, true_conc in zip(sim_times.tolist(), concentrations.tolist(), true_concentrations.tolist()):
            # if ii <13 and time-144 > 13:
            #     pass
            # else:
            if conc[0] == 0:
                conc[0] = true_conc[0]
            patient_data.append({
                'ID': patient_id, 'TIME': time-144, 'DV': conc[0], 'AMT': '.', 'PERI': 1, 'CYP':CYP, 'II':".", 'DRUG':formulation, 'AUC': auc, 'mdv':0, 'ss':'.', 'ST':ST, 'HT' : 35
            })
            
            
        all_patients_df.append(pd.DataFrame(patient_data))
    print("Generation complete.")
    return pd.concat(all_patients_df, ignore_index=True)


if __name__ == '__main__':
    # Define the number of patients for the virtual cohort
    NUM_VIRTUAL_PATIENTS = 1000
    
    # Generate the dataset
    cohort_df = generate_virtual_cohort(num_patients=NUM_VIRTUAL_PATIENTS)
    
    # Save the dataset to a CSV file
    # 1. Get a list of all unique IDs
    unique_ids = cohort_df['ID'].unique()

    # 2. Split the unique IDs into training (80%) and testing (20%) sets
    # random_state ensures the split is the same every time you run the code
    train_ids, test_ids = train_test_split(unique_ids, test_size=0.8, shuffle=False)

    # 3. Filter the original DataFrame to create train and test sets
    train_df = cohort_df[cohort_df['ID'].isin(train_ids)]
    test_df = cohort_df[cohort_df['ID'].isin(test_ids)]

    
    # 4. Save the two new DataFrames to separate .csv files
    train_df.to_csv('virtual_cohort_train.csv', index=False)
    test_df.to_csv('virtual_cohort_test.csv', index=False)
    
    print(f"\nSuccessfully created virtual cohort with {NUM_VIRTUAL_PATIENTS} patients.")
    print(f"Data saved to virtual_cohort_train")
    
    # Display the first few rows of the generated file
    print("\n--- File Head ---")
    print(cohort_df.head(15))