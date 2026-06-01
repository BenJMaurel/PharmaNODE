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
# PLACEHOLDER PARAMETERS FOR THE MYCOPHENOLIC ACID (MPA) MODEL
# NOTE: These are not from the paper but are plausible values for demonstration.
# The papers describe the model structure but do not provide a final parameter table.
# ==============================================================================

POPULATION_PARAMS = {
    'CL': 15.0,        # Apparent Clearance (L/h)
    'V': 50.0,         # Apparent Volume of Distribution (L)
    'F1': 0.5,         # Fraction of dose absorbed via the first pathway (50%)
    'MT1': 0.75,       # Mean Transit Time for the first pathway (h)
    'NN1': 3,          # Number of transit compartments for the first pathway
    'MT2': 6.0,        # Mean Transit Time for the second pathway (h)
    'NN2': 5,          # Number of transit compartments for the second pathway
}

# Inter-Patient Variability (IPV) - placeholder values
IPV_OMEGA = {
    'CL': 0.30,
    'V': 0.25,
    'F1': 0.15,
    'MT1': 0.4,
    'MT2': 0.5,
}

# Residual error is modeled as: Y = F * (1 + ε_prop) + ε_add
RESIDUAL_ERROR_PROP_SD = 0.20  # Proportional error SD (20%) - placeholder
RESIDUAL_ERROR_ADD_SD = 0.1    # Additive error SD (0.1 mg/L) - placeholder

class MycophenolicAcidPK(nn.Module):
    """
    A class to simulate the pharmacokinetics of Mycophenolic Acid (MPA).
    
    This model is a single-compartment model with first-order elimination and
    a double gamma absorption phase, as described conceptually in the papers by
    Woillard et al..
    """
    def __init__(self, device=torch.device("cpu"), adjoint=False):
        super().__init__()
        
        dose_list = [250.0, 500.0, 750.0, 1000.0, 1500.0]
        self.dose_mg = dose_list[random.randint(0, len(dose_list) - 1)]
        self.device = device
        self.odeint = odeint_adjoint if adjoint else odeint
        
        self.pop_params = POPULATION_PARAMS
        self.ipv = IPV_OMEGA
        
        self.individual_params = {}
        
        self.nn1 = int(self.pop_params['NN1'])
        self.nn2 = int(self.pop_params['NN2'])

    def _sample_individual_parameters(self):
        params_to_sample = ['CL', 'V', 'F1', 'MT1', 'MT2']
        for p_name in params_to_sample:
            tv_p = self.pop_params[p_name]
            eta = torch.randn(1).item() * self.ipv.get(p_name, 0.0)
            ind_param = torch.tensor(tv_p, device=self.device) * torch.exp(torch.tensor(eta, device=self.device))
            if p_name == 'F1':
                ind_param = torch.sigmoid(ind_param) # Constraint F1 to be between 0 and 1
            self.individual_params[p_name] = ind_param

        self.individual_params['k_e'] = self.individual_params['CL'] / self.individual_params['V']
        self.individual_params['k_tr1'] = self.nn1 / self.individual_params['MT1']
        self.individual_params['k_tr2'] = self.nn2 / self.individual_params['MT2']

    def forward(self, t, state):
        gut1_comps = state[:self.nn1]
        gut2_comps = state[self.nn1:self.nn1 + self.nn2]
        A_central = state[-1]

        k_tr1 = self.individual_params['k_tr1']
        k_tr2 = self.individual_params['k_tr2']
        k_e = self.individual_params['k_e']
        
        d_gut1_dt = []
        d_gut1_dt.append(-k_tr1 * gut1_comps[0])
        for i in range(1, self.nn1):
            d_gut1_dt.append(k_tr1 * gut1_comps[i-1] - k_tr1 * gut1_comps[i])
        
        d_gut2_dt = []
        d_gut2_dt.append(-k_tr2 * gut2_comps[0])
        for i in range(1, self.nn2):
            d_gut2_dt.append(k_tr2 * gut2_comps[i-1] - k_tr2 * gut2_comps[i])

        input_to_central = (k_tr1 * gut1_comps[-1]) + (k_tr2 * gut2_comps[-1])
        dA_central_dt = input_to_central - (k_e * A_central)

        return (*d_gut1_dt, *d_gut2_dt, dA_central_dt)
    
    def get_initial_state(self):
        t0 = torch.tensor([0.0], device=self.device)
        num_states = self.nn1 + self.nn2 + 1
        state = tuple(torch.tensor([0.0], device=self.device) for _ in range(num_states))
        return t0, state
    
    def state_update(self, state):
        state_list = list(state)
        f1 = self.individual_params['F1']
        state_list[0] = state_list[0] + self.dose_mg * f1
        state_list[self.nn1] = state_list[self.nn1] + self.dose_mg * (1 - f1)
        return tuple(state_list)
     
    def simulate(self, dosing_times, time_points):
        """
        Simulates the drug concentration over time for a given multiple-dosing regimen.

        Args:
            dosing_times (list or np.array): A list of times at which doses are administered.
            time_points (torch.Tensor): A tensor of time points at which to evaluate the concentration.

        Returns:
            torch.Tensor: A tensor of drug concentrations at the specified time_points.
        """
        self._sample_individual_parameters()
        t0, state = self.get_initial_state()
        
        # Create an output tensor to store results
        output_concentrations = torch.zeros_like(time_points)

        # Sort unique dosing times to process chronologically
        dosing_times = sorted(list(set(dosing_times)))
        last_time = 0.0

        # Handle dose at t=0
        if 0.0 in dosing_times:
            state = self.state_update(state)
        
        # Loop through each dosing interval
        for dose_time in dosing_times:
            if dose_time <= last_time:
                continue

            # Identify time points within the current interval
            mask = (time_points > last_time) & (time_points <= dose_time)
            interval_t_points = time_points[mask]
            
            # Create integration times for the ODE solver for this interval
            integration_times = torch.cat([torch.tensor([last_time], device=self.device), interval_t_points])

            if len(integration_times) > 1:
                # Solve the ODEs for the interval
                solution = self.odeint(self, state, integration_times, atol=1e-6, rtol=1e-6)
                
                # Store the calculated concentrations, excluding the start point
                concentrations = solution[-1][1:] / self.individual_params['V']
                output_concentrations[mask] = concentrations
                
                # Update the state for the next interval
                state = tuple(s[-1] for s in solution)
            
            # Apply the dose at the end of the interval
            state = self.state_update(state)
            last_time = dose_time

        # Final integration from the last dose to the end of the simulation
        final_mask = time_points > last_time
        final_t_points = time_points[final_mask]
        if len(final_t_points) > 0:
            integration_times = torch.cat([torch.tensor([last_time], device=self.device), final_t_points])
            solution = self.odeint(self, state, integration_times, atol=1e-6, rtol=1e-6)

            concentrations = solution[-1][1:] / self.individual_params['V']

        return concentrations


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
        
        # Instantiate the model for this patient
        pk_model = MycophenolicAcidPK()

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

        # mask = torch.isin(all_sim_times, sim_times)
        # true_concentrations = true_concentrations_all[mask]
        true_concentrations = true_concentrations_all.squeeze(-1)
        # --- Structure the data for this patient ---
        patient_data = []
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
        auc = np.trapz(true_concentrations_all.squeeze(-1), all_sim_times - 144.)
        if auc < 100 or auc > 800:
            continue 
        patient_data.append({
            'ID': patient_id, 'TIME': 0.0, 'DV': '.', 'AMT': pk_model.dose_mg, 'PERI': 1, 'AUC': auc, 'mdv':1, 'ss':1
        })
        for time, conc, true_conc in zip(sim_times.tolist(), concentrations.tolist(), true_concentrations.tolist()):
            # if ii <13 and time-144 > 13:
            #     pass
            # else:
            if conc[0] == 0:
                conc[0] = true_conc[0]
            patient_data.append({
                'ID': patient_id, 'TIME': time-144, 'DV': conc[0], 'AMT': '.', 'PERI': 1, 'AUC': auc, 'mdv':0, 'ss':0
            })
            
        all_patients_df.append(pd.DataFrame(patient_data))
    print("Generation complete.")
    return pd.concat(all_patients_df, ignore_index=True)


if __name__ == '__main__':
    # Define the number of patients for the virtual cohort
    NUM_VIRTUAL_PATIENTS = 200
    
    # Generate the dataset
    cohort_df = generate_virtual_cohort(num_patients=NUM_VIRTUAL_PATIENTS)
    
    
    # Save the dataset to a CSV file
    # 1. Get a list of all unique IDs
    unique_ids = cohort_df['ID'].unique()

    # 2. Split the unique IDs into training (80%) and testing (20%) sets
    # random_state ensures the split is the same every time you run the code
    train_ids, test_ids = train_test_split(unique_ids, test_size=0.2, shuffle=False)

    # 3. Filter the original DataFrame to create train and test sets
    train_df = cohort_df[cohort_df['ID'].isin(train_ids)]
    test_df = cohort_df[cohort_df['ID'].isin(test_ids)]

    
    # 4. Save the two new DataFrames to separate .csv files
    train_df.to_csv('virtual_mmf_train.csv', index=False)
    test_df.to_csv('virtual_mmf_test.csv', index=False)
    
    print(f"\nSuccessfully created virtual cohort with {NUM_VIRTUAL_PATIENTS} patients.")
    print(f"Data saved to virtual_cohort_train")
    
    # Display the first few rows of the generated file
    print("\n--- File Head ---")
    print(cohort_df.head(15))
# ==============================================================================
# Example Simulation
# ==============================================================================
# if __name__ == '__main__':
#     # Create an instance of the model
#     pk_model = MycophenolicAcidPK()
#     nbr_patient = 100
#     # Define the time points for the simulation
#     sim_times = torch.linspace(0, 12, 100)  # Simulate over a 12-hour interval

#     # Run the simulation
#     all_concentrations = []
#     for i in range(nbr_patient):
#         concentrations = pk_model.simulate(sim_times)
#         all_concentrations.append(concentrations)
#     all_concentrations = np.concatenate(all_concentrations, axis = 1)
#     # Convert to numpy for plotting
#     sim_times_np = sim_times.detach().cpu().numpy()
#     # concentrations_np = concentrations.detach().cpu().numpy()

#     # Add some residual error for realism
#     # prop_error = np.random.normal(0, RESIDUAL_ERROR_PROP_SD, len(concentrations_np))
#     # add_error = np.random.normal(0, RESIDUAL_ERROR_ADD_SD, len(concentrations_np))
#     # observed_concentrations = concentrations_np * (1 + prop_error) + add_error
#     # observed_concentrations[observed_concentrations < 0] = 0 # Ensure no negative concentrations
#     # Plot the results
#     plt.figure(figsize=(10, 6))
#     for i in range(nbr_patient):
#         plt.plot(sim_times_np, all_concentrations[:,i], alpha = 0.5, color='blue')
#     # plt.scatter(sim_times_np, observed_concentrations, label='Observed Concentrations (with error)', color='red', alpha=0.6)
#     plt.title(f"Simulated MPA Concentration Profile (Dose: {pk_model.dose_mg} mg)")
#     plt.xlabel("Time (hours)")
#     plt.ylabel("MPA Concentration (mg/L)")
#     plt.legend()
#     plt.grid(True)
#     plt.show()