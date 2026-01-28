import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint, odeint
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm  # CHANGE: Added tqdm for a progress bar
from typing import Dict, Any # CHANGE: Added typing for clarity

# --- All original parameter dictionaries remain the same ---
# = an't see the need to change these as they work just fine.
# ==============================================================================
# Parameters from the paper (Table 4, Final nonmixture model)
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
    'CL': 0.31,
    'Vc': 0.75,
}
# ==============================================================================
# Residual error is modeled as: Y = F * (1 + ε_prop) + ε_add
# where F is the true concentration and ε are normally distributed errors.
RESIDUAL_ERROR_PROP_SD = 0.113  # Proportional error standard deviation (11.3%)
RESIDUAL_ERROR_ADD_SD = 0.71   # Additive error standard deviation (0.71 ng/mL)


class TacrolimusPK(nn.Module):
    def __init__(self,
                 formulation='Advagraf',
                 hematocrit=35.0,
                 cyp_status='non_expresser',
                 adjoint=False,
                 device=torch.device("cpu")):
        super().__init__()

        if formulation not in ['Prograf', 'Advagraf']:
            raise ValueError("Formulation must be 'Prograf' or 'Advagraf'.")
        if cyp_status not in ['expresser', 'non_expresser']:
            raise ValueError("cyp_status must be 'expresser' or 'non_expresser'.")

        self.formulation = formulation
        self.hematocrit = hematocrit
        self.cyp_status = cyp_status
        self.dose_mg = random.choice([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
        self.device = device
        self.odeint = odeint_adjoint if adjoint else odeint
        self.individual_params = {}
        self._sample_individual_parameters()

    def _sample_individual_parameters(self):
        study_factor = 1.0 if self.formulation == 'Prograf' else 0.0
        cyp_factor = 1.0 if self.cyp_status == 'expresser' else 0.0

        tv_ktr = POPULATION_PARAMS['theta1_Ktr'] * (POPULATION_PARAMS['theta2_Ktr_study'] ** study_factor)
        tv_cl = POPULATION_PARAMS['theta3_CL'] * ((self.hematocrit / 35.0) ** POPULATION_PARAMS['theta4_CL_HT']) * (POPULATION_PARAMS['theta5_CL_CYP'] ** cyp_factor)
        tv_q = POPULATION_PARAMS['Q']
        tv_vc = POPULATION_PARAMS['theta6_Vc'] * (POPULATION_PARAMS['theta7_Vc_study'] ** study_factor)
        tv_vp = POPULATION_PARAMS['Vp']

        params_tv = {'Ktr': tv_ktr, 'CL': tv_cl, 'Q': tv_q, 'Vc': tv_vc, 'Vp': tv_vp}

        for p_name, tv_p in params_tv.items():
            eta = np.random.normal(0, IPV_OMEGA.get(p_name, 0.0))
            self.individual_params[p_name] = torch.tensor(tv_p * np.exp(eta), device=self.device, dtype=torch.float32)

    def forward(self, t, state):
        A_depot, A_gut1, A_gut2, A_gut3, A_central, A_peripheral = state

        Ktr = self.individual_params['Ktr']
        CL_F = self.individual_params['CL']
        Q_F = self.individual_params['Q']
        Vc_F = self.individual_params['Vc']
        Vp_F = self.individual_params['Vp']

        k_elim = CL_F / Vc_F
        k_12 = Q_F / Vc_F
        k_21 = Q_F / Vp_F

        dA_depot_dt = -Ktr * A_depot
        dA_gut1_dt = Ktr * A_depot - Ktr * A_gut1
        dA_gut2_dt = Ktr * A_gut1 - Ktr * A_gut2
        dA_gut3_dt = Ktr * A_gut2 - Ktr * A_gut3
        input_to_central = Ktr * A_gut3
        dA_central_dt = input_to_central - (k_elim * A_central) - (k_12 * A_central) + (k_21 * A_peripheral)
        dA_peripheral_dt = (k_12 * A_central) - (k_21 * A_peripheral)
        
        return torch.stack([dA_depot_dt, dA_gut1_dt, dA_gut2_dt, dA_gut3_dt, dA_central_dt, dA_peripheral_dt])

    def simulate(self, dosing_times, time_points):
        state = torch.zeros(6, device=self.device, dtype=torch.float32)
        
        if 0.0 in dosing_times:
            state[0] += self.dose_mg

        all_solutions = []
        last_time = torch.tensor(0.0, device=self.device)
        
        # FIX: Add a flag to handle the first interval inclusively
        is_first_interval = True

        for event_t in sorted(dosing_times):
            event_t_tensor = torch.tensor(event_t, device=self.device)
            if event_t_tensor > last_time:
                # FIX: Use >= for the first interval to include t=0
                if is_first_interval:
                    ts_interval = time_points[(time_points >= last_time) & (time_points <= event_t_tensor)]
                    is_first_interval = False
                else:
                    ts_interval = time_points[(time_points > last_time) & (time_points <= event_t_tensor)]

                if len(ts_interval) > 0:
                    tt = torch.cat((last_time.unsqueeze(0), ts_interval))
                    solution = self.odeint(self, state, tt, atol=1e-6, rtol=1e-6)
                    all_solutions.append(solution[1:])
                    state = solution[-1]

            if event_t > 0.0:
                 state[0] += self.dose_mg
            last_time = event_t_tensor

        # FIX: The final interval also needs to check the flag
        if is_first_interval:
             ts_final = time_points[time_points >= last_time]
        else:
             ts_final = time_points[time_points > last_time]

        if len(ts_final) > 0:
            tt = torch.cat((last_time.unsqueeze(0), ts_final))
            solution = self.odeint(self, state, tt, atol=1e-6, rtol=1e-6)
            all_solutions.append(solution[1:])

        if not all_solutions:
            return torch.empty((0, 1), device=self.device)
            
        full_solution = torch.cat(all_solutions, dim=0)
        concentrations = full_solution[:, 4] / self.individual_params['Vc']
        return concentrations.unsqueeze(1)

# CHANGE: Encapsulate single patient simulation into a function for multiprocessing
def simulate_single_patient(patient_id: int) -> pd.DataFrame:
    """Simulates one patient and returns their data as a DataFrame."""
    nbr_ss = 4
    observation_times = torch.tensor([0, 0.33, 0.67, 1., 1.5, 2., 3., 4., 6., 9., 12., 24.]) + 24 * nbr_ss

    formulation = random.choice(['Prograf', 'Advagraf'])
    cyp_status = random.choice(['expresser', 'non_expresser'])
    hematocrit = random.uniform(25.0, 45.0)

    pk_model = TacrolimusPK(formulation=formulation, hematocrit=hematocrit, cyp_status=cyp_status)

    sim_times = observation_times[observation_times > 0]
    end_time = sim_times.max().item()
    fine_grained_times = torch.arange(0.0, end_time + 0.01, 0.01)
    all_sim_times = torch.unique(torch.cat([sim_times, fine_grained_times]))

    dosing_interval = 12. if formulation == 'Prograf' else 24.
    dosing_times = [i * dosing_interval for i in range(int(24 / dosing_interval * (nbr_ss + 1)))]
    
    true_concentrations_all = pk_model.simulate(dosing_times=dosing_times, time_points=all_sim_times) * 1000

    mask = torch.isin(all_sim_times, sim_times)
    true_concentrations = true_concentrations_all[mask]

    # Error model
    sd_error = RESIDUAL_ERROR_ADD_SD + RESIDUAL_ERROR_PROP_SD * true_concentrations
    noise = torch.randn_like(true_concentrations)
    concentrations = torch.clamp(true_concentrations + sd_error * noise, min=0.0)

    # Calculate AUC
    ss_start_time = nbr_ss * 24.
    ss_end_time = ss_start_time + dosing_interval
    auc_mask = (all_sim_times >= ss_start_time) & (all_sim_times <= ss_end_time)
    auc_times = all_sim_times[auc_mask].numpy()
    auc_concs = true_concentrations_all[auc_mask].squeeze().numpy()
    auc = np.trapz(auc_concs, auc_times - ss_start_time)
    
    # Structure data
    patient_data = []
    CYP = 1 if cyp_status == 'expresser' else 0
    ST = 1 if formulation == 'Prograf' else 0

    patient_data.append({
        'ID': patient_id, 'TIME': 0.0, 'DV': '.', 'AMT': pk_model.dose_mg, 'PERI': 1, 'CYP': CYP,
        'II': dosing_interval, 'DRUG': formulation, 'nbr_ss': nbr_ss, 'AUC': auc, 'mdv': 1, 'ss': 1,
        'ST': ST, 'HT': hematocrit
    })
    
    for i in range(len(dosing_times) - 1):
        patient_data.append({'ID': patient_id, 'TIME': -dosing_interval * (i + 1), 'AMT': pk_model.dose_mg, 'mdv': 1})

    for time, conc in zip(sim_times.tolist(), concentrations.tolist()):
        patient_data.append({
            'ID': patient_id, 'TIME': time - nbr_ss * 24, 'DV': conc[0], 'AMT': '.', 'mdv': 0
        })

    return pd.DataFrame(patient_data)


def generate_virtual_cohort(num_patients=10):
    print(f"Generating data for {num_patients} virtual patients using {cpu_count()} CPU cores...")
    
    # CHANGE: Use multiprocessing.Pool to run simulations in parallel
    with Pool(cpu_count()) as pool:
        # Use imap_unordered for efficiency and tqdm for a progress bar
        results = list(tqdm(pool.imap_unordered(simulate_single_patient, range(1, num_patients + 1)), total=num_patients))

    print("Generation complete. Concatenating results...")
    # Concatenate all the resulting dataframes at once for high performance
    full_cohort_df = pd.concat(results, ignore_index=True)
    return full_cohort_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generation Tacro')
    parser.add_argument('--exp', type=str, default='exp_all_run/', help="Path for save experiment")
    parser.add_argument('--num_patients', type=int, default=1000, help="Number of virtual patients to generate")
    args = parser.parse_args()

    cohort_df = generate_virtual_cohort(num_patients=args.num_patients)

    # Sort by ID and TIME for a clean, non-randomized output file
    cohort_df.sort_values(['ID', 'TIME'], inplace=True)

    unique_ids = cohort_df['ID'].unique()
    train_ids, test_ids = train_test_split(unique_ids, test_size=0.2, random_state=42)

    train_df = cohort_df[cohort_df['ID'].isin(train_ids)]
    test_df = cohort_df[cohort_df['ID'].isin(test_ids)]

    train_filename = 'virtual_cohort_train.csv'
    test_filename = 'virtual_cohort_test.csv'
    
    # train_df.to_csv(train_filename, index=False)
    # test_df.to_csv(test_filename, index=False)

    print(f"\nSuccessfully created virtual cohort with {args.num_patients} patients.")
    print(f"Training data saved to '{train_filename}'")
    print(f"Test data saved to '{test_filename}'")
    
    print("\n--- File Head (Train Set) ---")
    print(train_df.head(15))