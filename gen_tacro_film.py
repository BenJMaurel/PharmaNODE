import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint, odeint
import numpy as np
import random
import pandas as pd
from lib.read_tacro import auc_linuplogdown
from sklearn.model_selection import train_test_split
import argparse
import torch.distributions as dist
import os

# Population params and IPV/IOV dictionaries (same as original)
POPULATION_PARAMS = {
    'theta3_CL': 21.2, 'theta1_Ktr': 3.34, 'theta2_Ktr_study': 1.53,
    'theta4_CL_HT': -3.14, 'theta5_CL_CYP': 2.00, 'theta6_Vc' : 486,
    'Q': 79.0, 'theta7_Vc_study': 0.29, 'Vp': 271.0,
    'Vp2' : 292.0, 'Q2' : 75.0, 'theta_Km': 0.01, 'theta_Vmax': 21.2 * 0.01
}

IPV_OMEGA = {
    'CL':  np.sqrt(0.08), 'Vc':  np.sqrt(0.10), 'Q':   np.sqrt(0.29), 'Vp':  np.sqrt(0.36),
    'Ktr': np.sqrt(0.06), 'Vp2': np.sqrt(0.36), 'Q2': np.sqrt(0.29),
    'Vmax': np.sqrt(0.08), 'Km': np.sqrt(0.10)
}

IOV_KAPPA = {'Ktr': 0.33, 'CL': 0.31, 'Vc': 0.75}

RESIDUAL_ERROR_PROP_SD = 0.113
RESIDUAL_ERROR_ADD_SD = 0.71

# RESIDUAL_ERROR_PROP_SD = 0.01
# RESIDUAL_ERROR_ADD_SD = 0.01

class TacrolimusPK(nn.Module):
    def __init__(self, formulation='Advagraf', hematocrit=35.0, cyp_status='non_expresser',
                 distribution_type = 'log_normal', scenario = 1, device=torch.device("cpu"), adjoint=False):
        super().__init__()
        self.distribution_type= distribution_type
        self.formulation = formulation
        self.hematocrit = hematocrit
        self.cyp_status = cyp_status
        self.dose_mg = 2.0 # Default, will be overwritten
        self.device = device
        self.odeint = odeint_adjoint if adjoint else odeint
        
        self.pop_params = POPULATION_PARAMS
        self.ipv = IPV_OMEGA
        self.iov = IOV_KAPPA
        self.scenario = scenario
        self.individual_params = {}

    def _sample_individual_parameters(self, t_df=4):
        study_factor = 1.0 if self.formulation == 'Prograf' else 0.0
        cyp_factor = 1.0 if self.cyp_status == 'expresser' else 0.0

        tv_ktr = self.pop_params['theta1_Ktr'] * (self.pop_params['theta2_Ktr_study'] ** study_factor)
        tv_cl = self.pop_params['theta3_CL'] * ((self.hematocrit / 35.0) ** self.pop_params['theta4_CL_HT']) * (self.pop_params['theta5_CL_CYP'] ** cyp_factor)
        tv_q = self.pop_params['Q']
        tv_vc = self.pop_params['theta6_Vc'] * (self.pop_params['theta7_Vc_study'] ** study_factor)
        tv_vp = self.pop_params['Vp']

        if self.scenario ==3:
            tv_vmax = self.pop_params['theta_Vmax'] * \
                    ((self.hematocrit / 35.0) ** self.pop_params['theta4_CL_HT']) * \
                    (self.pop_params['theta5_CL_CYP'] ** cyp_factor)
            tv_km = self.pop_params['theta_Km']

        distribution_type = self.distribution_type
        
        params = {'Ktr': tv_ktr, 'CL': tv_cl, 'Q': tv_q, 'Vc': tv_vc, 'Vp': tv_vp}
        if self.scenario ==3:
            params = {'Ktr': tv_ktr, 'CL': tv_cl, 'Q': tv_q, 'Vc': tv_vc, 'Vp': tv_vp, 
                      'Vp2': self.pop_params['Vp2'], 'Q2': self.pop_params['Q2'],
                      'Vmax': tv_vmax, 'Km': tv_km}

        for p_name, tv_p in params.items():
            eta, kappa = 0.0, 0.0
            if distribution_type == 'log_normal':
                eta = torch.randn(1).item() * self.ipv.get(p_name, 0.0)
            elif distribution_type == 'log_t':
                t_distribution = dist.StudentT(df=t_df)
                scale_factor = np.sqrt((t_df - 2) / t_df) if t_df > 2 else 1.0
                if p_name in self.ipv:
                    eta = t_distribution.sample().item() * self.ipv[p_name] * scale_factor
            self.individual_params[p_name] = torch.tensor(tv_p, device=self.device) * torch.exp(torch.tensor(eta, device=self.device))
        return self.individual_params

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

        if self.scenario <3:
            dA_central_dt = input_to_central - (k_elim * A_central) - (k_12 * A_central) + (k_21 * A_peripheral)
            dA_peripheral_dt = (k_12 * A_central) - (k_21 * A_peripheral)
        else:
            Vmax = self.individual_params['Vmax']
            Km = self.individual_params['Km']
            C_central = A_central / Vc_F
            dA_central_dt = input_to_central - (Vmax * C_central) / (Km + C_central) - (k_12 * A_central) + (k_21 * A_peripheral)
            dA_peripheral_dt = (k_12 * A_central) - (k_21 * A_peripheral)

        return dA_depot_dt, dA_gut1_dt, dA_gut2_dt, dA_gut3_dt, dA_central_dt, dA_peripheral_dt
    
    def get_initial_state(self):
        t0 = torch.tensor([0.0], device=self.device)
        state = tuple([torch.tensor([0.0], device=self.device) for _ in range(6)])
        return t0, state
    
    def state_update(self, state):
        A_depot, A_gut1, A_gut2, A_gut3, A_central, A_peripheral = state
        A_depot = A_depot + self.dose_mg
        return A_depot, A_gut1, A_gut2, A_gut3, A_central, A_peripheral
     
    def simulate(self, dosing_times, time_points, resample=True):
        if resample or not self.individual_params:
            self._sample_individual_parameters(t_df = 3)
        t0, state = self.get_initial_state()
        
        if 0.0 in dosing_times:
             state = self.state_update(state)
        
        all_concentrations = [torch.tensor([[0.0]])]
        dosing_times = sorted(dosing_times)
        last_time = t0
        
        for i, event_t in enumerate(dosing_times):
            if event_t > last_time:
                ts_interval = time_points[(time_points > last_time) & (time_points <= event_t)]
                if len(ts_interval) > 0:
                    tt = torch.cat([last_time, ts_interval])
                    solution = self.odeint(self, state, tt, atol=1e-6, rtol=1e-6)
                    concentrations = solution[4][1:] / self.individual_params['Vc']
                    all_concentrations.append(concentrations)
                    state = tuple(s[-1] for s in solution)
            if event_t > 0.0:
                 state = self.state_update(state) 
            last_time = torch.tensor([event_t], device=self.device)
        
        ts_final = time_points[time_points > last_time]
        if len(ts_final) > 0:
            tt = torch.cat([last_time, ts_final])
            solution = self.odeint(self, state, tt, atol=1e-6, rtol=1e-6)
            concentrations = solution[4][1:] / self.individual_params['Vc']
            all_concentrations.append(concentrations)
        return torch.cat(all_concentrations)

def generate_patient_visit(pk_model, visit_id, patient_id, nbr_ss, observation_times, formulation, hematocrit, cyp_status):
    sim_times = observation_times[observation_times > 0]
    all_sim_times = torch.unique(torch.cat([sim_times, torch.arange(0.0, sim_times.max().item(), 0.01)]))
    
    if formulation == 'Advagraf':
        true_concentrations_all = pk_model.simulate([i*24 for i in range(nbr_ss+1)], all_sim_times, resample=False)*1000
        ii = 24.
    else:
        true_concentrations_all = pk_model.simulate([24*nbr_ss - 12*(nbr_ss - i) for i in range(nbr_ss+1)], all_sim_times, resample=False)*1000
        ii = 12.
        
    mask = torch.isin(all_sim_times, sim_times)
    true_concentrations = true_concentrations_all[mask]

    sd_error = RESIDUAL_ERROR_ADD_SD + RESIDUAL_ERROR_PROP_SD * true_concentrations
    noise = torch.randn_like(true_concentrations)
    concentrations = torch.clamp(true_concentrations + sd_error * noise, min=0.0)
    CL_F = pk_model.individual_params['CL'].item()
    Q_F = pk_model.individual_params['Q'].item()
    Vc_F = pk_model.individual_params['Vc'].item()
    Vp_F = pk_model.individual_params['Vp'].item()

    k_elim = CL_F / Vc_F
    k_12 = Q_F / Vc_F 
    k_21 = Q_F / Vp_F
    if ii == 12.:
        auc = np.trapz(true_concentrations_all.squeeze(-1)[(all_sim_times >= nbr_ss*24) & (all_sim_times <= (nbr_ss+1)*24 - 12)], all_sim_times[(all_sim_times >= (nbr_ss)*24) & (all_sim_times <= (nbr_ss+1)*24 - 12)] - (nbr_ss+1)*24.)
        ST = 1
    else:
        auc = np.trapz(true_concentrations_all.squeeze(-1)[all_sim_times >= (nbr_ss)*24], all_sim_times[all_sim_times >= nbr_ss*24] - nbr_ss*24.)
        ST = 0

    CYP = 1 if cyp_status == 'expresser' else 0

    patient_data = []
    patient_data.append({'ID': patient_id, 'VISIT': visit_id, 'TIME': 0.0, 'DV': '.', 'AMT': pk_model.dose_mg, 'PERI': 1, 'CYP':CYP, 'II':ii, 'DRUG':formulation, 'nbr_ss': nbr_ss,'AUC': auc, 'mdv':1, 'ss':1, 'ST':ST, 'HT': hematocrit, 'K_ELIM': k_elim, 'K_12': k_12, 'K_21': k_21})
    for i in range(nbr_ss):
        patient_data.append({'ID': patient_id, 'VISIT': visit_id, 'TIME': -ii*(i+1), 'DV': '.', 'AMT': pk_model.dose_mg, 'PERI': 1, 'CYP':CYP, 'II':ii, 'DRUG':formulation, 'AUC': auc, 'mdv':1, 'ss':1, 'ST':ST, 'HT': hematocrit, 'K_ELIM': k_elim, 'K_12': k_12, 'K_21': k_21})
    for time, conc, true_conc in zip(sim_times.tolist(), concentrations.tolist(), true_concentrations.tolist()):
        if conc[0] == 0: conc[0] = true_conc[0]
        patient_data.append({'ID': patient_id, 'VISIT': visit_id, 'TIME': time-nbr_ss*24, 'DV': conc[0], 'AMT': '.', 'PERI': 1, 'CYP':CYP, 'II':'.','nbr_ss': nbr_ss, 'DRUG':formulation, 'AUC': auc, 'mdv':0, 'ss':'.', 'ST':ST, 'HT' : hematocrit, 'K_ELIM': k_elim, 'K_12': k_12, 'K_21': k_21})
    
    return patient_data

def generate_virtual_cohort_film(num_patients=10, scenario=1):
    all_patients_df = []
    nbr_ss = 6
    observation_times = torch.tensor([0, 0.33, 0.67, 1., 1.5, 2., 3., 4., 6., 9., 12., 24.])+24*nbr_ss
    dose_list = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]/6.0

    print(f"Generating data for {num_patients} virtual patients (2 visits each)...")
    for patient_id in range(1, num_patients + 1):
        formulation = random.choice(['Prograf', 'Advagraf'])
        cyp_status = random.choice(['expresser', 'non_expresser'])
        hematocrit = random.uniform(25.0, 45.0) if scenario == 2 else 35.0

        pk_model = TacrolimusPK(formulation=formulation, hematocrit=hematocrit, distribution_type='log_normal', cyp_status=cyp_status, scenario=scenario)
        pk_model._sample_individual_parameters() # Sample once per patient

        # VISIT 1
        dose_1 = random.choice(dose_list)
        pk_model.dose_mg = dose_1
        data_v1 = generate_patient_visit(pk_model, 1, patient_id, nbr_ss, observation_times, formulation, hematocrit, cyp_status)

        # VISIT 2
        dose_2 = random.choice(dose_list)
        while dose_2 == dose_1: dose_2 = random.choice(dose_list)
        pk_model.dose_mg = dose_2
        data_v2 = generate_patient_visit(pk_model, 2, patient_id, nbr_ss, observation_times, formulation, hematocrit, cyp_status)
        

        all_patients_df.append(pd.DataFrame(data_v1 + data_v2))

    print("Generation complete.")
    return pd.concat(all_patients_df, ignore_index=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generation Tacro FiLM')
    parser.add_argument('--exp', type=str, default='exp_film_run/', help="Path for save experiment")
    parser.add_argument('--num_patients', type=int, default=1000, help="Number of virtual patients to generate")
    parser.add_argument('--scenario', type=int, default=1, help="Type of scenario you want (cf paper)")
    args = parser.parse_args()
    
    os.makedirs(f'exp_run_all/exp_film_run/{args.exp}', exist_ok=True)
    cohort_df = generate_virtual_cohort_film(num_patients=args.num_patients, scenario=args.scenario)
    
    unique_ids = cohort_df['ID'].unique()
    train_ids, test_ids = train_test_split(unique_ids, test_size=0.8, shuffle=False)
    
    train_df = cohort_df[cohort_df['ID'].isin(train_ids)]
    test_df = cohort_df[cohort_df['ID'].isin(test_ids)]
    
    train_df.to_csv(f'exp_run_all/exp_film_run/{args.exp}/virtual_cohort_film_train.csv', index=False)
    test_df.to_csv(f'exp_run_all/exp_film_run/{args.exp}/virtual_cohort_film_test.csv', index=False)
    
    print(f"\nSuccessfully created FiLM cohort with {args.num_patients} patients.")