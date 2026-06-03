import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint, odeint
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import os

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

RESIDUAL_ERROR_PROP_SD = 0.113  
RESIDUAL_ERROR_ADD_SD = 0.71  

class BatchedTacrolimusPK(nn.Module):
    def __init__(self, is_prograf, hematocrit, is_expresser, dose_mg, scenario=1, device=torch.device("cpu"), adjoint=False):
        super().__init__()
        
        self.N = len(is_prograf)
        self.is_prograf = is_prograf.to(device)
        self.hematocrit = hematocrit.to(device)
        self.is_expresser = is_expresser.to(device)
        self.dose_mg = dose_mg.to(device)
        self.scenario = scenario
        self.device = device
        self.odeint = odeint_adjoint if adjoint else odeint
        
        self.pop_params = POPULATION_PARAMS
        self.ipv = IPV_OMEGA
        self.individual_params = self._sample_individual_parameters()

    def _sample_individual_parameters(self):
        study_factor = self.is_prograf
        cyp_factor = self.is_expresser

        tv_ktr = self.pop_params['theta1_Ktr'] * (self.pop_params['theta2_Ktr_study'] ** study_factor)
        tv_cl = self.pop_params['theta3_CL'] * ((self.hematocrit / 35.0) ** self.pop_params['theta4_CL_HT']) * (self.pop_params['theta5_CL_CYP'] ** cyp_factor)
        tv_q = torch.full((self.N,), self.pop_params['Q'], device=self.device)
        tv_vc = self.pop_params['theta6_Vc'] * (self.pop_params['theta7_Vc_study'] ** study_factor)
        tv_vp = torch.full((self.N,), self.pop_params['Vp'], device=self.device)
        
        params = {'Ktr': tv_ktr, 'CL': tv_cl, 'Q': tv_q, 'Vc': tv_vc, 'Vp': tv_vp}
        
        if self.scenario == 3:
            tv_vmax = self.pop_params['theta_Vmax'] * \
                    ((self.hematocrit / 35.0) ** self.pop_params['theta4_CL_HT']) * \
                    (self.pop_params['theta5_CL_CYP'] ** cyp_factor)
            tv_km = torch.full((self.N,), self.pop_params['theta_Km'], device=self.device)
            
            params.update({
                'Vp2': torch.full((self.N,), self.pop_params['Vp2'], device=self.device), 
                'Q2': torch.full((self.N,), self.pop_params['Q2'], device=self.device),
                'Vmax': tv_vmax,
                'Km': tv_km
            })

        ind_params = {}
        for p_name, tv_p in params.items():
            eta = torch.randn(self.N, device=self.device) * self.ipv.get(p_name, 0.0)
            ind_params[p_name] = tv_p * torch.exp(eta)
            
        return ind_params

    def forward(self, t, state):
        A_depot, A_gut1, A_gut2, A_gut3, A_central, A_peripheral = state

        Ktr, CL_F, Q_F = self.individual_params['Ktr'], self.individual_params['CL'], self.individual_params['Q']
        Vc_F, Vp_F = self.individual_params['Vc'], self.individual_params['Vp']

        k_12 = Q_F / Vc_F  
        k_21 = Q_F / Vp_F  

        dA_depot_dt = -Ktr * A_depot
        dA_gut1_dt = Ktr * A_depot - Ktr * A_gut1
        dA_gut2_dt = Ktr * A_gut1 - Ktr * A_gut2
        dA_gut3_dt = Ktr * A_gut2 - Ktr * A_gut3
        input_to_central = Ktr * A_gut3

        if self.scenario < 3:
            k_elim = CL_F / Vc_F
            dA_central_dt = input_to_central - (k_elim * A_central) - (k_12 * A_central) + (k_21 * A_peripheral)
            dA_peripheral_dt = (k_12 * A_central) - (k_21 * A_peripheral)
        else:
            Vmax, Km = self.individual_params['Vmax'], self.individual_params['Km']
            C_central = A_central / Vc_F
            dA_central_dt = input_to_central - (Vmax * C_central) / (Km + C_central) - (k_12 * A_central) + (k_21 * A_peripheral)
            dA_peripheral_dt = (k_12 * A_central) - (k_21 * A_peripheral)

        return dA_depot_dt, dA_gut1_dt, dA_gut2_dt, dA_gut3_dt, dA_central_dt, dA_peripheral_dt
    
    def simulate_ode(self, dosing_times, time_points):
        state = tuple([torch.zeros(self.N, device=self.device) for _ in range(6)])
        all_concentrations = [torch.zeros(self.N, device=self.device).unsqueeze(0)]
        dosing_times = sorted(dosing_times)
        last_time = torch.tensor([0.0], device=self.device)
        
        if 0.0 in dosing_times:
            state = (state[0] + self.dose_mg, *state[1:])
            
        for i, event_t in enumerate(dosing_times):
            if event_t > last_time.item():
                ts_interval = time_points[(time_points > last_time.item()) & (time_points <= event_t)]
                if len(ts_interval) > 0:
                    tt = torch.cat([last_time, ts_interval])
                    solution = self.odeint(self, state, tt, atol=1e-6, rtol=1e-6)
                    concentrations = solution[4][1:] / self.individual_params['Vc']
                    all_concentrations.append(concentrations)
                    state = tuple(s[-1] for s in solution)
                    
            if event_t > 0.0:
                 state = (state[0] + self.dose_mg, *state[1:])
            last_time = torch.tensor([event_t], device=self.device)
            
        ts_final = time_points[time_points > last_time.item()]
        if len(ts_final) > 0:
            tt = torch.cat([last_time, ts_final])
            solution = self.odeint(self, state, tt, atol=1e-6, rtol=1e-6)
            concentrations = solution[4][1:] / self.individual_params['Vc']
            all_concentrations.append(concentrations)
            
        return torch.cat(all_concentrations, dim=0) 

    def _build_system_matrix(self):
        A = torch.zeros(self.N, 6, 6, device=self.device)
        Ktr, CL_F, Q_F = self.individual_params['Ktr'], self.individual_params['CL'], self.individual_params['Q']
        Vc_F, Vp_F = self.individual_params['Vc'], self.individual_params['Vp']

        k_elim = CL_F / Vc_F
        k_12 = Q_F / Vc_F
        k_21 = Q_F / Vp_F

        A[:, 0, 0] = -Ktr; A[:, 1, 0] = Ktr; A[:, 1, 1] = -Ktr
        A[:, 2, 1] = Ktr; A[:, 2, 2] = -Ktr; A[:, 3, 2] = Ktr
        A[:, 3, 3] = -Ktr; A[:, 4, 3] = Ktr
        A[:, 4, 4] = -(k_elim + k_12); A[:, 4, 5] = k_21
        A[:, 5, 4] = k_12; A[:, 5, 5] = -k_21
        return A

    def simulate_analytical(self, dosing_times, time_points):
        A = self._build_system_matrix()
        state = torch.zeros(self.N, 6, 1, device=self.device)
        all_concentrations = [torch.zeros(self.N, device=self.device).unsqueeze(0)]
        
        dosing_times = sorted(dosing_times)
        last_time = 0.0
        
        if 0.0 in dosing_times:
            state[:, 0, 0] += self.dose_mg
            
        for event_t in dosing_times:
            if event_t > last_time:
                ts_interval = time_points[(time_points > last_time) & (time_points <= event_t)]
                for t in ts_interval:
                    dt = t - last_time
                    P = torch.matrix_exp(A * dt)
                    state = torch.bmm(P, state)
                    conc = state[:, 4, 0] / self.individual_params['Vc']
                    all_concentrations.append(conc.unsqueeze(0))
                    last_time = t.item()
                    
            if event_t > 0.0:
                 state[:, 0, 0] += self.dose_mg
                 
        ts_final = time_points[time_points > last_time]
        for t in ts_final:
            dt = t - last_time
            P = torch.matrix_exp(A * dt)
            state = torch.bmm(P, state)
            conc = state[:, 4, 0] / self.individual_params['Vc']
            all_concentrations.append(conc.unsqueeze(0))
            last_time = t.item()
        
        return torch.cat(all_concentrations, dim=0) 

    def simulate(self, dosing_times, time_points):
        if self.scenario < 3:
            return self.simulate_analytical(dosing_times, time_points)
        else:
            return self.simulate_ode(dosing_times, time_points)

def process_cohort_batch(formulation_str, is_prograf_val, num_patients, start_id, scenario, nbr_ss, observation_times, device):
    if num_patients == 0:
        return []
        
    is_prograf_tensor = torch.full((num_patients,), is_prograf_val, device=device)
    is_expresser_tensor = torch.randint(0, 2, (num_patients,), device=device).float()
    
    if scenario != 2:
        hematocrit_tensor = torch.full((num_patients,), 35.0, device=device)
    else:
        hematocrit_tensor = torch.empty(num_patients, device=device).uniform_(25.0, 45.0)

    dose_choices = torch.tensor([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], device=device)
    dose_idx = torch.randint(0, len(dose_choices), (num_patients,), device=device)
    dose_mg = dose_choices[dose_idx]

    model = BatchedTacrolimusPK(is_prograf_tensor, hematocrit_tensor, is_expresser_tensor, dose_mg, scenario=scenario, device=device)

    sim_times = observation_times[observation_times > 0]
    fine_grained_times = torch.arange(0.0, sim_times.max().item(), 0.1, device=device)
    all_sim_times = torch.unique(torch.cat([sim_times, fine_grained_times]))

    ii = 12.0 if formulation_str == 'Prograf' else 24.0
    ST = 1 if formulation_str == 'Prograf' else 0
    dosing_times = [24*nbr_ss - 12*(nbr_ss - i) for i in range(nbr_ss+1)] if formulation_str == 'Prograf' else [i*24 for i in range(nbr_ss+1)]

    true_concentrations_all = model.simulate(dosing_times=dosing_times, time_points=all_sim_times) * 1000
    mask = torch.isin(all_sim_times, sim_times)
    true_concentrations = true_concentrations_all[mask, :]

    tc_np = true_concentrations.cpu().numpy()
    tc_all_np = true_concentrations_all.cpu().numpy()
    all_times_np = all_sim_times.cpu().numpy()
    
    dose_np = dose_mg.cpu().numpy()
    cyp_np = is_expresser_tensor.cpu().numpy()
    ht_np = hematocrit_tensor.cpu().numpy()
    k_elim_np = (model.individual_params['CL'] / model.individual_params['Vc']).cpu().numpy()
    k_12_np = (model.individual_params['Q'] / model.individual_params['Vc']).cpu().numpy()
    k_21_np = (model.individual_params['Q'] / model.individual_params['Vp']).cpu().numpy()
    
    sim_times_np = sim_times.cpu().numpy()

    if formulation_str == 'Prograf':
        time_mask = (all_times_np >= nbr_ss*24) & (all_times_np <= (nbr_ss+1)*24 - 12)
        base_time = (nbr_ss+1)*24.0
    else:
        time_mask = all_times_np >= nbr_ss*24
        base_time = nbr_ss*24.0
        
    import scipy.integrate
    auc_np = scipy.integrate.trapezoid(tc_all_np[time_mask, :], all_times_np[time_mask] - base_time, axis=0)

    noise = np.random.randn(*tc_np.shape)
    sd_error = RESIDUAL_ERROR_ADD_SD + RESIDUAL_ERROR_PROP_SD * tc_np
    obs_np = np.clip(tc_np + sd_error * noise, a_min=0.0, a_max=None)
    import pdb; pdb.set_trace()
    for m in range(len(sim_times_np)):
        if obs_np[m, 0] == 0:
            obs_np[m, :] = tc_np[m, :]

    all_rows = []
    for n in range(num_patients):
        pid = start_id + n
        CYP = int(cyp_np[n])
        auc = auc_np[n]
        ht = ht_np[n]
        k_e, k_12, k_21 = k_elim_np[n], k_12_np[n], k_21_np[n]
        
        common = {
            'ID': pid, 'PERI': 1, 'CYP': CYP, 'II': ii, 'DRUG': formulation_str,
            'nbr_ss': nbr_ss, 'AUC': auc, 'ST': ST, 'HT': ht,
            'K_ELIM': k_e, 'K_12': k_12, 'K_21': k_21
        }

        all_rows.append({**common, 'TIME': 0.0, 'DV': '.', 'AMT': dose_np[n], 'mdv': 1, 'ss': 1})
        for i in range(nbr_ss):
            all_rows.append({**common, 'TIME': -ii*(i+1), 'DV': '.', 'AMT': dose_np[n], 'mdv': 1, 'ss': 1})
        for m, t in enumerate(sim_times_np):
            all_rows.append({**common, 'TIME': t - nbr_ss*24, 'DV': obs_np[m, n], 'AMT': '.', 'II': '.', 'mdv': 0, 'ss': '.'})
    return all_rows

def generate_virtual_cohort(num_patients=10, scenario=1):
    nbr_ss = 6
    observation_times = torch.tensor([0, 0.33, 0.67, 1., 1.5, 2., 3., 4., 6., 9., 12., 24.]) + 24 * nbr_ss
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Generating data for {num_patients} virtual patients (Batched on {device})...")
    
    num_prograf = num_patients // 2
    num_advagraf = num_patients - num_prograf
    
    rows_prograf = process_cohort_batch('Prograf', 1.0, num_prograf, 1, scenario, nbr_ss, observation_times, device)
    rows_advagraf = process_cohort_batch('Advagraf', 0.0, num_advagraf, num_prograf + 1, scenario, nbr_ss, observation_times, device)

    print("Generation complete.")
    
    COLUMNS = ['ID', 'TIME', 'DV', 'AMT', 'PERI', 'CYP', 'II', 'DRUG', 'nbr_ss', 'AUC', 'mdv', 'ss', 'ST', 'HT', 'K_ELIM', 'K_12', 'K_21']
    df = pd.DataFrame(rows_prograf + rows_advagraf)
    return df[COLUMNS].sort_values(by=['ID', 'TIME']).reset_index(drop=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generation Tacro')
    parser.add_argument('--exp', type=str, default='./results/', help="Path for save experiment")
    parser.add_argument('--num_patients', type=int, default=300, help="Number of virtual patients to generate")
    parser.add_argument('--first_at', type=int, default=1, help="Do you want to generate test set also?")
    parser.add_argument('--scenario', type=int, default=3, help="Type of scenario you want (cf paper)")
    args = parser.parse_args()
    
    NUM_VIRTUAL_PATIENTS = args.num_patients
    cohort_df = generate_virtual_cohort(num_patients=NUM_VIRTUAL_PATIENTS, scenario=args.scenario)
    
    unique_ids = cohort_df['ID'].unique()

    if args.first_at == 1:
        train_ids, test_ids = train_test_split(unique_ids, test_size=0.2, shuffle=False)
    elif args.first_at == 2:
        _, test_ids = train_test_split(unique_ids, test_size=0.8, shuffle=False)
    else:
        train_ids = unique_ids

    os.makedirs(f'./results/{args.exp}', exist_ok=True)
    
    if args.first_at <= 1:
        train_df = cohort_df[cohort_df['ID'].isin(train_ids)]
        train_df.to_csv('virtual_cohort_train.csv', index=False)
        train_df.to_csv(f'./results/{args.exp}/virtual_cohort_train.csv', index=False)
    if args.first_at >= 1:
        test_df = cohort_df[cohort_df['ID'].isin(test_ids)]
        test_df.to_csv('virtual_cohort_test.csv', index=False)
        test_df.to_csv(f'./results/{args.exp}/virtual_cohort_test.csv', index=False)
        
    print(f"\nSuccessfully created virtual cohort with {NUM_VIRTUAL_PATIENTS} patients.")