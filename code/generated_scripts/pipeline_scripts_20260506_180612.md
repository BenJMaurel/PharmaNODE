```python
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
import os

# ==============================================================================
# Population Parameters (Theta values) based on mrgsolve input
# ==============================================================================

POPULATION_PARAMS = {
    'TVCL': 21.2,          # Typical Clearance (L/h)
    'TVV1': 486.0,         # Typical Central Volume (L)
    'TVQ': 79.0,           # Typical Intercomp Clearance (L/h)
    'TVV2': 271.0,         # Typical Peripheral Volume (L)
    'TVKTR': 3.34,         # Typical Transfer Rate (1/h)
    'HTCL': -1.14,         # Exponent for Hematocrit on CL
    'CYPCL': 2.00,         # Multiplier for CYP on CL
    'STKTR': 1.53,         # Multiplier for KTR for Prograf
    'STV1': 0.29           # Multiplier for V1 for Prograf
}

# Inter-Patient Variability (IPV) as standard deviation of the random effect
IPV_OMEGA = {
    'CL':  np.sqrt(0.08), 
    'Vc':  np.sqrt(0.10), 
    'Q':   np.sqrt(0.29), 
    'Vp':  np.sqrt(0.36), 
    'Ktr': np.sqrt(0.06)
}

# Inter-Occasion Variability (IOV)
IOV_KAPPA = {
    'Ktr': 0.33,
    'CL': 0.31,
    'Vc': 0.75,
}

# Residual error
RESIDUAL_ERROR_PROP_SD = 0.113  
RESIDUAL_ERROR_ADD_SD = 0.71  

class TacrolimusPK(nn.Module):
    """
    A class to simulate the pharmacokinetics of Tacrolimus.
    Implements a 2-compartment model with 3 transit compartments (Erlang absorption).
    """
    def __init__(self, 
                 formulation='Advagraf', 
                 hematocrit=35.0, 
                 cyp_status='non_expresser',
                 distribution_type = 'log_normal',
                 scenario = 1,
                 device=torch.device("cpu"), 
                 adjoint=False):
        super().__init__()
        
        if formulation not in ['Prograf', 'Advagraf']:
            raise ValueError("Formulation must be 'Prograf' or 'Advagraf'.")
        if cyp_status not in ['expresser', 'non_expresser']:
            raise ValueError("cyp_status must be 'expresser' or 'non_expresser'.")
            
        self.distribution_type = distribution_type
        self.formulation = formulation
        self.hematocrit = hematocrit
        self.cyp_status = cyp_status
        dose_list = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        self.dose_mg = dose_list[random.randint(0, len(dose_list) - 1)]
        self.device = device
        self.odeint = odeint_adjoint if adjoint else odeint
        
        self.pop_params = POPULATION_PARAMS
        self.ipv = IPV_OMEGA
        self.iov = IOV_KAPPA
        self.scenario = scenario
        self.individual_params = {}

    def _sample_individual_parameters(self, t_df=4):
        # 1. Calculate Typical Values (TV) based on covariates
        study_factor = 1.0 if self.formulation == 'Prograf' else 0.0
        cyp_factor = 1.0 if self.cyp_status == 'expresser' else 0.0

        tv_ktr = self.pop_params['TVKTR'] * (self.pop_params['STKTR'] ** study_factor)
        tv_cl = self.pop_params['TVCL'] * (self.pop_params['CYPCL'] ** cyp_factor) * ((self.hematocrit / 35.0) ** self.pop_params['HTCL'])
        tv_q = self.pop_params['TVQ']
        tv_vc = self.pop_params['TVV1'] * (self.pop_params['STV1'] ** study_factor)
        tv_vp = self.pop_params['TVV2']

        distribution_type = self.distribution_type
        params = {'Ktr': tv_ktr, 'CL': tv_cl, 'Q': tv_q, 'Vc': tv_vc, 'Vp': tv_vp}

        for p_name, tv_p in params.items():
            eta = 0.0
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
        """
        State vector: (A_depot, A_trans1, A_trans2, A_trans3, A_central, A_peripheral)
        """
        A_depot, A_t1, A_t2, A_t3, A_c, A_p = state

        Ktr = self.individual_params['Ktr']
        CL = self.individual_params['CL']
        Q = self.individual_params['Q']
        Vc = self.individual_params['Vc']
        Vp = self.individual_params['Vp']

        # Micro-rate constants
        k_el = CL / Vc
        k_12 = Q / Vc
        k_21 = Q / Vp

        # ODEs
        dA_depot = -Ktr * A_depot
        dA_t1    = Ktr * A_depot - Ktr * A_t1
        dA_t2    = Ktr * A_t1 - Ktr * A_t2
        dA_t3    = Ktr * A_t2 - Ktr * A_t3
        dA_c     = Ktr * A_t3 - (k_el * A_c) - (k_12 * A_c) + (k_21 * A_p)
        dA_p     = (k_12 * A_c) - (k_21 * A_p)

        return dA_depot, dA_t1, dA_t2, dA_t3, dA_c, dA_p
    
    def get_initial_state(self):
        t0 = torch.tensor([0.0], device=self.device)
        state = tuple(torch.tensor([0.0], device=self.device) for _ in range(6))
        return t0, state
    
    def state_update(self, state):
        A_depot, A_t1, A_t2, A_t3, A_c, A_p = state
        A_depot = A_depot + self.dose_mg
        return A_depot, A_t1, A_t2, A_t3, A_c, A_p
     
    def simulate(self, dosing_times, time_points):
        self._sample_individual_parameters(t_df=4)
        t0, state = self.get_initial_state()
        
        if 0.0 in dosing_times:
             state = self.state_update(state)
        
        all_concentrations = [torch.tensor([[0.0]], device=self.device)]
        dosing_times = sorted(dosing_times)
        last_time = t0
        
        for event_t in dosing_times:
            if event_t > last_time:
                ts_interval = time_points[(time_points > last_time) & (time_points <= event_t)]
                if len(ts_interval) > 0:
                    tt = torch.cat([last_time, ts_interval])
                    solution = self.odeint(self, state, tt, atol=1e-6, rtol=1e-6)
                    # Index 4 is Central Compartment
                    concentrations = (solution[4][1:] / self.individual_params['Vc']) * 1000
                    all_concentrations.append(concentrations)
                    state = tuple(s[-1] for s in solution)
            
            if event_t > 0.0:
                 state = self.state_update(state) 
            last_time = torch.tensor([event_t], device=self.device)
        
        ts_final = time_points[time_points > last_time]
        if len(ts_final) > 0:
            tt = torch.cat([last_time, ts_final])
            solution = self.odeint(self, state, tt, atol=1e-6, rtol=1e-6)
            concentrations = (solution[4][1:] / self.individual_params['Vc']) * 1000
            all_concentrations.append(concentrations)
        return torch.cat(all_concentrations)

def compare_distributions(num_patients=5000, t_df=4):
    results = []
    for i in range(num_patients):
        formulation = random.choice(['Prograf', 'Advagraf'])
        cyp_status = random.choice(['expresser', 'non_expresser'])
        model_normal = TacrolimusPK(formulation=formulation, hematocrit=35.0, distribution_type='log_normal', cyp_status=cyp_status, scenario=1)
        params_ln = model_normal._sample_individual_parameters()
        for p_name, p_val in params_ln.items():
            results.append({'patient_id': i, 'param': p_name, 'value': float(p_val), 'distribution': 'Log-Normal'})
    df = pd.DataFrame(results)
    sns.set_style("whitegrid")
    param_list = ['CL', 'Vc', 'Ktr', 'Q', 'Vp']
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten() 
    for i, p_name in enumerate(param_list):
        ax = axes[i]
        df_param = df[df['param'] == p_name]
        sns.kdeplot(data=df_param, x='value', hue='distribution', ax=ax, fill=True, alpha=0.1)
        ax.set_title(f'Distribution of {p_name}')
    axes[5].set_visible(False)
    plt.tight_layout()

def compare_trajectories(num_patients=20):
    results = []
    sim_times = torch.arange(0, 24.1, 0.1) 
    for i in range(num_patients):
        formulation = random.choice(['Prograf', 'Advagraf'])
        cyp = random.choice(['expresser', 'non_expresser'])
        hct = random.uniform(25.0, 45.0)
        model = TacrolimusPK(formulation=formulation, hematocrit=hct, cyp_status=cyp, scenario=1)
        conc = model.simulate(dosing_times=[0.0], time_points=sim_times)
        conc = conc.detach().cpu().numpy().flatten() * 1000
        t_np = sim_times.detach().cpu().numpy()
        for t, c in zip(t_np, conc):
            results.append({'Patient': i, 'Time': t, 'Concentration': c, 'Model': 'Linear'})
    df = pd.DataFrame(results)
    sns.lineplot(data=df, x='Time', y='Concentration', hue='Model', units='Patient', estimator=None, alpha=0.5)

def generate_virtual_cohort(num_patients=10, scenario=1):
    all_rows = [] 
    nbr_ss = 6
    observation_times = torch.tensor([0, 0.33, 0.67, 1., 1.5, 2., 3., 4., 6., 9., 12., 24.]) + 24*nbr_ss
    generated_patients = 0
    while generated_patients < num_patients:
        formulation = random.choice(['Prograf', 'Advagraf'])
        cyp_status = random.choice(['expresser', 'non_expresser'])
        hematocrit = 35.0
        pk_model = TacrolimusPK(formulation=formulation, hematocrit=hematocrit, distribution_type='log_normal', cyp_status=cyp_status, scenario=scenario)
        
        sim_times = observation_times[observation_times > 0]
        start_time = 0.0
        end_time = sim_times.max().item()
        fine_grained_times = torch.arange(start_time, end_time, 0.1)
        all_sim_times = torch.unique(torch.cat([sim_times, fine_grained_times]))
        
        if formulation == 'Advagraf':
            doses = [i*24 for i in range(nbr_ss+1)]
        else:
            doses = [24*nbr_ss - 12*(nbr_ss - i) for i in range(nbr_ss+1)]
            
        true_concentrations_all = pk_model.simulate(dosing_times=doses, time_points=all_sim_times) * 1000
        mask = torch.isin(all_sim_times, sim_times)
        true_concentrations = true_concentrations_all[mask]
        
        CL_F, Q_F, Vc_F, Vp_F = pk_model.individual_params['CL'].item(), pk_model.individual_params['Q'].item(), pk_model.individual_params['Vc'].item(), pk_model.individual_params['Vp'].item()
        k_elim, k_12, k_21 = CL_F/Vc_F, Q_F/Vc_F, Q_F/Vp_F

        ii = 24.0 if formulation == 'Advagraf' else 12.0
        CYP = 1 if cyp_status == 'expresser' else 0
        ST = 0 if formulation == 'Advagraf' else 1
        
        sd_error = RESIDUAL_ERROR_ADD_SD + RESIDUAL_ERROR_PROP_SD * true_concentrations
        noise = torch.randn_like(true_concentrations)
        concentrations = torch.clamp(true_concentrations + sd_error * noise, min=0.0)

        if formulation == 'Prograf':    
            auc = np.trapezoid(true_concentrations_all.squeeze(-1)[(all_sim_times >= nbr_ss*24) & (all_sim_times <= (nbr_ss+1)*24 - 12)], all_sim_times[(all_sim_times >= (nbr_ss)*24) & (all_sim_times <= (nbr_ss+1)*24 - 12)] - (nbr_ss+1)*24.)
        else:
            auc = np.trapezoid(true_concentrations_all.squeeze(-1)[all_sim_times >= (nbr_ss)*24], all_sim_times[all_sim_times >= nbr_ss*24] - nbr_ss*24.)

        generated_patients += 1
        patient_id = generated_patients
        patient_data = []
        patient_data.append({'ID': patient_id, 'TIME': 0.0, 'DV': '.', 'AMT': pk_model.dose_mg, 'PERI': 1, 'CYP':CYP, 'II':ii, 'DRUG':formulation, 'nbr_ss': nbr_ss,'AUC': auc, 'mdv':1, 'ss':1, 'ST':ST, 'HT': hematocrit, 'K_ELIM': k_elim, 'K_12': k_12, 'K_21': k_21})
        for i in range(nbr_ss):
            patient_data.append({'ID': patient_id, 'TIME': -ii*(i+1), 'DV': '.', 'AMT': pk_model.dose_mg, 'PERI': 1, 'CYP':CYP, 'II':ii, 'DRUG':formulation, 'AUC': auc, 'mdv':1, 'ss':1, 'ST':ST, 'HT': hematocrit, 'K_ELIM': k_elim, 'K_12': k_12, 'K_21': k_21})
        for time, conc, true_conc in zip(sim_times.tolist(), concentrations.tolist(), true_concentrations.tolist()):
            val = conc[0] if conc[0] != 0 else true_conc[0]
            patient_data.append({'ID': patient_id, 'TIME': time-nbr_ss*24, 'DV': val, 'AMT': '.', 'PERI': 1, 'CYP':CYP, 'II':'.','nbr_ss': nbr_ss, 'DRUG':formulation, 'AUC': auc, 'mdv':0, 'ss':'.', 'ST':ST, 'HT' : hematocrit, 'K_ELIM': k_elim, 'K_12': k_12, 'K_21': k_21})
        all_rows.extend(patient_data)

    return pd.DataFrame(all_rows)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generation Tacro')
    parser.add_argument('--exp', type=str, default='./results/', help="Path for save experiment")
    parser.add_argument('--num_patients', type=int, default=300, help="Number of virtual patients to generate")
    parser.add_argument('--first_at', type=int, default=1, help="Do you want to generate test set also?")
    parser.add_argument('--scenario', type=int, default=1, help="Type of scenario you want (cf paper)")
    args = parser.parse_args()
    
    cohort_df = generate_virtual_cohort(num_patients=args.num_patients, scenario=args.scenario)
    unique_ids = cohort_df['ID'].unique()

    if args.first_at == 1:
        train_ids, test_ids = train_test_split(unique_ids, test_size=0.2, shuffle=False)
    elif args.first_at == 2:
        _, test_ids = train_test_split(unique_ids, test_size=0.8, shuffle=False)
    else:
        train_ids = unique_ids
        test_ids = []

    if args.first_at <= 1:
        train_df = cohort_df[cohort_df['ID'].isin(train_ids)]
        try:
            os.makedirs(f'./results/{args.exp}', exist_ok=True)
            train_df.to_csv(f'./results/{args.exp}/virtual_cohort_train.csv', index=False)
        except:
            train_df.to_csv('virtual_cohort_train.csv', index=False)
    if args.first_at >= 1 and len(test_ids) > 0:
        test_df = cohort_df[cohort_df['ID'].isin(test_ids)]
        try:
            test_df.to_csv(f'./results/{args.exp}/virtual_cohort_test.csv', index=False)
        except:
            test_df.to_csv('virtual_cohort_test.csv', index=False)
            
    print(f"Successfully created virtual cohort with {args.num_patients} patients.")
```

```R
# Title: Tacrolimus Pharmacokinetic Analysis 
# Author: benjamin maurel
# Date: 2025-09-09

suppressPackageStartupMessages(library(argparse))
suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(mrgsolve))
suppressPackageStartupMessages(library(mapbayr))
suppressPackageStartupMessages(library(MESS))
suppressPackageStartupMessages(library(lixoftConnectors))
library(glue)
suppressPackageStartupMessages(library(furrr))

parser <- ArgumentParser(description = "Run mapbayr analysis for Tacrolimus PK data.")
parser$add_argument("--virtual_cohort", type = "character", required = TRUE)
parser$add_argument("--output_dir", type = "character", default = ".")
parser$add_argument("--cores", type = "integer", default = 1)
parser$add_argument("--experiment", type = "character", default = ".")
parser$add_argument("--monolix_path", type = "character", default = NULL)
args <- parser$parse_args()

monolix_path <- args$monolix_path
if (is.null(monolix_path) || monolix_path == "") {
  monolix_path <- "/Applications/MonolixSuite2024R1.app/Contents/Resources/monolixSuite"
}

initializeLixoftConnectors(software = "monolix", path = monolix_path)
model_file_path   <- "test_model.txt"
data_file_path    <- file.path(args$output_dir, "virtual_cohort_train.csv")
project_save_path <- file.path(args$output_dir, "2_test_tacro.mlxtran")

column_mapping <- c(
  ID          = "ID",
  TIME        = "TIME",
  DV          = "DV",
  AMT         = "AMT",
  CYP         = "CYP",
  ST          = "ST",
  HT          = "HT"
)

newProject(
  modelFile = model_file_path,
  data = list(dataFile = data_file_path, headerTypes = column_mapping )
)

setIndividualParameterModel(list( correlationBlocks = list(id = list()), 
                                  covariateModel = list(CL = c(CYP = TRUE))) )
setIndividualParameterModel(list( correlationBlocks = list(id = list()), 
                                  covariateModel = list(Vc = c(ST = TRUE))) )
setIndividualParameterModel(list( correlationBlocks = list(id = list()), 
                                  covariateModel = list(KTR = c(ST = TRUE))) )

saveProject(projectFile = project_save_path)
runPopulationParameterEstimation()

monolix_results <- getEstimatedPopulationParameters()
results_df <- data.frame(Parameter = names(monolix_results), Value = unname(monolix_results))

omega_Cl <- monolix_results[["omega_CL"]]^2
omega_Vc <- monolix_results[["omega_Vc"]]^2
omega_Q <- monolix_results[["omega_Q"]]^2
omega_Vp <- monol_results[["omega_Vp"]]^2 # Note: verify key names from Monolix
omega_KTR <- monolix_results[["omega_KTR"]]^2
prop <- monolix_results[["b"]]
add <- monolix_results[['a']]

code_tac <- glue("
[PROB]
[PARAM] @annotated
TVCL : 21.2 : Typical value of clearance (L/h)
TVV1 : 486 : Typical apparent central volume of distribution (L)
TVQ : 79 : Typical intercomp clearance 1 (L/h)
TVV2 : 271 : Typical peripheral volume of distribution (L)
TVKTR : 3.34 : Typical transfer rate constant (1/h)
HTCL : -1.14 : Effect of hematocrit on clearance
CYPCL : 2.00 : Effect of CYP on clearance
STKTR : 1.53 : Effect of study on KTR
STV1 : 0.29 : Effect of study on V1

ETA1 : 0 : ETA on clearance
ETA2 : 0 : ETA on V1
ETA3 : 0 : ETA on Q
ETA4 : 0 : ETA on V2
ETA5 : 0 : ETA on KTR

$PARAM @annotated @covariates
HT : 35 : Hematocrit (percentage)
ST : 1 : Prograf (1) adv (0)
CYP : 0 : Expressor (1) non-expressor (0)

[CMT] @annotated
DEPOT : Dosing compartment (mg) [ADM]
TRANS1 : Transit compartment 1 (mg)
TRANS2 : Transit compartment 2 (mg)
TRANS3 : Transit compartment 3 (mg)
CENT : Central compartment (mg) [OBS]
PERI : Peripheral compartment (mg)

[OMEGA]
{{omega_Cl}}
{{omega_Vc}}
{{omega_Q}}
{{omega_Vp}}
{{omega_KTR}}

[MAIN]
double CL_app = TVCL * pow(CYPCL, CYP) * exp(ETA1 + ETA(1));
double V1_app = TVV1 * pow(STV1, ST) * exp(ETA2 + ETA(2));
double Q = TVQ * exp(ETA3 + ETA(3));
double V2 = TVV2 * exp(ETA4 + ETA(4));
double KTR = TVKTR * pow(STKTR, ST) * exp(ETA5 + ETA(5));

[SIGMA] @annotated
PROP : {{prop}}
ADD : {{add}}

[ODE]
dxdt_DEPOT = -KTR * DEPOT;
dxdt_TRANS1 = KTR * DEPOT - KTR * TRANS1;
dxdt_TRANS2 = KTR * TRANS1 - KTR * TRANS2;
dxdt_TRANS3 = KTR * TRANS2 - KTR * TRANS3;
dxdt_CENT = KTR * TRANS3 - (CL_app + Q) * CENT / V1_app + Q * PERI / V2;
dxdt_PERI = Q * CENT / V1_app - Q * PERI / V2;

[TABLE]
double CONC = CENT / (V1_app)*1000;
capture DV = CONC * (1 + PROP) + ADD;
$CAPTURE DV CL_app V1_app Q V2 KTR
")

mod_tac <- mcode("tac_model", code_tac)
mod_tac_updated <- param(
  mod_tac,
  TVCL  = monolix_results[["CL_pop"]],
  TVV1  = monolix_results[["Vc_pop"]],
  TVQ   = monolix_results[["Q_pop"]],
  TVV2  = monolix_results[["Vp_pop"]],
  TVKTR = monolix_results[["KTR_pop"]],
  CYPCL = exp(monolix_results[["beta_CL_CYP_1"]]),
  STKTR = exp(monolix_results[["beta_KTR_ST_1"]]),
  STV1  = exp(monolix_results[["beta_Vc_ST_1"]])
)

raw_data <- read_csv(args$virtual_cohort, na = c("null", ".", "NA", ""), trim_ws = TRUE, show_col_types = FALSE)
obs_data_to_process <- raw_data %>%
  filter(is.na(AMT)) %>%
  filter(near(TIME, 0) | between(TIME, 0.8, 1.2) | between(TIME, 2.4, 3.6))

if(args$cores > 1) { plan(multisession, workers = args$cores) } else { plan(sequential) }

run_one_id <- function(patient_id, all_raw_data, obs_data) {
  df_obs <- obs_data %>% filter(ID == patient_id)
  if(nrow(df_obs) != 3) return(NULL)
  
  df_dose <- all_raw_data %>% filter(ID == patient_id, !is.na(AMT), TIME == 0)
  if(nrow(df_dose) != 1) return(NULL)
  
  amt_val <- df_dose$AMT
  drug_val <- df_dose$DRUG
  cyp_val <- df_dose$CYP
  auc_obs <- df_dose$AUC
  st_val  <- if_else(drug_val == "Advagraf", 0, 1)
  ii_val  <- if_else(drug_val == "Advagraf", 24, 12)
  
  est_obj <- tryCatch({
    mod_tac_updated %>%
      adm_rows(time = 0, amt = amt_val, ss = 1, ii = ii_val, addl = 4) %>%
      add_covariates(CYP = cyp_val, ST = st_val) %>%
      obs_rows(time = df_obs$TIME[1], DV = df_obs$DV[1]) %>%
      obs_rows(time = df_obs$TIME[2], DV = df_obs$DV[2]) %>%
      obs_rows(time = df_obs$TIME[3], DV = df_obs$DV[3]) %>%
      mapbayest(verbose = FALSE)
  }, error = function(e) return(NULL))

  if (is.null(est_obj)) return(NULL)

  auc_start <- 0
  auc_end <- if (st_val == 1) 12 else 24
  aug <- mapbayr::augment(est_obj, start = auc_start, end = auc_end, delta = 0.1)
  
  ipred_win <- aug$aug_tab %>% filter(type == "IPRED", dplyr::between(time, 0, auc_end)) %>% transmute(time, DV = value)
  auc_ipred <- MESS::auc(ipred_win$time, ipred_win$DV)
  ind_params <- get_param(est_obj, .name = c("CL_app", "V1_app", "Q", "V2", "KTR"))
  
  list(
    plot = plot(aug, main = sprintf("ID %s — amt=%g mg — ii=%gh — ST=%s", patient_id, amt_val, ii_val, as.character(st_val)), xlim = c(0, 24)), 
    results = tibble(ID = patient_id, ST = st_val, amt = amt_val, ii = ii_val, CYP = cyp_val, AUC_observed = auc_obs, auc_ipred = as.numeric(auc_ipred), Cl = ind_params[1], Vc = ind_params[2], Q = ind_params[3], Vp = ind_params[4], Ktr = ind_params[5])
  )
}

stamp    <- format(Sys.Date(), "%Y%m%d")
pdf_file <- file.path(args$output_dir, paste0("tacro_mapbayest_plots_", stamp, ".pdf"))
csv_file <- file.path(args$output_dir, paste0("tacro_mapbayest_auc_",   stamp, ".csv"))

id_list <- obs_data_to_process %>% count(ID) %>% filter(n == 3) %>% pull(ID)
analysis_output <- future_map(id_list, ~run_one_id(.x, all_raw_data = raw_data, obs_data = obs_data_to_process), .options = furrr_options(seed = TRUE), .progress = TRUE)
analysis_output <- purrr::compact(analysis_output)
all_plots <- map(analysis_output, "plot")
results_df  <- map_dfr(analysis_output, "results")

pdf(pdf_file, width = 8, height = 6)
walk(all_plots, print) 
dev.off()
write_csv(results_df, csv_file)

summary_stats <- results_df %>% mutate(bias = (auc_ipred - AUC_observed) / AUC_observed, bias_sq = bias^2) %>% summarise(relative_bias_percent = mean(bias, na.rm = TRUE) * 100, rmse_percent = sqrt(mean(bias_sq, na.rm = TRUE)) * 100)
cat("\n--- Analysis Summary ---\n"); print(summary_stats); cat("------------------------\n\n")
```