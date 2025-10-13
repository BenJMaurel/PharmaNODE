###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

# Create a synthetic dataset
from __future__ import absolute_import, division
from __future__ import print_function
import os
import matplotlib
if os.path.exists("/Users/yulia"):
	matplotlib.use('TkAgg')
else:
	matplotlib.use('Agg')

import numpy as np
import numpy.random as npr
from scipy.special import expit as sigmoid

from torchdiffeq import odeint, odeint_adjoint
from torchdiffeq import odeint_event

import torch.nn as nn
import pickle
import matplotlib.pyplot as plt
import matplotlib.image
import torch
import lib.utils as utils

# ======================================================================================

fixed_effects = {
    "M0": 26.9, "KL": 0.00391, "KD": 0.0589, "R": 0.0202, "EC50": 2.14,
    "Te": 288, "s": 1.22, "E": 1, "beta": 7.7, "CLinf_h": 0.00418, "CLm": 0.959,
    "KCL_h": 942, "gamma": 1.07, "V": 4.23, "Q_h": 0.0463, "V2": 3.59,
    "Vm_h": 0.00501, "Km": 0.0305
}

# Define standard deviations of random effects (assumed to be log-normal)
random_effects = {
    "M0": 0.566, "KL": 0.869, "KD": 0.596, "R": 1.19, "EC50": 1.61,
    "CLinf_h": 0.475, "CLm": 0.792, "KCL_h": 2.09, "gamma": 1.67,
    "V": 0.321, "Q_h": 0.612, "V2": 0.451, "Vm_h": 1.66, "Km": 1.03
}

def get_next_val(init, t, tmin, tmax, final = None):
	if final is None:
		return init
	val = init + (final - init) / (tmax - tmin) * t
	return val


def generate_periodic(time_steps, init_freq, init_amplitude, starting_point, 
	final_freq = None, final_amplitude = None, phi_offset = 0.):

	tmin = time_steps.min()
	tmax = time_steps.max()

	data = []
	t_prev = time_steps[0]
	phi = phi_offset
	for t in time_steps:
		dt = t - t_prev
		amp = get_next_val(init_amplitude, t, tmin, tmax, final_amplitude)
		freq = get_next_val(init_freq, t, tmin, tmax, final_freq)
		phi = phi + 2 * np.pi * freq * dt # integrate to get phase

		y = amp * np.sin(phi) + starting_point
		t_prev = t
		data.append([t,y])
	return np.array(data)

def assign_value_or_sample(value, sampling_interval = [0.,1.]):
	if value is None:
		int_length = sampling_interval[1] - sampling_interval[0]
		return np.random.random() * int_length + sampling_interval[0]
	else:
		return value

class TimeSeries:
	def __init__(self, device = torch.device("cpu")):
		self.device = device
		self.z0 = None

	def init_visualization(self):
		self.fig = plt.figure(figsize=(10, 4), facecolor='white')
		self.ax = self.fig.add_subplot(111, frameon=False)
		plt.show(block=False)

	def visualize(self, truth):
		self.ax.plot(truth[:,0], truth[:,1])

	def add_noise(self, traj_list, time_steps, noise_weight):
		n_samples = traj_list.size(0)

		# Add noise to all the points except the first point
		n_tp = len(time_steps) - 1
		noise = np.random.sample((n_samples, n_tp))
		noise = torch.Tensor(noise).to(self.device)

		traj_list_w_noise = traj_list.clone()
		# Dimension [:,:,0] is a time dimension -- do not add noise to that
		traj_list_w_noise[:,1:,0] += noise_weight * noise
		return traj_list_w_noise


class PKExample(nn.Module):
    def __init__(self, device = torch.device("cpu"), 
	adjoint=False, BW = 70, B2M = 4.5, IGT = 0 ,AS = 0 , SEX = 1, DM = 1, VM = 0.1, KM = 0.3, CL0 = 0.3,
                 OMEGA=[1.0, 0.05, 0.02, 0.05, 0.02, 0.01, 0.01], max_t = 20.):
        super().__init__()
        self.t0 = nn.Parameter(torch.tensor([14.0])) 
        self.max_t = max_t
        # Typical values as learnable parameters
        self.BW = BW
        self.B2M = B2M
        self.IGT = IGT
        self.AS = AS
        self.SEX = SEX
        self.DM = DM
        self.VM = VM
        self.KM = KM
        self.CL0 = CL0
        self.device = device
        # Variability (OMEGA) as learnable parameters
        self.OMEGA = nn.Parameter(torch.tensor(OMEGA, dtype=torch.float32))  # Assuming all OMEGAs are learnable
        self.dose = 3.0
        self.odeint = odeint_adjoint if adjoint else odeint
        self.initialize_parameters()
    def init_visualization(self):
        self.fig = plt.figure(figsize=(10, 4), facecolor='white')
        self.ax = self.fig.add_subplot(111, frameon=False)
        plt.show(block=False)

    def visualize(self, truth):
        self.ax.plot(truth[:,0], truth[:,1])

    def initialize_parameters(self):
        self.sampled_params = {}
        for param, pop_value in fixed_effects.items():
            if param in random_effects:
                omega = random_effects[param]
                # Sample from log-normal using PyTorch
                self.sampled_params[param] = torch.exp(torch.randn(1) * omega + torch.log(torch.tensor(pop_value)))
            else:
                # If no variability, use the fixed value
                self.sampled_params[param] = pop_value
        
    def forward(self, t, state):
        L, A = state
        CLinf = self.sampled_params['CLinf_h']*24
        Q = self.sampled_params['Q_h']*24
        V_m = self.sampled_params['Vm_h']/24
        Cic = L/self.sampled_params['V']
        k21 = Q/self.sampled_params['V2']
        k12 = Q/self.sampled_params['V']
        dxdt_L =  - 1/self.sampled_params['V']*CLinf *torch.exp( self.sampled_params['CLm'] )*L - V_m*L/( self.sampled_params['Km']+ Cic ) -k12*L + k21*A
        dxdt_A = k12*L- k21*A
        return dxdt_L, dxdt_A
	
	
    def get_initial_state(self):
        state = (torch.tensor([3.0/self.sampled_params['V']]), torch.tensor([0.0]))
        return self.t0, state
    
    def state_update(self, state):
        """Updates state based on an event (collision)."""
        L, A = state
        # c = c + self.sizedose
        L = L + 3.0/self.sampled_params['V']
        return L, A
     
    def simulate(self, time_steps=10):
        # event_times = self.get_doses_times(nbounces)
        # event_times = [torch.tensor(7.0), torch.tensor(14.0), torch.tensor(21.0), torch.tensor(28.0)]

        # k = torch.randint(30, 50, (1,)).item()  # Random integer between 0 and n
        k = 5
        # k = 20
        self.initialize_parameters()
        event_times = [torch.tensor(14.0 * i) for i in range(1, k + 1)] 
        # event_times = [torch.tensor(self.max_t)]
        # get dense path
        t0, state = self.get_initial_state()
        # trajectory = [torch.cat([states[None] for states in state], dim = -1)]
        trajectory = [state[0][None]]
        times = [t0.reshape(-1)]
        # trajectory = [state[0][None]]
        # times = [t0.reshape(-1)]
        indice_time_steps = 0
        for event_t in event_times:
            if event_t == event_times[-1]:
                tt = time_steps[indice_time_steps:]
            else:
                tt = time_steps[indice_time_steps: indice_time_steps + len(time_steps)//len(event_times)+1]
                indice_time_steps = indice_time_steps + len(time_steps)//len(event_times)
                
            solution = odeint(self, state, tt, atol=1e-3, rtol=1e-3)

            trajectory.append(solution[0][1:])
            times.append(tt[1:])
            
            state = self.state_update(tuple(s[-1] for s in solution))
            t0 = event_t
        
        return torch.cat(trajectory, dim=0).reshape(-1)
	
	    
    def sample_traj(self, time_steps, n_samples = 1, noise_weight = 1., cut_out_section=None):
        # event_times = self.get_doses_times(nbounces)
        # event_times = [torch.tensor(7.0), torch.tensor(14.0), torch.tensor(21.0), torch.tensor(28.0)]

        traj_list = []
        for i in range(n_samples):
            traj = self.simulate(time_steps)
			# Cut the time dimension
            traj_list.append(traj)

		# shape: [n_samples, n_timesteps, 2]
		# traj_list[:,:,0] -- time stamps
		# traj_list[:,:,1] -- values at the time stamps
        traj_list = np.array(traj_list)
        traj_list = torch.Tensor().new_tensor(traj_list, device = self.device)
        tensor_min = traj_list.min()  # Min along each column
        
        tensor_max = traj_list.max()  # Max along each column

        traj_list = (traj_list) / (tensor_max)
        # traj_list = self.add_noise(traj_list, time_steps, noise_weight)
        return traj_list.unsqueeze(-1)

import pandas as pd
from typing import List, Tuple, Any, Dict

def process_patient_data_from_csv(csv_file_path: str, target_dose_regimen: str) -> Tuple[Dict[Any, List[Tuple[Any, Any]]], Dict[Any, List[Tuple[Any, Any]]], List[Tuple[Any, Any]]]:
    """
    Processes patient data from a CSV file to extract y_1 and y_3 trajectories
    per patient, and Progression-Free Survival (PFS) data for a given dose regimen.

    Args:
        csv_file_path (str): The path to the CSV file.
        target_dose_regimen (str): The specific dose regimen to filter for (e.g., '3Q2W').

    Returns:
        Tuple[Dict[Any, List[Tuple[Any, Any]]], Dict[Any, List[Tuple[Any, Any]]], List[Tuple[Any, Any]]]:
        A tuple containing:
        - y1_trajectories_by_patient: A dictionary where keys are patient IDs and values are
                                      lists of (time, y) tuples for DVID == 1.
        - y3_trajectories_by_patient: A dictionary where keys are patient IDs and values are
                                      lists of (time, y) tuples for DVID == 3.
        - pfs_data_list: A list of (timePFS, CENS) tuples for patients on the target dose regimen.
                         Each tuple represents one patient's PFS data.
    """
    y1_trajectories_by_patient: Dict[Any, List[Tuple[Any, Any]]] = {}
    y3_trajectories_by_patient: Dict[Any, List[Tuple[Any, Any]]] = {}
    pfs_data_list: List[Tuple[Any, Any]] = []

    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
        return y1_trajectories_by_patient, y3_trajectories_by_patient, pfs_data_list
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return y1_trajectories_by_patient, y3_trajectories_by_patient, pfs_data_list

    required_columns = ['DOSEGRP', 'DVID', 'time', 'y', 'ID', 'timePFS', 'CENS']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns in '{csv_file_path}': {', '.join(missing_columns)}")
        return y1_trajectories_by_patient, y3_trajectories_by_patient, pfs_data_list

    df_filtered_dose = df[df['DOSEGRP'] == target_dose_regimen].copy()

    if df_filtered_dose.empty:
        print(f"No data found for dose regimen: '{target_dose_regimen}' in '{csv_file_path}'")
        return y1_trajectories_by_patient, y3_trajectories_by_patient, pfs_data_list

    # Get unique patient IDs within the filtered dose regimen
    unique_patient_ids = df_filtered_dose['ID'].unique()

    for patient_id in unique_patient_ids:
        patient_df = df_filtered_dose[df_filtered_dose['ID'] == patient_id]

        # Extract y_1 trajectory for the current patient
        df_y1_patient = patient_df[patient_df['DVID'] == 1]
        if not df_y1_patient.empty and 'time' in df_y1_patient.columns and 'y' in df_y1_patient.columns:
            # Sort by time to ensure trajectory is in order, if not already
            df_y1_patient = df_y1_patient.sort_values(by='time')
            y1_trajectory = list(zip(df_y1_patient['time'], df_y1_patient['y']))
            if y1_trajectory: # Only add if there are actual observations
                y1_trajectories_by_patient[patient_id] = y1_trajectory

        # Extract y_3 trajectory for the current patient
        df_y3_patient = patient_df[patient_df['DVID'] == 3]
        if not df_y3_patient.empty and 'time' in df_y3_patient.columns and 'y' in df_y3_patient.columns:
            # Sort by time
            df_y3_patient = df_y3_patient.sort_values(by='time')
            y3_trajectory = list(zip(df_y3_patient['time'], df_y3_patient['y']))
            if y3_trajectory: # Only add if there are actual observations
                y3_trajectories_by_patient[patient_id] = y3_trajectory
    
    # Extract PFS data (remains one record per patient in the filtered group)
    if not df_filtered_dose.empty and 'ID' in df_filtered_dose.columns and \
       'timePFS' in df_filtered_dose.columns and 'CENS' in df_filtered_dose.columns:
        
        df_pfs_unique_patients = df_filtered_dose[['ID', 'timePFS', 'CENS']].drop_duplicates(subset=['ID'])
        # Ensure we only consider patients for whom we have PFS data
        # df_pfs_unique_patients = df_pfs_unique_patients.dropna(subset=['timePFS', 'CENS']) # Optional: if you want to exclude patients with missing PFS time/status
        pfs_data_list = list(zip(df_pfs_unique_patients['timePFS'], df_pfs_unique_patients['CENS']))
    else:
        # This condition might be redundant given the initial check for df_filtered_dose.empty,
        # but kept for safety if column checks are more granular.
        print(f"No PFS data or missing 'ID'/'timePFS'/'CENS' columns for dose regimen '{target_dose_regimen}'.")
        
    return y1_trajectories_by_patient, y3_trajectories_by_patient, pfs_data_list

class Periodic_1d(TimeSeries):
	def __init__(self, device = torch.device("cpu"), 
		init_freq = 0.3, init_amplitude = 1.,
		final_amplitude = 10., final_freq = 1., 
		z0 = 0.):
		"""
		If some of the parameters (init_freq, init_amplitude, final_amplitude, final_freq) is not provided, it is randomly sampled.
		For now, all the time series share the time points and the starting point.
		"""
		super(Periodic_1d, self).__init__(device)
		
		self.init_freq = init_freq
		self.init_amplitude = init_amplitude
		self.final_amplitude = final_amplitude
		self.final_freq = final_freq
		self.z0 = z0

	def sample_traj(self, time_steps, n_samples = 1, noise_weight = 1.,
		cut_out_section = None):
		"""
		Sample periodic functions. 
		"""
		traj_list = []
		for i in range(n_samples):
			init_freq = assign_value_or_sample(self.init_freq, [0.4,0.8])
			if self.final_freq is None:
				final_freq = init_freq
			else:
				final_freq = assign_value_or_sample(self.final_freq, [0.4,0.8])
			init_amplitude = assign_value_or_sample(self.init_amplitude, [0.,1.])
			final_amplitude = assign_value_or_sample(self.final_amplitude, [0.,1.])

			noisy_z0 = self.z0 + np.random.normal(loc=0., scale=0.1)

			traj = generate_periodic(time_steps, init_freq = init_freq, 
				init_amplitude = init_amplitude, starting_point = noisy_z0, 
				final_amplitude = final_amplitude, final_freq = final_freq)

			# Cut the time dimension
			traj = np.expand_dims(traj[:,1:], 0)
			traj_list.append(traj)

		# shape: [n_samples, n_timesteps, 2]
		# traj_list[:,:,0] -- time stamps
		# traj_list[:,:,1] -- values at the time stamps
		traj_list = np.array(traj_list)
		traj_list = torch.Tensor().new_tensor(traj_list, device = self.device)
		traj_list = traj_list.squeeze(1)

		traj_list = self.add_noise(traj_list, time_steps, noise_weight)
		return traj_list

