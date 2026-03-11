###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Modified version of Yulia Rubanova
###########################

import matplotlib
#matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import os
from scipy.stats import kde

import numpy as np
import subprocess
import torch
from . import utils
import matplotlib.gridspec as gridspec
from .utils import get_device

from .encoder_decoder import *
from .rnn_baselines import *
from .ode_rnn import *
import torch.nn.functional as functional
from torch.distributions.normal import Normal
from .latent_ode import LatentODE
from .read_tacro import auc_linuplogdown
from .likelihood_eval import masked_gaussian_log_density
try:
	import umap
except:
	print("Couldn't import umap")

# from generate_timeseries import Periodic_1d
# from person_activity import PersonActivity

from .utils import compute_loss_all_batches


SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18
LARGE_SIZE = 22

def init_fonts(main_font_size = LARGE_SIZE):
	plt.rc('font', size=main_font_size)          # controls default text sizes
	plt.rc('axes', titlesize=main_font_size)     # fontsize of the axes title
	plt.rc('axes', labelsize=main_font_size - 2)    # fontsize of the x and y labels
	plt.rc('xtick', labelsize=main_font_size - 2)    # fontsize of the tick labels
	plt.rc('ytick', labelsize=main_font_size - 2)    # fontsize of the tick labels
	plt.rc('legend', fontsize=main_font_size - 2)    # legend fontsize
	plt.rc('figure', titlesize=main_font_size)  # fontsize of the figure title


def plot_trajectories(ax, traj, time_steps, min_y = None, max_y = None, title = "", 
		add_to_plot = False, label = None, add_legend = False, dim_to_show = 0,
		linestyle = '-', marker = 'o', mask = None, color = None, linewidth = 1, max_x = None, min_x = None):
	# expected shape of traj: [n_traj, n_timesteps, n_dims]
	# The function will produce one line per trajectory (n_traj lines in total)
	if not add_to_plot:
		ax.cla()
	ax.set_title(title)
	ax.set_xlabel('Time')
	ax.set_ylabel('x')
	
	if min_x is not None:
		ax.set_xlim(left = min_x)

	if max_x is not None:
		ax.set_xlim(right = max_x)

	if min_y is not None:
		ax.set_ylim(bottom = min_y)

	if max_y is not None:
		ax.set_ylim(top = max_y)

	for i in range(traj.size()[0]):
		d = traj[i].cpu().numpy()[:, dim_to_show]
		ts = time_steps.cpu().numpy()
		if mask is not None:
			m = mask[i].cpu().numpy()[:, dim_to_show]
			d = d[m == 1]
			ts = ts[m == 1]
		ax.plot(ts, d, linestyle = linestyle, label = label, marker=marker, color = color, linewidth = linewidth)

	if add_legend:
		ax.legend()


def plot_std(ax, traj, traj_std, time_steps, min_y = None, max_y = None, title = "", 
	add_to_plot = False, label = None, alpha=0.2, color = None):

	# take only the first (and only?) dimension
	mean_minus_std = (traj - traj_std).cpu().numpy()[:, :, 0]
	mean_plus_std = (traj + traj_std).cpu().numpy()[:, :, 0]

	for i in range(traj.size()[0]):
		ax.fill_between(time_steps.cpu().numpy(), mean_minus_std[i], mean_plus_std[i], 
			alpha=alpha, color = color)



def plot_vector_field(ax, odefunc, latent_dim, device):
	# Code borrowed from https://github.com/rtqichen/ffjord/blob/29c016131b702b307ceb05c70c74c6e802bb8a44/diagnostics/viz_toy.py
	K = 13j
	y, x = np.mgrid[-6:6:K, -6:6:K]
	K = int(K.imag)
	zs = torch.from_numpy(np.stack([x, y], -1).reshape(K * K, 2)).to(device, torch.float32)
	if latent_dim > 2:
		# Plots dimensions 0 and 2
		zs = torch.cat((zs, torch.zeros(K * K, latent_dim-2)), 1)
	dydt = odefunc(0, zs)
	dydt = -dydt.cpu().detach().numpy()
	if latent_dim > 2:
		dydt = dydt[:,:2]

	mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
	dydt = (dydt / mag)
	dydt = dydt.reshape(K, K, 2)

	ax.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], #color = dydt[:, :, 0],
		cmap="coolwarm", linewidth=2)

	# ax.quiver(
	# 	x, y, dydt[:, :, 0], dydt[:, :, 1],
	# 	np.exp(logmag), cmap="coolwarm", pivot="mid", scale = 100,
	# )
	ax.set_xlim(-6, 6)
	ax.set_ylim(-6, 6)
	#ax.axis("off")


def plot_auc(ax, data, reconstructions, times, times_pred, auc_be=None, auc_red=None, labels=None, plot_all = True, save_csv = True, patient_ids = None, args = None):
    # Create subplots: e.g., 4 rows x 7 cols (28 total)
    def to_numpy(var):
        if var is None:
            return None
        if torch.is_tensor(var):
            return var.detach().cpu().numpy()
        if isinstance(var, list):
            # If it's a list of tensors, detach each one and stack
            if len(var) > 0 and torch.is_tensor(var[0]):
                return np.array([v.detach().cpu().numpy() for v in var])
            return np.array(var)
        return np.array(var)
    data = to_numpy(data)
    reconstructions = to_numpy(reconstructions)
    time_steps = to_numpy(times)
    time_steps_to_predict = to_numpy(times_pred)
    auc_be = to_numpy(auc_be)
    auc_red = to_numpy(auc_red)
    
    y_true_times = time_steps
    rec_mean = np.mean(reconstructions, axis=0)
    nbr_plot = 10
    pages = reconstructions.shape[1]//10
    if plot_all:
        for j in range(pages):
            fig, axes = plt.subplots(2, nbr_plot//2, figsize=(15, 15), sharex=True, sharey=True)
            axes = axes.flatten()
            for i in range(nbr_plot):
                ax = axes[i]
                patient_reconstructions = reconstructions[:, i+j*nbr_plot, :, 0]
                
                # 2. Calculate the statistics across the samples (axis=0)
                mean_pred = np.mean(patient_reconstructions, axis=0)
                # For a 95% interval, we use the 2.5th and 97.5th percentiles
                lower_bound = np.percentile(patient_reconstructions, 1.0, axis=0)
                upper_bound = np.percentile(patient_reconstructions, 99.0, axis=0)
                # Plotting the prediction (blue line)
                # The '.get_lines()' check ensures we only grab handles for the legend once
                if i == 0:
                    if np.sum(patient_reconstructions[:,50:]) < 0:
                        shade = ax.fill_between(times_pred[:50], lower_bound[:50], upper_bound[:50], 
                                    color="skyblue", alpha=0.8, label="95% Uncertainty Interval")
            # The mean prediction line 📈
                        line1, = ax.plot(times_pred[:50], mean_pred[:50], label="Prediction NODE", color="blue")
                    else:
                        line1, = ax.plot(times_pred[:50], mean_pred[:50], label="Prediction NODE", color="blue")
                        shade = ax.fill_between(times_pred[:50], lower_bound[:50], upper_bound[:50], 
                                    color="skyblue", alpha=0.8, label="95% Uncertainty Interval")
                else:
                    if np.sum(patient_reconstructions[:,50:]) < 0:
                        ax.plot(times_pred[:50], mean_pred[:50], color="blue")
                        shade = ax.fill_between(times_pred[:50], lower_bound[:50], upper_bound[:50], 
                                    color="skyblue", alpha=0.8, label="95% Uncertainty Interval")	
                    else:
                        ax.plot(times_pred[:50], mean_pred[:50], color="blue")
                        shade = ax.fill_between(times_pred[:50], lower_bound[:50], upper_bound[:50], 
                                    color="skyblue", alpha=0.8, label="95% Uncertainty Interval")
                    # else:
                    #     ax.plot(times_pred[:50], rec_mean[i, :50, 0], color="blue")
                    
            
                # Plotting the real data points (red markers)
                # Added linestyle='None' to only show markers without connecting them with a line
                if i == 0:
                    line2, = ax.plot(times[i+j*nbr_plot], data[i+j*nbr_plot], label="Points réels", color="red", marker="o", linestyle='None')
                else:
                    ax.plot(times[i+j*nbr_plot], data[i+j*nbr_plot], color="red", marker="o", linestyle='None')
                
                ax.set_title(f"Patient {i+j*nbr_plot}")
                ax.grid(True)
        
            # 2. ADD A SINGLE, CENTRALIZED LEGEND
            # This avoids cluttering the first subplot. We place it at the top of the figure.
            fig.legend(handles=[line1, line2], loc='upper center', ncol=2, fontsize='x-large', bbox_to_anchor=(0.5, 1.0))
            
            # Adjust layout spacing so the figure legend fits above the subplots.
            plt.tight_layout(rect=[0, 0, 1, 0.98])
            plt.subplots_adjust(hspace=0.4)
            try:
                plt.savefig(f"exp_run_all/{args.load}/all_predict_NODE_TAC_improved_{str(j)}.png", dpi=300, bbox_inches="tight")
            except:
                pass
            # plt.show()

	# Lists to hold data for each patient before combining
    if save_csv:
        all_predictions_list = []
        all_observations_list = []
    
        num_patients = data.shape[0]
    
        print(f"Processing data for {num_patients} patients...")
    
        for i in range(num_patients):
            if patient_ids is not None:
                patient_id = patient_ids[i]
            else:
                patient_id = i # Or use labels if available: labels[i]
            
            # --- Process Predictions ---
            patient_reconstructions = reconstructions[:, i, :, 0]
            
            # Calculate statistics (mean and bounds) for the prediction curves
            mean_pred = np.mean(patient_reconstructions, axis=0)
            # Use 2.5 and 97.5 percentiles here to obtain a 95% interval (instead of 1.0 and 99.0).
            lower_bound = np.percentile(patient_reconstructions, 2.5, axis=0)
            upper_bound = np.percentile(patient_reconstructions, 97.5, axis=0)
            
            # Create a DataFrame for this patient's predictions
            # Using the first 50 time points as in your original code
            if np.sum(patient_reconstructions[:,50:]) == 0:
                pred_df = pd.DataFrame({
                    'patient_id': patient_id,
                    'time': times_pred[:50],
                    'prediction': mean_pred[:50],
                    'lower_ci': lower_bound[:50],
                    'upper_ci': upper_bound[:50]
                })
            else:
                pred_df = pd.DataFrame({
                    'patient_id': patient_id,
                    'time': times_pred,
                    'prediction': mean_pred,
                    'lower_ci': lower_bound,
                    'upper_ci': upper_bound
                })

            all_predictions_list.append(pred_df)
            
            # --- Process Real Observations ---
            obs_df = pd.DataFrame({
                'patient_id': patient_id,
                'time': times[i],
                'value': data[i].squeeze(-1)
            })
            all_observations_list.append(obs_df)
    
        # Combine all individual DataFrames into two final ones
        final_predictions_df = pd.concat(all_predictions_list, ignore_index=True)
        final_observations_df = pd.concat(all_observations_list, ignore_index=True)
    
        # Save to CSV files
        final_predictions_df.to_csv(f"exp_run_all/{args.load}/predictions.csv", index=False)
        final_observations_df.to_csv(f"exp_run_all/{args.load}/observations.csv", index=False)
        final_predictions_df.to_csv("predictions.csv", index=False)
        final_observations_df.to_csv("observations.csv", index=False)
    if auc_red is not None:
        reference = np.array(auc_red)
    else:
        reference = np.array([auc_linuplogdown(dat.squeeze(-1), times) for dat in data])
        # reference = auc_linuplogdown(data.squeeze(-1), times, axis=1)
    
    predicted = np.mean(np.array(reconstructions), axis=0)
    predicted = np.array([np.trapz(predict.squeeze(-1), times_pred) for predict in predicted])
    # Scatter plot
    if labels is not None:
        unique_labels = np.unique(labels)
        cmap = plt.cm.get_cmap('coolwarm', len(unique_labels)) # Get a colormap for unique labels
        for i, label_val in enumerate(unique_labels):
            mask = (labels == label_val)
            ax.scatter(reference[mask], predicted[mask], color=cmap(i), s=20, alpha=0.7, label=f'Category {label_val}')
        ax.legend() # Show legend for categories
    else:
        ax.scatter(reference, predicted, color='black', alpha=0.6, s=20)

    # Regression line
    coeffs = np.polyfit(predicted, reference, deg=1)
    x_vals = np.linspace(min(predicted), max(predicted), 100)
    y_vals = coeffs[0] * x_vals + coeffs[1]
    ax.plot(x_vals, y_vals, linestyle='--', color='gray', linewidth=2)

    # Labels and title
    ax.set_xlabel("Reference AUC0–12h/24h (mg·h/L)")
    ax.set_ylabel("Predicted AUC0–12h/24h (mg·h/L)")
    ax.grid(True)
    ax.set_xlim(min(predicted) - 0.1 * abs(min(predicted)), max(predicted) + 0.1 * abs(min(predicted)))
    ax.set_ylim(min(reference) - 0.1 * abs(min(reference)), max(reference) + 0.1 * abs(min(reference)))
    
    # Calculate overall error and RMSE
    error_overall = np.mean((reference - predicted) / reference)
    error_one_by_one = ((reference - predicted) / reference)
    rmse_overall = np.sqrt(np.mean(((reference - predicted) / reference)**2))
    # Find the indices where the error is greater than the threshold
    threshold = 0.15
    indices_above_threshold = np.where(np.abs(error_one_by_one) > threshold)
    print(f"The indices of values with error above {threshold*100}% are: {indices_above_threshold[0]} with value {error_one_by_one[indices_above_threshold[0]]}")
    # Calculate error and RMSE for each category if labels are provided
    category_metrics = {}
    if labels is not None:
        unique_labels = np.unique(labels)
        for label_val in unique_labels:
            mask = (labels == label_val)
            category_reference = reference[mask]
            category_predicted = predicted[mask]

            if len(category_reference) > 0: # Avoid division by zero or empty categories
                category_error = np.mean((category_reference - category_predicted) / category_reference)
                category_rmse = np.sqrt(np.mean(((category_reference - category_predicted) / category_reference)**2))
                category_metrics[label_val] = {'error': category_error, 'rmse': category_rmse}
            else:
                category_metrics[label_val] = {'error': np.nan, 'rmse': np.nan} # Assign NaN for empty categories

    # Handle auc_be if provided
    error_be_overall = None
    rmse_be_overall = None
    category_metrics_be = {}
    if auc_be is not None:
        if auc_be.shape == (len(auc_be),2,1):
            auc_jb = auc_be[:,1,:]
            auc_be = auc_be[:,0,:]
            auc_be_squeezed = np.array(auc_jb).squeeze(-1)*100
            if np.sqrt(np.mean(((reference - auc_be_squeezed) / reference)**2))>  np.sqrt(np.mean(((reference - np.array(auc_jb).squeeze(-1)) / reference)**2)):
                auc_be_squeezed = np.array(auc_jb).squeeze(-1)
            ax.scatter(reference, auc_be_squeezed, c=labels, cmap='coolwarm', s=20, alpha=0.2, label='AUC_BE')         
            error_jb_overall = np.mean((reference - auc_be_squeezed) / reference)
            error_jb_one_by_one = ((reference - auc_be_squeezed) / reference)
            rmse_jb_overall = np.sqrt(np.mean(((reference - auc_be_squeezed) / reference)**2))
            indices_above_threshold = np.where(np.abs(error_jb_one_by_one) > threshold)
            
            print(f"The BE1 indices of values with percentage error above {threshold*100}% are: {indices_above_threshold[0]} with values {error_jb_one_by_one[indices_above_threshold[0]]}")
            if labels is not None:
                for label_val in unique_labels:
                    mask = (labels == label_val)
                    category_reference = reference[mask]
                    category_auc_be = auc_be_squeezed[mask]            
                    if len(category_reference) > 0:
                        category_error_be = np.mean((category_reference - category_auc_be) / category_reference)
                        category_rmse_be = np.sqrt(np.mean(((category_reference - category_auc_be) / category_reference)**2))
                        category_metrics_be[label_val] = {'error': category_error_be, 'rmse': category_rmse_be}
                    else:
                        category_metrics_be[label_val] = {'error': np.nan, 'rmse': np.nan}
            auc_be_squeezed = np.array(auc_be).squeeze(-1)*100
            if np.sqrt(np.mean(((reference - auc_be_squeezed) / reference)**2))>  np.sqrt(np.mean(((reference - np.array(auc_be).squeeze(-1)) / reference)**2)):
                auc_be_squeezed = np.array(auc_be).squeeze(-1)
            ax.scatter(reference, auc_be_squeezed, c=labels, cmap='coolwarm', s=20, alpha=0.2, label='AUC_BE1')
        auc_be_squeezed = np.array(auc_be).squeeze(-1)*100
        if np.sqrt(np.mean(((reference - auc_be_squeezed) / reference)**2))>  np.sqrt(np.mean(((reference - np.array(auc_be).squeeze(-1)) / reference)**2)):
            auc_be_squeezed = np.array(auc_be).squeeze(-1)
			
        ax.scatter(reference, auc_be_squeezed, c=labels, cmap='coolwarm', s=20, alpha=0.2, label='AUC_BE2')
        error_be_overall = np.mean((reference - auc_be_squeezed) / reference)
        error_be_one_by_one = ((reference - auc_be_squeezed) / reference)
        rmse_be_overall = np.sqrt(np.mean(((reference - auc_be_squeezed) / reference)**2))
        indices_above_threshold = np.where(np.abs(error_be_one_by_one) > threshold)
		
        print(f"The BE2 indices of values with percentage error above {threshold*100}% are: {indices_above_threshold[0]} with values {error_be_one_by_one[indices_above_threshold[0]]}")
        if labels is not None:
            for label_val in unique_labels:
                mask = (labels == label_val)
                category_reference = reference[mask]
                category_auc_be = auc_be_squeezed[mask]

                if len(category_reference) > 0:
                    category_error_be = np.mean((category_reference - category_auc_be) / category_reference)
                    category_rmse_be = np.sqrt(np.mean(((category_reference - category_auc_be) / category_reference)**2))
                    category_metrics_be[label_val] = {'error': category_error_be, 'rmse': category_rmse_be}
                else:
                    category_metrics_be[label_val] = {'error': np.nan, 'rmse': np.nan}


    fig2, ax2 = plt.subplots()    
    if labels is not None and auc_red is not None:
        unique_labels = np.unique(labels)
        cmap = plt.cm.get_cmap('coolwarm', len(unique_labels))
        for i, label_val in enumerate(unique_labels):
            mask = (labels == label_val)
            # Plot primary prediction errors
            if auc_be is not None:
                ax2.scatter(reference[mask], error_be_one_by_one[mask] * 100, color=cmap(i), s=20, alpha=0.2, label=f'Category {label_val}, BE')
            ax2.scatter(reference[mask], error_one_by_one[mask] * 100, color=cmap(i), s=20, alpha=0.8, label=f'Category {label_val}')
    elif auc_be is not None:
        ax2.scatter(reference, error_one_by_one * 100, color='black', s=20, alpha=0.6)    
    # Add a horizontal line for the threshold
    ax2.axhline(y=threshold * 100, color='blue', linestyle='--', linewidth=2, label=f'Threshold ({threshold*100}%)') 
    ax2.axhline(y=-threshold * 100, color='blue', linestyle='--', linewidth=2, label=f'Threshold ({-threshold*100}%)')    
    ax2.set_xlabel("Reference AUC0–12h/24h (mg·h/L)")
    ax2.set_ylabel("Absolute Relative Error (%)")
    ax2.set_title("Relative Error vs. Reference Value")
    ax2.grid(True)
    ax2.legend()
    # Return results based on whether auc_be is provided and labels exist
    if auc_be is not None:
        if labels is not None:
            if auc_be.shape == (len(auc_be),2,1):
                return error_overall, rmse_overall, category_metrics, [error_jb_overall,error_be_overall], [rmse_jb_overall, rmse_be_overall], category_metrics_be, indices_above_threshold[0]
            else:
                return error_overall, rmse_overall, category_metrics, error_be_overall, rmse_be_overall, category_metrics_be, indices_above_threshold[0]               
        else:
            return error_overall, rmse_overall, error_be_overall, rmse_be_overall
    else:
        if labels is not None:
            return error_overall, rmse_overall, category_metrics
        else:
            return error_overall, rmse_overall

def get_meshgrid(npts, int_y1, int_y2):
	min_y1, max_y1 = int_y1
	min_y2, max_y2 = int_y2
	
	y1_grid = np.linspace(min_y1, max_y1, npts)
	y2_grid = np.linspace(min_y2, max_y2, npts)

	xx, yy = np.meshgrid(y1_grid, y2_grid)

	flat_inputs = np.concatenate((np.expand_dims(xx.flatten(),1), np.expand_dims(yy.flatten(),1)), 1)
	flat_inputs = torch.from_numpy(flat_inputs).float()

	return xx, yy, flat_inputs


def add_white(cmap):
	cmaplist = [cmap(i) for i in range(cmap.N)]
	# force the first color entry to be grey
	cmaplist[0] = (1.,1.,1.,1.0)
	# create the new map
	cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
	return cmap


class Visualizations():
	def __init__(self, device):
		self.init_visualization()
		init_fonts(SMALL_SIZE)
		self.device = device

	def init_visualization(self):
		self.fig = plt.figure(figsize=(12, 7), facecolor='white')
		
		self.ax_traj = []
		for i in range(1,4):
			self.ax_traj.append(self.fig.add_subplot(2,3,i, frameon=False))

		# self.ax_density = []
		# for i in range(4,7):
		# 	self.ax_density.append(self.fig.add_subplot(3,3,i, frameon=False))

		#self.ax_samples_same_traj = self.fig.add_subplot(3,3,7, frameon=False)
		self.ax_latent_traj = self.fig.add_subplot(2,3,4, frameon=False)
		self.ax_vector_field = self.fig.add_subplot(2,3,5, frameon=False)
		self.ax_traj_from_prior = self.fig.add_subplot(2,3,6, frameon=False)

		self.plot_limits = {}
		plt.show(block=False)

	def set_plot_lims(self, ax, name):
		if name not in self.plot_limits:
			self.plot_limits[name] = (ax.get_xlim(), ax.get_ylim())
			return

		xlim, ylim = self.plot_limits[name]
		ax.set_xlim(xlim)
		ax.set_ylim(ylim)
            
	def draw_all_plots_film(self, batch_dict, model, plot_name, experimentID=None, save=False, scaler=None):
		"""
        Visualizes the FiLM Extrapolation on the Original Scale.
        """
		import matplotlib.pyplot as plt
		import os
		import torch
		from scipy.special import inv_boxcox
		import numpy as np
        
		fig = plt.figure(figsize=(14, 10), facecolor='white')
        
        # Unpack the batch
		data_v1 = batch_dict["observed_data_v1"]
		tp_v1 = batch_dict["observed_tp_v1"]
		data_v1_target = batch_dict["data_to_predict_v1"]
		tp_v1_target = batch_dict["tp_to_predict_v1"]    
		data_v2 = batch_dict["data_to_predict_v2"]
		tp_v2 = batch_dict["tp_to_predict_v2"]
		dose_v1 = batch_dict["dose_v1"]
		dose_v2 = batch_dict["dose_v2"]
		static_v1 = batch_dict["static_v1"]
		delta_t = batch_dict.get("delta_t", None)
        # Get Predictions
		pred_x_v2, extra_info = model.get_reconstruction_extrapolation(
            data_v1=data_v1, time_steps_v1=tp_v1, time_steps_v2=tp_v2,
            dose_v1=dose_v1, dose_v2=dose_v2, time_steps_to_predict_v1=tp_v1_target, delta_t = delta_t,
            static_v1=static_v1, n_traj_samples=1
        )
        
        # Convert to numpy
		pred_x_v2 = pred_x_v2.squeeze(0).cpu().numpy()
		pred_x_v1 = extra_info["pred_x_v1"].squeeze(0).cpu().numpy()
		data_v1_cpu = data_v1.cpu().numpy()
		tp_v1_cpu = tp_v1.cpu().numpy()
		data_v1_target_cpu = data_v1_target.cpu().numpy()
		tp_v1_target_cpu = tp_v1_target.cpu().numpy()
		data_v2_cpu = data_v2.cpu().numpy()
		tp_v2_cpu = tp_v2.cpu().numpy()
		dose_v1_cpu = dose_v1.cpu().numpy()
		dose_v2_cpu = dose_v2.cpu().numpy()

        # --- NEW: Inverse Scaling Function ---
		def unscale(y):
			if scaler is None: 
				return y
			if 'best_lambda' in scaler.keys():
				y = inv_boxcox(y, scaler['best_lambda'])
				# y = torch.nan_to_num(y, nan=0.0)
			y = y*scaler['max_out']
			return y
        # Apply Unscaling
		data_v1_cpu = unscale(data_v1_cpu)
		data_v1_target_cpu = unscale(data_v1_target_cpu)
		data_v2_cpu = unscale(data_v2_cpu)
		pred_x_v1 = unscale(pred_x_v1)
		pred_x_v2 = unscale(pred_x_v2)
		
		n_plots_to_draw = min(4, data_v1.size(0))
		
		for i in range(n_plots_to_draw):
			ax = fig.add_subplot(2, 2, i+1)
            
			# Plot Visit 1 Observations
			ax.plot(tp_v1_cpu, data_v1_cpu[i, :, 0], 'bX', label=f'V1 Obs (Dose: {dose_v1_cpu[i]:.2f})', markersize=8)
			# Plot Visit 1 True Full Trajectory
			ax.plot(tp_v1_target_cpu, data_v1_target_cpu[i, :, 0], 'b-', alpha=0.3, label='V1 True Full')
			# Plot Visit 1 Base Prediction
			ax.plot(tp_v1_target_cpu, pred_x_v1[i, :, 0], color='cyan', linestyle='--', label='V1 Base Pred', linewidth=2)
			
			# Plot Visit 2 Ground Truth
			ax.plot(tp_v2_cpu, data_v2_cpu[i, :, 0], 'go', label=f'V2 True (Dose: {dose_v2_cpu[i]:.2f})', markersize=6)
			ax.plot(tp_v2_cpu, data_v2_cpu[i, :, 0], 'g-', alpha=0.3)
			# Plot Visit 2 FiLM Extrapolation Prediction
			ax.plot(tp_v2_cpu, pred_x_v2[i, :, 0], 'r--', label='V2 FiLM Pred', linewidth=2)
			
			ax.set_title(f'Patient {i+1} Extrapolation')
			ax.set_xlabel('Time (h)')
			ax.set_ylabel('Concentration (Original Scale)')
			ax.legend(fontsize=8, loc='upper right')
			ax.grid(True, linestyle='--', alpha=0.6)

		plt.tight_layout()
		if save:
			dirname = "plots/" + str(experimentID) + "/"
			os.makedirs(dirname, exist_ok=True)
			plt.savefig(dirname + plot_name)
			plt.close(fig)
		else:
			plt.show()

	def draw_one_density_plot(self, ax, model, data_dict, traj_id, 
		multiply_by_poisson = False):
		
		scale = 5
		cmap = add_white(plt.cm.get_cmap('Blues', 9)) # plt.cm.BuGn_r
		cmap2 = add_white(plt.cm.get_cmap('Reds', 9)) # plt.cm.BuGn_r
		#cmap = plt.cm.get_cmap('viridis')

		data = data_dict["data_to_predict"]
		time_steps = data_dict["tp_to_predict"]
		mask = data_dict["mask_predicted_data"]

		observed_data =  data_dict["observed_data"]
		observed_time_steps = data_dict["observed_tp"]
		observed_mask = data_dict["observed_mask"]

		npts = 50
		xx, yy, z0_grid = get_meshgrid(npts = npts, int_y1 = (-scale,scale), int_y2 = (-scale,scale))
		z0_grid = z0_grid.to(get_device(data))

		if model.latent_dim > 2:
			z0_grid = torch.cat((z0_grid, torch.zeros(z0_grid.size(0), model.latent_dim-2)), 1)

		if model.use_poisson_proc:
			n_traj, n_dims = z0_grid.size()
			# append a vector of zeros to compute the integral of lambda and also zeros for the first point of lambda
			zeros = torch.zeros([n_traj, model.input_dim + model.latent_dim]).to(get_device(data))
			z0_grid_aug = torch.cat((z0_grid, zeros), -1)
		else:
			z0_grid_aug = z0_grid

		# Shape of sol_y [n_traj_samples, n_samples, n_timepoints, n_latents]
		sol_y = model.diffeq_solver(z0_grid_aug.unsqueeze(0), time_steps)
		
		if model.use_poisson_proc:
			sol_y, log_lambda_y, int_lambda, _ = model.diffeq_solver.ode_func.extract_poisson_rate(sol_y)
			
			assert(torch.sum(int_lambda[:,:,0,:]) == 0.)
			assert(torch.sum(int_lambda[0,0,-1,:] <= 0) == 0.)

		pred_x = model.decoder(sol_y)

		# Plot density for one trajectory
		one_traj = data[traj_id]
		mask_one_traj = None
		if mask is not None:
			mask_one_traj = mask[traj_id].unsqueeze(0)
			mask_one_traj = mask_one_traj.repeat(npts**2,1,1).unsqueeze(0)

		ax.cla()

		# Plot: prior
		prior_density_grid = model.z0_prior.log_prob(z0_grid.unsqueeze(0)).squeeze(0)
		# Sum the density over two dimensions
		prior_density_grid = torch.sum(prior_density_grid, -1)
		
		# =================================================
		# Plot: p(x | y(t0))

		masked_gaussian_log_density_grid = masked_gaussian_log_density(pred_x, 
			one_traj.repeat(npts**2,1,1).unsqueeze(0),
			mask = mask_one_traj, 
			obsrv_std = model.obsrv_std).squeeze(-1)

		# Plot p(t | y(t0))
		if model.use_poisson_proc:
			poisson_info = {}
			poisson_info["int_lambda"] = int_lambda[:,:,-1,:]
			poisson_info["log_lambda_y"] = log_lambda_y
	
			poisson_log_density_grid = compute_poisson_proc_likelihood(
				one_traj.repeat(npts**2,1,1).unsqueeze(0),
				pred_x, poisson_info, mask = mask_one_traj)
			poisson_log_density_grid = poisson_log_density_grid.squeeze(0)
			
		# =================================================
		# Plot: p(x , y(t0))

		log_joint_density = prior_density_grid + masked_gaussian_log_density_grid
		if multiply_by_poisson:
			log_joint_density = log_joint_density + poisson_log_density_grid

		density_grid = torch.exp(log_joint_density)

		density_grid = torch.reshape(density_grid, (xx.shape[0], xx.shape[1]))
		density_grid = density_grid.cpu().numpy()

		ax.contourf(xx, yy, density_grid, cmap=cmap, alpha=1)

		# =================================================
		# Plot: q(y(t0)| x)
		#self.ax_density.set_title("Red: q(y(t0) | x)    Blue: p(x, y(t0))")
		ax.set_xlabel('z1(t0)')
		ax.set_ylabel('z2(t0)')

		data_w_mask = observed_data[traj_id].unsqueeze(0)
		if observed_mask is not None:
			data_w_mask = torch.cat((data_w_mask, observed_mask[traj_id].unsqueeze(0)), -1)
		z0_mu, z0_std = model.encoder_z0(
			data_w_mask, observed_time_steps)

		if model.use_poisson_proc:
			z0_mu = z0_mu[:, :, :model.latent_dim]
			z0_std = z0_std[:, :, :model.latent_dim]

		q_z0 = Normal(z0_mu, z0_std)

		q_density_grid = q_z0.log_prob(z0_grid)
		# Sum the density over two dimensions
		q_density_grid = torch.sum(q_density_grid, -1)
		density_grid = torch.exp(q_density_grid)

		density_grid = torch.reshape(density_grid, (xx.shape[0], xx.shape[1]))
		density_grid = density_grid.cpu().numpy()

		ax.contourf(xx, yy, density_grid, cmap=cmap2, alpha=0.3)
	


	def draw_all_plots_one_dim(self, data_dict, model,
		plot_name = "", save = False, experimentID = 0.):
		
		data =  data_dict["data_to_predict"]
		time_steps = data_dict["tp_to_predict"]
		mask = data_dict["mask_predicted_data"]
		observed_data =  data_dict["observed_data"]
		observed_time_steps = data_dict["observed_tp"]
		observed_mask = data_dict["observed_mask"]
		dose = data_dict["dose"]
		static = data_dict["static"]
		device = get_device(time_steps)

		time_steps_to_predict = time_steps
		if isinstance(model, LatentODE):
			# sample at the original time points
			time_steps_to_predict = utils.linspace_vector(time_steps[0], time_steps[-1], 100).to(device)

		reconstructions, info = model.get_reconstruction(time_steps_to_predict, 
			observed_data, observed_time_steps, dose = dose, static = static, mask = observed_mask, n_traj_samples = 10)

		n_traj_to_show = 3
		# plot only 10 trajectories
		data_for_plotting = observed_data[:n_traj_to_show]
		data_for_plotting_val = data[:n_traj_to_show]
		# mask_for_plotting = observed_mask[:n_traj_to_show]
		mask_for_plotting = None
		reconstructions_for_plotting = reconstructions.mean(dim=0)[:n_traj_to_show]
		reconstr_std = reconstructions.std(dim=0)[:n_traj_to_show]

		dim_to_show = 0
		max_y = max(
			data_for_plotting[:,:,dim_to_show].cpu().numpy().max(),
			reconstructions[:,:,dim_to_show].cpu().numpy().max())
		min_y = min(
			data_for_plotting[:,:,dim_to_show].cpu().numpy().min(),
			reconstructions[:,:,dim_to_show].cpu().numpy().min())

		############################################
		# Plot reconstructions, true postrior and approximate posterior

		cmap = plt.cm.get_cmap('Set1')
		for traj_id in range(3):
			# Plot observations
			plot_trajectories(self.ax_traj[traj_id], 
				data_for_plotting[traj_id].unsqueeze(0), observed_time_steps, 
				# mask = mask_for_plotting[traj_id].unsqueeze(0),
				mask = mask_for_plotting,
				min_y = min_y, max_y = max_y, #title="True trajectories", 
				marker = 'o', linestyle='', dim_to_show = dim_to_show,
				color = cmap(2))
			
			plot_trajectories(self.ax_traj[traj_id], 
				data[traj_id].unsqueeze(0), time_steps, 
				# mask = mask_for_plotting[traj_id].unsqueeze(0),
				mask = mask_for_plotting,
				min_y = min_y, max_y = max_y, #title="True trajectories", 
				marker = 'x', linestyle='', dim_to_show = dim_to_show,
				color = cmap(2), add_to_plot=True)
			
			# Plot reconstructions
			plot_trajectories(self.ax_traj[traj_id],
				reconstructions_for_plotting[traj_id].unsqueeze(0), time_steps_to_predict, 
				min_y = min_y, max_y = max_y, title="Sample {} (data space)".format(traj_id), dim_to_show = dim_to_show,
				add_to_plot = True, marker = '', color = cmap(3), linewidth = 3)
			# Plot variance estimated over multiple samples from approx posterior
			plot_std(self.ax_traj[traj_id], 
				reconstructions_for_plotting[traj_id].unsqueeze(0), reconstr_std[traj_id].unsqueeze(0), 
				time_steps_to_predict, alpha=0.5, color = cmap(3))
			self.set_plot_lims(self.ax_traj[traj_id], "traj_" + str(traj_id))
			
			# Plot true posterior and approximate posterior
			# self.draw_one_density_plot(self.ax_density[traj_id],
			# 	model, data_dict, traj_id = traj_id,
			# 	multiply_by_poisson = False)
			# self.set_plot_lims(self.ax_density[traj_id], "density_" + str(traj_id))
			# self.ax_density[traj_id].set_title("Sample {}: p(z0) and q(z0 | x)".format(traj_id))
		############################################
		# Get several samples for the same trajectory
		# one_traj = data_for_plotting[:1]
		# first_point = one_traj[:,0]

		# samples_same_traj, _ = model.get_reconstruction(time_steps_to_predict, 
		# 	observed_data[:1], observed_time_steps, mask = observed_mask[:1], n_traj_samples = 5)
		# samples_same_traj = samples_same_traj.squeeze(1)
		
		# plot_trajectories(self.ax_samples_same_traj, samples_same_traj, time_steps_to_predict, marker = '')
		# plot_trajectories(self.ax_samples_same_traj, one_traj, time_steps, linestyle = "", 
		# 	label = "True traj", add_to_plot = True, title="Reconstructions for the same trajectory (data space)")

		############################################
		# Plot trajectories from prior
		
		if isinstance(model, LatentODE):
			torch.manual_seed(1991)
			np.random.seed(1991)

			traj_from_prior = model.sample_traj_from_prior(time_steps_to_predict, n_traj_samples = 3)
			# Since in this case n_traj = 1, n_traj_samples -- requested number of samples from the prior, squeeze n_traj dimension
			traj_from_prior = traj_from_prior.squeeze(1)

			plot_trajectories(self.ax_traj_from_prior, traj_from_prior, time_steps_to_predict, 
				marker = '', linewidth = 3)
			self.ax_traj_from_prior.set_title("Samples from prior (data space)", pad = 20)
			#self.set_plot_lims(self.ax_traj_from_prior, "traj_from_prior")
		################################################

		# Plot z0
		# first_point_mu, first_point_std, first_point_enc = info["first_point"]

		# dim1 = 0
		# dim2 = 1
		# self.ax_z0.cla()
		# # first_point_enc shape: [1, n_traj, n_dims]
		# self.ax_z0.scatter(first_point_enc.cpu()[0,:,dim1], first_point_enc.cpu()[0,:,dim2])
		# self.ax_z0.set_title("Encodings z0 of all test trajectories (latent space)")
		# self.ax_z0.set_xlabel('dim {}'.format(dim1))
		# self.ax_z0.set_ylabel('dim {}'.format(dim2))

		################################################
		# Show vector field
		self.ax_vector_field.cla()
		error = plot_auc(self.ax_vector_field, data, reconstructions, time_steps, time_steps_to_predict)
		self.ax_vector_field.set_title("AUC error", pad = 20)
		# plot_vector_field(self.ax_vector_field, model.diffeq_solver.ode_func, model.latent_dim, device)
		# self.ax_vector_field.set_title("Slice of vector field (latent space)", pad = 20)
		self.set_plot_lims(self.ax_vector_field, "vector_field")
		#self.ax_vector_field.set_ylim((-0.5, 1.5))

		################################################
		# Plot trajectories in the latent space

		# shape before [1, n_traj, n_tp, n_latent_dims]
		# Take only the first sample from approx posterior
		latent_traj = info["latent_traj"][0,:n_traj_to_show]
		# shape before permute: [1, n_tp, n_latent_dims]

		self.ax_latent_traj.cla()
		cmap = plt.cm.get_cmap('Accent')
		n_latent_dims = latent_traj.size(-1)
		n_latent_dims = 2
		
		custom_labels = {}
		for i in range(n_latent_dims):
			col = cmap(i)
			plot_trajectories(self.ax_latent_traj, latent_traj, time_steps_to_predict, 
				title="Latent trajectories z(t) (latent space)", dim_to_show = i, color = col, 
				marker = '', add_to_plot = True,
				linewidth = 3)
			custom_labels['dim ' + str(i)] = Line2D([0], [0], color=col)
		
		self.ax_latent_traj.set_ylabel("z")
		self.ax_latent_traj.set_title("Latent trajectories z(t) (latent space)", pad = 20)
		self.ax_latent_traj.legend(custom_labels.values(), custom_labels.keys(), loc = 'lower left')
		self.set_plot_lims(self.ax_latent_traj, "latent_traj")

		################################################

		self.fig.tight_layout()
		# plt.draw()

		if save:
			dirname = "plots/" + str(experimentID) + "/"
			os.makedirs(dirname, exist_ok=True)
			self.fig.savefig(dirname + plot_name)
		return self.fig, error
	
	def draw_one_dim(self, data_dict, model,
		plot_name = "", save = False, experimentID = 0.):

		# data =  data_dict[7]
		# time_steps = data_dict[6]
		# mask = data_dict[8]
		data =  data_dict[4]
		time_steps = data_dict[3]
		mask = data_dict[5]
		observed_data =  data_dict[1]
		observed_time_steps = data_dict[0]
		observed_mask = data_dict[2]

		device = get_device(time_steps)

		time_steps_to_predict = time_steps
		if isinstance(model, LatentODE):
			# sample at the original time points
			time_steps_to_predict = utils.linspace_vector(time_steps[0], time_steps[-1], 100).to(device)

		reconstructions, info = model.get_reconstruction(time_steps_to_predict, 
			observed_data, observed_time_steps, mask = observed_mask, n_traj_samples = 10)
		if len(observed_data)>=3:
			n_traj_to_show = 3
		else:
			n_traj_to_show = 1
		# plot only 10 trajectories
		data_for_plotting = observed_data[:n_traj_to_show]
		mask_for_plotting = observed_mask[:n_traj_to_show]
		data_for_plotting_1 = data[:n_traj_to_show]
		mask_for_plotting_1 = mask[:n_traj_to_show]
		reconstructions_for_plotting = reconstructions.mean(dim=0)[:n_traj_to_show]
		reconstr_std = reconstructions.std(dim=0)[:n_traj_to_show]

		dim_to_show = 0
		max_y = max(
			data_for_plotting_1[:,:,dim_to_show].cpu().numpy().max(),
			reconstructions[:,:,dim_to_show].cpu().numpy().max())
		min_y = min(
			data_for_plotting_1[:,:,dim_to_show].cpu().numpy().min(),
			reconstructions[:,:,dim_to_show].cpu().numpy().min())

		# max_y = data_for_plotting_1[:,:,dim_to_show].cpu().numpy().max()
		# min_y = data_for_plotting_1[:,:,dim_to_show].cpu().numpy().min()
		############################################
		# Plot reconstructions, true postrior and approximate posterior

		cmap = plt.cm.get_cmap('Set1')
		for traj_id in range(n_traj_to_show):
			# Plot observations
			plot_trajectories(self.ax_traj[traj_id], 
				data_for_plotting[traj_id].unsqueeze(0), observed_time_steps, 
				mask = mask_for_plotting[traj_id].unsqueeze(0),
				min_y = min_y, max_y = max_y, #title="True trajectories", 
				marker = 'o', linestyle='', dim_to_show = dim_to_show,
				color = cmap(2))
			# Plot True values
			plot_trajectories(self.ax_traj[traj_id], 
				data_for_plotting_1[traj_id].unsqueeze(0), time_steps, 
				mask = mask_for_plotting_1[traj_id].unsqueeze(0),
				min_y = min_y, max_y = max_y, #title="True trajectories", 
				marker = 'x', linestyle='', dim_to_show = dim_to_show,
				color = cmap(2), min_x = 0, max_x = 100., add_to_plot = True)
			# Plot reconstructions
			plot_trajectories(self.ax_traj[traj_id],
				reconstructions_for_plotting[traj_id].unsqueeze(0), time_steps_to_predict, 
				min_y = min_y, max_y = max_y, title="Sample {} (data space)".format(traj_id), dim_to_show = dim_to_show,
				add_to_plot = True, marker = '', color = cmap(3), linewidth = 3, min_x = 0., max_x = 100.)
			# Plot variance estimated over multiple samples from approx posterior
			plot_std(self.ax_traj[traj_id], 
				reconstructions_for_plotting[traj_id].unsqueeze(0), reconstr_std[traj_id].unsqueeze(0), 
				time_steps_to_predict, alpha=0.5, color = cmap(3))
			self.set_plot_lims(self.ax_traj[traj_id], "traj_" + str(traj_id))
			
			# Plot true posterior and approximate posterior
			# self.draw_one_density_plot(self.ax_density[traj_id],
			# 	model, data_dict, traj_id = traj_id,
			# 	multiply_by_poisson = False)
			# self.set_plot_lims(self.ax_density[traj_id], "density_" + str(traj_id))
			# self.ax_density[traj_id].set_title("Sample {}: p(z0) and q(z0 | x)".format(traj_id))
		############################################
		# Get several samples for the same trajectory
		# one_traj = data_for_plotting[:1]
		# first_point = one_traj[:,0]

		# samples_same_traj, _ = model.get_reconstruction(time_steps_to_predict, 
		# 	observed_data[:1], observed_time_steps, mask = observed_mask[:1], n_traj_samples = 5)
		# samples_same_traj = samples_same_traj.squeeze(1)
		
		# plot_trajectories(self.ax_samples_same_traj, samples_same_traj, time_steps_to_predict, marker = '')
		# plot_trajectories(self.ax_samples_same_traj, one_traj, time_steps, linestyle = "", 
		# 	label = "True traj", add_to_plot = True, title="Reconstructions for the same trajectory (data space)")

		############################################
		# Plot trajectories from prior
		
		if isinstance(model, LatentODE):
			torch.manual_seed(1991)
			np.random.seed(1991)

			traj_from_prior = model.sample_traj_from_prior(time_steps_to_predict, n_traj_samples = 3)
			# Since in this case n_traj = 1, n_traj_samples -- requested number of samples from the prior, squeeze n_traj dimension
			traj_from_prior = traj_from_prior.squeeze(1)

			plot_trajectories(self.ax_traj_from_prior, traj_from_prior, time_steps_to_predict, 
				marker = '', linewidth = 3)
			self.ax_traj_from_prior.set_title("Samples from prior (data space)", pad = 20)
			#self.set_plot_lims(self.ax_traj_from_prior, "traj_from_prior")
		################################################

		# Plot z0
		# first_point_mu, first_point_std, first_point_enc = info["first_point"]

		# dim1 = 0
		# dim2 = 1
		# self.ax_z0.cla()
		# # first_point_enc shape: [1, n_traj, n_dims]
		# self.ax_z0.scatter(first_point_enc.cpu()[0,:,dim1], first_point_enc.cpu()[0,:,dim2])
		# self.ax_z0.set_title("Encodings z0 of all test trajectories (latent space)")
		# self.ax_z0.set_xlabel('dim {}'.format(dim1))
		# self.ax_z0.set_ylabel('dim {}'.format(dim2))

		################################################
		# Show vector field
		self.ax_vector_field.cla()
		plot_vector_field(self.ax_vector_field, model.diffeq_solver.ode_func, model.latent_dim, device)
		self.ax_vector_field.set_title("Slice of vector field (latent space)", pad = 20)
		self.set_plot_lims(self.ax_vector_field, "vector_field")
		#self.ax_vector_field.set_ylim((-0.5, 1.5))

		################################################
		# Plot trajectories in the latent space

		# shape before [1, n_traj, n_tp, n_latent_dims]
		# Take only the first sample from approx posterior
		latent_traj = info["latent_traj"][0,:n_traj_to_show]
		# shape before permute: [1, n_tp, n_latent_dims]

		self.ax_latent_traj.cla()
		cmap = plt.cm.get_cmap('Accent')
		n_latent_dims = latent_traj.size(-1)
		n_latent_dims = 2
		custom_labels = {}
		for i in range(n_latent_dims):
			col = cmap(i)
			plot_trajectories(self.ax_latent_traj, latent_traj, time_steps_to_predict, 
				title="Latent trajectories z(t) (latent space)", dim_to_show = i, color = col, 
				marker = '', add_to_plot = True,
				linewidth = 3)
			custom_labels['dim ' + str(i)] = Line2D([0], [0], color=col)
		
		self.ax_latent_traj.set_ylabel("z")
		self.ax_latent_traj.set_title("Latent trajectories z(t) (latent space)", pad = 20)
		self.ax_latent_traj.legend(custom_labels.values(), custom_labels.keys(), loc = 'lower left')
		self.set_plot_lims(self.ax_latent_traj, "latent_traj")

		################################################

		self.fig.tight_layout()
		# plt.draw()

		if save:
			dirname = "plots/" + str(experimentID) + "/"
			os.makedirs(dirname, exist_ok=True)
			self.fig.savefig(dirname + plot_name)
		return self.fig








