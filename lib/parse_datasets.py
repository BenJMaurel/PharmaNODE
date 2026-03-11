###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Modified version of Yulia Rubanova
###########################

import os
import numpy as np

import torch
import torch.nn as nn
import pandas as pd
import lib.utils as utils
from lib.diffeq_solver import DiffeqSolver
from torch.distributions import uniform

from torch.utils.data import DataLoader
from lib.read_tacro import TacroDataset, collate_fn_tacro, extract_gen_tac

from sklearn import model_selection
import random

#####################################################################################################
def parse_datasets(args, device):
	random.seed(args.seed)

	def basic_collate_fn(batch, time_steps, args = args, device = device, data_type = "train"):
		batch = torch.stack(batch)
		data_dict = {
			"data": batch, 
			"time_steps": time_steps}

		data_dict = utils.split_and_subsample_batch(data_dict, args, data_type = data_type)
		return data_dict


	dataset_name = args.dataset
	n_total_tp = args.timepoints + args.extrap
	max_t_extrap = args.max_t / args.timepoints * n_total_tp

	##################################################################
	# PK Tacro dataset: For this version only PK_Tacro is functional
	if dataset_name == 'PK_Tacro' or dataset_name == 'PK_MMF' or dataset_name == 'Theo':
		train = True
		# train = False
		test = 1 - train
		if dataset_name == 'PK_Tacro':
			available_datasets = {'gen_tac': extract_gen_tac} # Add your extraction method here to test with another dataset
			datasets_to_load_train = ['gen_tac']
			datasets_to_load = ['gen_tac']
			dict_list = []
			dict_list_train = []
			max_out_list = []
			max_out_list_train = []
			dataset_obj = {}
			dataset_obj_train = {}
			
			for name in datasets_to_load:
				if name in available_datasets:
					# Get the correct function from the map
					extract_function = available_datasets[name]
					# Call the function and get the results
					data_dict, max_out_value = extract_function(plot=True, exp = args.experiment)
					if name not in datasets_to_load_train:
						dict_list.append(data_dict)
					dataset_obj.update(data_dict) # Merge dictionaries
					max_out_list.append(max_out_value)
				if name in datasets_to_load_train:
					# Get the correct function from the map
					extract_function = available_datasets[name]
					# Call the function and get the results
					data_dict, max_out_value = extract_function(plot=False, exp = args.experiment)
					# Collect the results
					dict_list_train.append(data_dict)
					max_out_list_train.append(max_out_value)
			
		max_out = {}
		if not isinstance(max_out_list[0], float):
			_max_out_list = []
			_best_lambda = []
			for list in max_out_list:
				_max_out_list.append(max_out_list[0][0])
				_best_lambda.append(max_out_list[0][1])
			max_out_list = _max_out_list
			max_out['best_lambda'] = np.array(_best_lambda)
		max_out['max_out']=  np.array(max_out_list)
		if datasets_to_load_train == ['gen_tac']:
			train_keys, test_keys = utils.virtual_train_test_list_dict(dict_list_train, train_fraq = 0.8)
		else:
			train_keys, test_keys = utils.split_train_test_list_dict(dict_list_train, train_fraq = 0.8)
		if test == 1:
			_, test_keys_2 =  utils.split_train_test_list_dict(dict_list, train_fraq = 0.0, shuffle = False)
			try:
				test_keys.extend(test_keys_2)
			except:
				test_keys = test_keys_2
			# max_out = np.array(max_out_list)
		
		ids_df = pd.DataFrame({'ID': test_keys})
		# Save the DataFrame to a CSV file.
		# The index=False part is important to avoid an extra unnamed column.
		ids_df.to_csv('/Users/benjaminmaurel/Downloads/ids_to_keep.csv', index=False)

		dataset_train = TacroDataset({k: dataset_obj[k] for k in train_keys})
		dataset_test = TacroDataset({k: dataset_obj[k] for k in test_keys})
		n_samples = len(train_keys) + len(test_keys)
		
		input_dim = 1
		batch_size = min(args.batch_size, args.n)
		train_dataloader = DataLoader(dataset_train, batch_size = batch_size, shuffle=False,
			collate_fn= lambda batch: collate_fn_tacro(batch, args = args, data_type = "train", device = device))
		test_dataloader = DataLoader(dataset_test, batch_size = args.n, shuffle=False,
			collate_fn= lambda batch: collate_fn_tacro(batch, args = args, data_type = "test", device = device))
		data_objects = {"dataset_train": dataset_train,
					"dataset_test": dataset_test,
					"train_dataloader": utils.inf_generator(train_dataloader), 
					"test_dataloader": utils.inf_generator(test_dataloader),
					"input_dim": input_dim,
					"n_train_batches": len(train_dataloader),
					"n_test_batches": len(test_dataloader), "max_out": max_out}
		return data_objects

	########### 1d datasets ###########

	# Sampling args.timepoints time points in the interval [0, args.max_t]
	# Sample points for both training sequence and explapolation (test)
	distribution = uniform.Uniform(torch.Tensor([0.0]),torch.Tensor([max_t_extrap]))
	time_steps_extrap =  distribution.sample(torch.Size([n_total_tp-1]))[:,0]
	time_steps_extrap = torch.cat((torch.Tensor([0.0]), time_steps_extrap))
	time_steps_extrap = torch.sort(time_steps_extrap)[0]

	dataset_obj = None
	##################################################################
	# Sample a periodic function
	if dataset_name == "periodic":
		dataset_obj = Periodic_1d(
			init_freq = None, init_amplitude = 1.,
			final_amplitude = 1., final_freq = None, 
			z0 = 1.)

	if dataset_name == 'PK_Example':
		dataset_obj = PKExample(max_t = args.max_t)
	##################################################################

	if dataset_obj is None:
		raise Exception("Unknown dataset: {}".format(dataset_name))

	dataset = dataset_obj.sample_traj(time_steps_extrap, n_samples = args.n, 
		noise_weight = args.noise_weight)

	# Process small datasets
	dataset = dataset.to(device)
	time_steps_extrap = time_steps_extrap.to(device)

	train_y, test_y = utils.split_train_test(dataset, train_fraq = 0.8)

	n_samples = len(dataset)
	input_dim = dataset.size(-1)

	batch_size = min(args.batch_size, args.n)
	train_dataloader = DataLoader(train_y, batch_size = batch_size, shuffle=False,
		collate_fn= lambda batch: basic_collate_fn(batch, time_steps_extrap, data_type = "train"))
	test_dataloader = DataLoader(test_y, batch_size = args.n, shuffle=False,
		collate_fn= lambda batch: basic_collate_fn(batch, time_steps_extrap, data_type = "test"))
	
	data_objects = {#"dataset_obj": dataset_obj, 
				"train_dataloader": utils.inf_generator(train_dataloader), 
				"test_dataloader": utils.inf_generator(test_dataloader),
				"input_dim": input_dim,
				"n_train_batches": len(train_dataloader),
				"n_test_batches": len(test_dataloader)}

	return data_objects


