###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import os
import numpy as np

import torch
import torch.nn as nn
import pandas as pd
import lib.utils as utils
from lib.diffeq_solver import DiffeqSolver
from generate_timeseries import Periodic_1d, PKExample
from torch.distributions import uniform

from torch.utils.data import DataLoader
from mujoco_physics import HopperPhysics
from physionet import PhysioNet, variable_time_collate_fn, get_data_min_max
from person_activity import PersonActivity, variable_time_collate_fn_activity
from lib.read_tacro import TacroDataset, collate_fn_tacro, create_dataset, extract_in_val_mmf, extract_tls, extract_pccp, extract_pigrec, extract_ped, extract_stablocine, extract_concept, extract_stimmugrep, extract_theo, extract_pccp_tac, extract_aadapt_tac, extract_gen_tac

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
	# MuJoCo dataset
	if dataset_name == "hopper":
		dataset_obj = HopperPhysics(root='data', download=True, generate=False, device = device)
		dataset = dataset_obj.get_dataset()[:args.n]
		dataset = dataset.to(device)


		n_tp_data = dataset[:].shape[1]

		# Time steps that are used later on for exrapolation
		time_steps = torch.arange(start=0, end = n_tp_data, step=1).float().to(device)
		time_steps = time_steps / len(time_steps)

		dataset = dataset.to(device)
		time_steps = time_steps.to(device)

		if not args.extrap:
			# Creating dataset for interpolation
			# sample time points from different parts of the timeline, 
			# so that the model learns from different parts of hopper trajectory
			n_traj = len(dataset)
			n_tp_data = dataset.shape[1]
			n_reduced_tp = args.timepoints

			# sample time points from different parts of the timeline, 
			# so that the model learns from different parts of hopper trajectory
			start_ind = np.random.randint(0, high=n_tp_data - n_reduced_tp +1, size=n_traj)
			end_ind = start_ind + n_reduced_tp
			sliced = []
			for i in range(n_traj):
				  sliced.append(dataset[i, start_ind[i] : end_ind[i], :])
			dataset = torch.stack(sliced).to(device)
			time_steps = time_steps[:n_reduced_tp]

		# Split into train and test by the time sequences
		train_y, test_y = utils.split_train_test(dataset, train_fraq = 0.8)

		n_samples = len(dataset)
		input_dim = dataset.size(-1)

		batch_size = min(args.batch_size, args.n)
		train_dataloader = DataLoader(train_y, batch_size = batch_size, shuffle=False,
			collate_fn= lambda batch: basic_collate_fn(batch, time_steps, data_type = "train"))
		test_dataloader = DataLoader(test_y, batch_size = n_samples, shuffle=False,
			collate_fn= lambda batch: basic_collate_fn(batch, time_steps, data_type = "test"))
		
		data_objects = {"dataset_obj": dataset_obj, 
					"train_dataloader": utils.inf_generator(train_dataloader), 
					"test_dataloader": utils.inf_generator(test_dataloader),
					"input_dim": input_dim,
					"n_train_batches": len(train_dataloader),
					"n_test_batches": len(test_dataloader)}
		return data_objects

	##################################################################
	# Physionet dataset

	if dataset_name == "physionet":
		train_dataset_obj = PhysioNet('data/physionet', train=True, 
										quantization = args.quantization,
										download=True, n_samples = min(10000, args.n), 
										device = device)
		# Use custom collate_fn to combine samples with arbitrary time observations.
		# Returns the dataset along with mask and time steps
		test_dataset_obj = PhysioNet('data/physionet', train=False, 
										quantization = args.quantization,
										download=True, n_samples = min(10000, args.n), 
										device = device)

		# Combine and shuffle samples from physionet Train and physionet Test
		total_dataset = train_dataset_obj[:len(train_dataset_obj)]

		if not args.classif:
			# Concatenate samples from original Train and Test sets
			# Only 'training' physionet samples are have labels. Therefore, if we do classifiction task, we don't need physionet 'test' samples.
			total_dataset = total_dataset + test_dataset_obj[:len(test_dataset_obj)]

		# Shuffle and split
		train_data, test_data = model_selection.train_test_split(total_dataset, train_size= 0.8, 
			random_state = 42, shuffle = True)

		record_id, tt, vals, mask, labels = train_data[0]

		n_samples = len(total_dataset)
		input_dim = vals.size(-1)

		batch_size = min(min(len(train_dataset_obj), args.batch_size), args.n)
		data_min, data_max = get_data_min_max(total_dataset)

		train_dataloader = DataLoader(train_data, batch_size= batch_size, shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn(batch, args, device, data_type = "train",
				data_min = data_min, data_max = data_max))
		test_dataloader = DataLoader(test_data, batch_size = n_samples, shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn(batch, args, device, data_type = "test",
				data_min = data_min, data_max = data_max))

		attr_names = train_dataset_obj.params
		data_objects = {"dataset_obj": train_dataset_obj, 
					"train_dataloader": utils.inf_generator(train_dataloader), 
					"test_dataloader": utils.inf_generator(test_dataloader),
					"input_dim": input_dim,
					"n_train_batches": len(train_dataloader),
					"n_test_batches": len(test_dataloader),
					"attr": attr_names, #optional
					"classif_per_tp": False, #optional
					"n_labels": 1} #optional
		return data_objects


	##################################################################
	# PK Tacro dataset
	if dataset_name == 'PK_Tacro' or dataset_name == 'PK_MMF' or dataset_name == 'Theo':
		train = True
		# train = False
		test = 1- train
		if dataset_name == 'PK_Tacro':
			# data_dict_1, max_out_1 = create_dataset("/Users/benjaminmaurel/Documents/Data_LG/FileSenderDownload_7-2-2025__9-5-56/validation_modele_vs_trapeze/tacro_foie/Adv_dev_2.csv", "/Users/benjaminmaurel/Documents/Data_LG/FileSenderDownload_7-2-2025__9-5-56/validation_modele_vs_trapeze/tacro_foie/adv_foie_p2_JD.csv")
			# data_dict_2, max_out_2 = create_dataset("/Users/benjaminmaurel/Documents/Data_LG/FileSenderDownload_7-2-2025__9-5-56/validation_modele_vs_trapeze/tacro_foie/Adv_dev_3.csv", "/Users/benjaminmaurel/Documents/Data_LG/FileSenderDownload_7-2-2025__9-5-56/validation_modele_vs_trapeze/tacro_foie/adv_foie_p3_JD.csv")
			# dataset_obj = {**data_dict_1, **data_dict_2}
			# max_out = max(max_out_1, max_out_2)
			# # dataset_obj, max_out = extract_tls()
			# # dataset_obj, max_out = extract_pccp()
			# # max_out = np.array(max_out_list_train)
			available_datasets = {
			'pccp': extract_pccp_tac, 'aadapt' : extract_aadapt_tac, 'gen_tac': extract_gen_tac
		}
			datasets_to_load_train = args.train_datasets.split(',')
			datasets_to_load_test = args.test_datasets.split(',')
			
			dict_list_train = []
			max_out_list_train = []
			dataset_obj_train = {}
			
			for name in datasets_to_load_train:
				if name in available_datasets:
					extract_function = available_datasets[name]
					if name == 'gen_tac':
						data_dict, max_out_value = extract_function(args.train_data_path, args.test_data_path, plot=False, exp=args.exp)
					else:
						data_dict, max_out_value = extract_function(args.train_data_path, args.pharmac_path, args.auc_path, plot=False)
					dict_list_train.append(data_dict)
					max_out_list_train.append(max_out_value)
					dataset_obj_train.update(data_dict)

			dict_list_test = []
			max_out_list_test = []
			dataset_obj_test = {}

			for name in datasets_to_load_test:
				if name in available_datasets:
					extract_function = available_datasets[name]
					if name == 'gen_tac':
						data_dict, max_out_value = extract_function(args.train_data_path, args.test_data_path, plot=False, exp=args.exp)
					else:
						data_dict, max_out_value = extract_function(args.test_data_path, args.pharmac_path, args.auc_path, plot=False)
					dict_list_test.append(data_dict)
					max_out_list_test.append(max_out_value)
					dataset_obj_test.update(data_dict)

		elif dataset_name == 'Theo':
			dict_list_train, max_out_list_train = extract_theo(args.train_data_path, plot=False)
			dataset_obj_train = dict_list_train
			dict_list_test, max_out_list_test = extract_theo(args.test_data_path, plot=False)
			dataset_obj_test = dict_list_test
		else:
			available_datasets = {
				'pccp': extract_pccp,
				'pigrec': extract_pigrec,
				'concept': extract_concept,
				'stablocine': extract_stablocine,
				'ped': extract_ped,
				'stimmugrep': extract_stimmugrep,
			}
			datasets_to_load_train = args.train_datasets.split(',')
			datasets_to_load_test = args.test_datasets.split(',')

			dict_list_train = []
			max_out_list_train = []
			dataset_obj_train = {}
			
			for name in datasets_to_load_train:
				if name in available_datasets:
					extract_function = available_datasets[name]
					data_dict, max_out_value = extract_function(plot=False)
					dict_list_train.append(data_dict)
					max_out_list_train.append(max_out_value)
					dataset_obj_train.update(data_dict)

			dict_list_test = []
			max_out_list_test = []
			dataset_obj_test = {}
			
			for name in datasets_to_load_test:
				if name in available_datasets:
					extract_function = available_datasets[name]
					data_dict, max_out_value = extract_function(plot=False)
					dict_list_test.append(data_dict)
					max_out_list_test.append(max_out_value)
					dataset_obj_test.update(data_dict)

		max_out = {}
		if max_out_list_train and not isinstance(max_out_list_train[0], float):
			_max_out_list_train = []
			_best_lambda_train = []
			for lst in max_out_list_train:
				_max_out_list_train.append(lst[0])
				_best_lambda_train.append(lst[1])
			max_out_list_train = _max_out_list_train
			max_out['best_lambda_train'] = np.array(_best_lambda_train)

		if max_out_list_test and not isinstance(max_out_list_test[0], float):
			_max_out_list_test = []
			_best_lambda_test = []
			for lst in max_out_list_test:
				_max_out_list_test.append(lst[0])
				_best_lambda_test.append(lst[1])
			max_out_list_test = _max_out_list_test
			max_out['best_lambda_test'] = np.array(_best_lambda_test)

		max_out_list = max_out_list_train + max_out_list_test
		max_out['max_out'] = np.array(max_out_list)

		train_keys, _ = utils.split_train_test_list_dict(dict_list_train, train_fraq=0.8)
		_, test_keys = utils.split_train_test_list_dict(dict_list_test, train_fraq=0.0, shuffle=False)

		dataset_train = TacroDataset({k: dataset_obj_train[k] for k in train_keys})
		dataset_test = TacroDataset({k: dataset_obj_test[k] for k in test_keys})
		n_samples = len(train_keys) + len(test_keys)
		print(n_samples)
		
		input_dim = 1
		# input_dim = 1
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



	##################################################################
	# Human activity dataset

	if dataset_name == "activity":
		n_samples =  min(10000, args.n)
		dataset_obj = PersonActivity('data/PersonActivity', 
							download=True, n_samples =  n_samples, device = device)
		print(dataset_obj)
		# Use custom collate_fn to combine samples with arbitrary time observations.
		# Returns the dataset along with mask and time steps

		# Shuffle and split
		train_data, test_data = model_selection.train_test_split(dataset_obj, train_size= 0.8, 
			random_state = 42, shuffle = True)

		train_data = [train_data[i] for i in np.random.choice(len(train_data), len(train_data))]
		test_data = [test_data[i] for i in np.random.choice(len(test_data), len(test_data))]

		record_id, tt, vals, mask, labels = train_data[0]
		input_dim = vals.size(-1)

		batch_size = min(min(len(dataset_obj), args.batch_size), args.n)
		train_dataloader = DataLoader(train_data, batch_size= batch_size, shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn_activity(batch, args, device, data_type = "train"))
		test_dataloader = DataLoader(test_data, batch_size=n_samples, shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn_activity(batch, args, device, data_type = "test"))

		data_objects = {"dataset_obj": dataset_obj, 
					"train_dataloader": utils.inf_generator(train_dataloader), 
					"test_dataloader": utils.inf_generator(test_dataloader),
					"input_dim": input_dim,
					"n_train_batches": len(train_dataloader),
					"n_test_batches": len(test_dataloader),
					"classif_per_tp": True, #optional
					"n_labels": labels.size(-1)}

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


