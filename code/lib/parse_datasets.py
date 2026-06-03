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
from lib.pk_drug import (
	DrugStudyConfig,
	PKDataset,
	collate_fn_pk,
	extract_gen_pk,
)

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
			drug_config = None
			if args.experiment is not None:
				drug_config = DrugStudyConfig.from_experiment(args.experiment)

			if drug_config is not None:
				base = drug_config.results_path
				train_fp = os.path.join(base, "virtual_cohort_train.csv")
				test_fp = os.path.join(base, "virtual_cohort_test.csv")
				data_dict, max_out_value = extract_gen_pk(
					drug_config, file_path=[train_fp, test_fp], plot=False
				)
				
				train_df = pd.read_csv(train_fp)
				test_df = pd.read_csv(test_fp)
				
				train_ids = list(train_df['ID'].astype(str).unique())
				test_ids = list(test_df['ID'].astype(str).unique())
				
				train_keys = [k for k in train_ids if k in data_dict]
				test_keys = [k for k in test_ids if k in data_dict]
				
				data_dict_train = {k: data_dict[k] for k in train_keys}
				data_dict_test = {k: data_dict[k] for k in test_keys}
				
				dataset_train = PKDataset(data_dict_train)
				dataset_test = PKDataset(data_dict_test)
				collate_train = lambda batch: collate_fn_pk(
					batch, drug_config, args=args, device=device, data_type="train"
				)
				collate_test = lambda batch: collate_fn_pk(
					batch, drug_config, args=args, device=device, data_type="test"
				)
			else:
				available_datasets = {'gen_tac': extract_gen_tac}
				datasets_to_load_train = ['gen_tac']
				datasets_to_load = ['gen_tac']
				dict_list = []
				dict_list_train = []
				max_out_list = []
				dataset_obj = {}

				for name in datasets_to_load:
					if name in available_datasets:
						extract_function = available_datasets[name]
						data_dict, max_out_value = extract_function(plot=True, exp=args.experiment)
						dataset_obj.update(data_dict)
						max_out_list.append(max_out_value)
					if name in datasets_to_load_train:
						extract_function = available_datasets[name]
						data_dict, max_out_value = extract_function(plot=False, exp=args.experiment)
						dict_list_train.append(data_dict)

				if datasets_to_load_train == ['gen_tac']:
					test_fp = f"./results/{args.experiment}/virtual_cohort_test.csv" if args.experiment else "virtual_cohort_test.csv"
					if args.load:
						test_fp = f"./results/{args.load}/virtual_cohort_test.csv"
					if os.path.exists(test_fp):
						test_df = pd.read_csv(test_fp)
						if 'ID_new' in test_df.columns:
							test_keys = list(test_df['ID_new'].astype(str).unique())
						else:
							test_keys = list(test_df['ID'].astype(str).unique())
						
						# Ensure they exist in dataset_obj
						test_keys = [k for k in test_keys if k in dataset_obj]
						train_keys = [k for k in dataset_obj.keys() if k not in test_keys]
					else:
						train_keys, test_keys = utils.virtual_train_test_list_dict(
							dict_list_train, train_fraq=0.8
						)
				else:
					train_keys, test_keys = utils.split_train_test_list_dict(
						dict_list_train, train_fraq=0.8
					)

				dataset_train = TacroDataset({k: dataset_obj[k] for k in train_keys})
				dataset_test = TacroDataset({k: dataset_obj[k] for k in test_keys})
				collate_train = lambda batch: collate_fn_tacro(
					batch, args=args, data_type="train", device=device
				)
				collate_test = lambda batch: collate_fn_tacro(
					batch, args=args, data_type="test", device=device
				)

		max_out = {}
		if isinstance(max_out_value, (list, tuple)) and len(max_out_value) >= 2:
			max_out['best_lambda'] = np.array([max_out_value[1]])
			max_out['max_out'] = np.array([max_out_value[0]])
		else:
			max_out['max_out'] = np.array([max_out_value])
		n_samples = len(train_keys) + len(test_keys)

		input_dim = 1
		batch_size = min(args.batch_size, args.n)
		train_dataloader = DataLoader(
			dataset_train,
			batch_size=batch_size,
			shuffle=False,
			collate_fn=collate_train,
		)
		test_dataloader = DataLoader(
			dataset_test,
			batch_size=len(dataset_test),
			shuffle=False,
			collate_fn=collate_test,
		)
		data_objects = {"dataset_train": dataset_train,
					"dataset_test": dataset_test,
					"train_dataloader": utils.inf_generator(train_dataloader), 
					"test_dataloader": utils.inf_generator(test_dataloader),
					"input_dim": input_dim,
					"n_train_batches": len(train_dataloader),
					"n_test_batches": len(test_dataloader), "max_out": max_out,
					"n_covariates": dataset_train.n_covariates if hasattr(dataset_train, 'n_covariates') else 0}
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


