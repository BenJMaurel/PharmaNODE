###########################
# Priority 1 -- disentangled-decoder ablation: TRAINING
#
# Trains the dose-conditioned latent ODE baseline (lib/dose_conditioned.py).
# Standalone: it only imports from lib/ and never touches the checkpoints or
# result folders of the experiments already reported in the paper. Checkpoints
# go to  <save>/exp_dosecond_run/<exp>/experiment_dosecond_<id>[_best].ckpt
###########################

import os
import sys
import json
import time
import argparse

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal

import lib.utils as utils
from lib.read_tacro import (extract_gen_tac_film, TacroFilmDataset,
                            collate_fn_tacro_film, set_static_hematocrit)
from lib.dose_conditioned import create_dose_conditioned_model


def build_parser():
	p = argparse.ArgumentParser('Dose-conditioned latent ODE baseline (revision Priority 1)')
	# --- experiment / data ---
	p.add_argument('--experiment', type=str, required=True,
		help="Experiment ID; the FiLM dataset folder name.")
	p.add_argument('--data-dir', type=str, default='./results/exp_film_run',
		help="Parent folder holding <experiment>/virtual_cohort_film_{train,test}.csv")
	p.add_argument('--save', type=str, default='./results/',
		help="Root folder for checkpoints and logs.")
	p.add_argument('--load', type=str, default=None,
		help="Resume from the checkpoint of this experiment ID.")
	p.add_argument('--consistent-scaling', action='store_true',
		help="Fit the output scaler on train+test jointly (as test_film.py does) "
			 "instead of per-file (as run_models.py does).")
	# --- optimisation ---
	p.add_argument('--niters', type=int, default=6000, help="Number of epochs.")
	p.add_argument('--lr', type=float, default=1e-2)
	p.add_argument('-b', '--batch-size', type=int, default=512)
	p.add_argument('--seed', type=int, default=42)
	p.add_argument('--patience', type=int, default=10000)
	p.add_argument('--smoothing_factor', type=float, default=0.99)
	p.add_argument('--n-traj-samples', type=int, default=3)
	# --- architecture (kept identical to the main model unless stated) ---
	p.add_argument('-l', '--latents', type=int, default=10)
	p.add_argument('--rec-dims', type=int, default=20)
	p.add_argument('--rec-layers', type=int, default=1)
	p.add_argument('--gen-layers', type=int, default=1)
	p.add_argument('-u', '--units', type=int, default=100)
	p.add_argument('-g', '--gru-units', type=int, default=100)
	p.add_argument('--z0-encoder', type=str, default='odernn')
	p.add_argument('--noise-weight', type=float, default=0.01)
	# --- what makes this model different ---
	p.add_argument('--cond-mode', type=str, default='both', choices=['ode', 'decoder', 'both'],
		help="Where the dose enters: the vector field, the decoder, or both.")
	p.add_argument('--n-occ', type=int, default=0,
		help="Occasion covariates conditioning the field and decoder: 2 = [days/100, Ht/30]. "
		     "0 (default) reproduces the published architecture exactly.")
	p.add_argument('--formulation', choices=['all', 'prograf', 'advagraf'], default='all',
		help="Restrict to one formulation; re-pools train+test and re-splits 80/20 within it.")
	p.add_argument('--blank-formulation', action='store_true',
		help="Zero static[:,1] so the encoder cannot use the formulation indicator.")
	p.add_argument('--init-from', type=str, default=None,
		help="Load weights from this checkpoint before training (transfer/fine-tuning).")
	p.add_argument('--freeze', type=str, default='',
		help="Comma-separated top-level modules to freeze, e.g. 'encoder_z0'.")
	p.add_argument('--encoder-dose', type=str, default='concat', choices=['concat', 'zero'],
		help="'concat' keeps the encoder exactly as in the main model (z0 is pushed "
			 "towards dose-invariance by the loss only); 'zero' blanks the dose channel "
			 "and the dose entry of the static vector, making z0 dose-free by construction.")
	p.add_argument('--ode-time', action='store_true',
		help="Also feed the integration time tau to the vector field (f(z, tau, d)).")
	p.add_argument('--decoder-hidden', type=int, default=0,
		help="0 = linear decoder (exactly the main model's decoder plus a dose input); "
			 ">0 = one hidden layer of this width.")
	p.add_argument('--decoder-residual', action='store_true',
		help="Decoder = linear map on [z, dose] + zero-initialised MLP branch (starts at the linear "
			 "decoder). Needs --decoder-hidden > 0. Default off.")
	p.add_argument('--direction', type=str, default='both', choices=['v1', 'both'],
		help="'v1' encodes Visit 1 only; 'both' also encodes Visit 2, mirroring the "
			 "bidirectional protocol the OT-FiLM model is trained with.")
	p.add_argument('--iwae', action='store_true',
		help="Use an IWAE bound instead of the plain ELBO (off by default: the point "
			 "of this baseline is that it needs none of the extra machinery).")
	p.add_argument('--kl-warmup-epochs', type=int, default=20)
	p.add_argument('--learn-obsrv-std', action='store_true',
		help="Learn the observation-noise SD jointly instead of fixing it.")
	p.add_argument('--static-dim', type=int, default=3,
		help="Static covariate channels fed to the encoder: 3 = [dose, formulation, CYP] "
		     "as published, 4 adds hematocrit. Matches run_models.py so the two arms stay "
		     "identical outside the vector field.")
	p.add_argument('--obsrv-std', type=float, default=None,
		help="Observation-noise SD for the Gaussian likelihood; overrides --noise-weight. "
			 "See the note in run_models.py -- the default is far below the empirical "
			 "residual scale, which collapses the posterior.")
	p.add_argument('--kl-coef-max', type=float, default=1.0,
		help="Multiplier on the annealed KL coefficient (beta).")
	p.add_argument('--lr-per-epoch', action='store_true',
		help="Decay the LR once per epoch instead of once per gradient step, making "
			 "the schedule batch-size invariant. See the note in run_models.py.")
	p.add_argument('--log-suffix', type=str, default='',
	               help="Appended to the log filename so runs sharing an experiment "
	                    "(e.g. different seeds) do not overwrite each other's log. "
	                    "Default '' reproduces previous behaviour.")
	p.add_argument('--ckpt-dense-from', type=int, default=0,
	               help="Epoch past which trajectory checkpoints switch to "
	                    "--ckpt-dense-every spacing, so a final metric can be read off "
	                    "several late checkpoints rather than one arbitrary epoch.")
	p.add_argument('--ckpt-dense-every', type=int, default=0,
	               help="Checkpoint spacing once --ckpt-dense-from is passed "
	                    "(defaults to --ckpt-every).")
	p.add_argument('--eval-every', type=int, default=10,
	               help='run the periodic test-set evaluation every N epochs (default 10). '
	               'Larger values cut evaluation overhead on big cohorts; checkpoint '
	               'spacings must be a multiple of it.')
	p.add_argument('--ckpt-every', type=int, default=0,
	               help="If >0, also dump a checkpoint every N epochs into a 'traj/' "
	                    "subfolder. Diagnostic only; never touches the main/_best "
	                    "checkpoint. Default 0 reproduces previous behaviour.")
	p.add_argument('--tensorboard', action='store_true')
	return p



def config_tag(args, spec):
	"""Suffix encoding the non-default flags, so runs with different settings do
	not overwrite each other. All-default runs get an empty tag and therefore
	keep the original filename."""
	parts = []
	for attr, default, fmt in spec:
		val = getattr(args, attr, default)
		if val != default:
			parts.append(fmt(val) if callable(fmt) else f"{fmt}-{val}")
	return ("__" + "_".join(parts)) if parts else ""


DOSECOND_TAG_SPEC = [
	('encoder_dose', 'concat', 'encdose'),
	('cond_mode', 'both', 'cond'),
	('direction', 'both', 'dir'),
	('decoder_hidden', 0, 'dech'),
	('decoder_residual', False, lambda v: 'decres'),
	('ode_time', False, lambda v: 'odetime'),
	('iwae', False, lambda v: 'iwae'),
	('consistent_scaling', False, lambda v: 'jointscale'),
	('obsrv_std', None, 'sig'),
	('kl_coef_max', 1.0, 'beta'),
	('n_occ', 0, 'nocc'),
	('formulation', 'all', 'form'),
	('blank_formulation', False, lambda v: 'blankform'),
	('freeze', '', lambda v: 'frozen-' + v.replace(',', '-')),
]


def load_data(args, device):
	train_csv = os.path.join(args.data_dir, args.experiment, "virtual_cohort_film_train.csv")
	test_csv = os.path.join(args.data_dir, args.experiment, "virtual_cohort_film_test.csv")
	for f in (train_csv, test_csv):
		if not os.path.exists(f):
			raise FileNotFoundError(f"Missing dataset file: {f}")

	if args.consistent_scaling:
		data_train, scale = extract_gen_tac_film(file_path=[train_csv, test_csv])
		data_test, _ = extract_gen_tac_film(file_path=[train_csv, test_csv])
		train_ids = set(TacroFilmDataset(extract_gen_tac_film(file_path=[train_csv])[0]).patient_ids)
		data_train = {k: v for k, v in data_train.items() if k in train_ids}
		data_test = {k: v for k, v in data_test.items() if k not in train_ids}
	else:
		data_train, scale = extract_gen_tac_film(file_path=[train_csv])
		data_test, _ = extract_gen_tac_film(file_path=[test_csv])

	if getattr(args, 'formulation', 'all') != 'all':
		# the published split is by ID order and does not balance formulation
		want = 1 if getattr(args, 'formulation', 'all') == 'prograf' else 0
		pool = {**data_train, **data_test}
		keep = sorted([k for k, v in pool.items() if int(v['v1']['static'][1]) == want])
		rng = np.random.RandomState(0); rng.shuffle(keep)
		ntr = int(round(0.8 * len(keep)))
		data_train = {k: pool[k] for k in keep[:ntr]}
		data_test = {k: pool[k] for k in keep[ntr:]}
		print(f"formulation filter '{getattr(args, 'formulation', 'all')}': "
		      f"{len(data_train)} train / {len(data_test)} test patients")

	max_out = {'max_out': np.array(scale[0]), 'best_lambda': np.array(scale[1])}

	train_loader = DataLoader(TacroFilmDataset(data_train), batch_size=args.batch_size,
		shuffle=True, collate_fn=lambda x: collate_fn_tacro_film(x, device))
	test_loader = DataLoader(TacroFilmDataset(data_test), batch_size=args.batch_size,
		shuffle=False, collate_fn=lambda x: collate_fn_tacro_film(x, device))

	return {
		"input_dim": 1,
		"train_dataloader": utils.inf_generator(train_loader),
		"test_dataloader": utils.inf_generator(test_loader),
		"n_train_batches": len(train_loader),
		"n_test_batches": len(test_loader),
		"max_out": max_out,
	}


def evaluate(model, data_obj, args, kl_coef):
	"""Average the training loss dict over all test batches."""
	keys = ["loss", "likelihood", "mse", "kl_first_p", "std_first_p",
			"rec_loss_v1", "rec_loss_v2", "kl_loss", "rmse_auc"]
	totals = {k: 0.0 for k in keys}
	mse_cond = torch.zeros(4)
	n = data_obj["n_test_batches"]
	for _ in range(n):
		batch_dict = utils.get_next_batch_film(data_obj["test_dataloader"])
		res = model.compute_dose_cond_losses(batch_dict,
			n_traj_samples=10, kl_coef=kl_coef, max_out=data_obj["max_out"],
			direction=args.direction, iwae=args.iwae)
		for k in keys:
			v = res[k]
			totals[k] += float(v.item() if torch.is_tensor(v) else v)
		mse_cond += res["mse_cond"].detach().cpu()
	out = {k: v / n for k, v in totals.items()}
	out["mse_cond"] = (mse_cond / n).tolist()
	return out


def main():
	args = build_parser().parse_args()
	# See run_models.py: the hematocrit channel is opt-in via --static-dim 4.
	set_static_hematocrit(int(getattr(args, 'static_dim', 3)) >= 4)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	experimentID = args.load if args.load is not None else args.experiment

	ckpt_dir = os.path.join(args.save, "exp_dosecond_run", str(args.experiment))
	utils.makedirs(ckpt_dir)
	tag = config_tag(args, DOSECOND_TAG_SPEC)
	ckpt_path = os.path.join(ckpt_dir, f"experiment_dosecond_{experimentID}{tag}.ckpt")
	best_ckpt_path = ckpt_path.replace(".ckpt", "_best.ckpt")

	utils.makedirs("logs/")
	logger = utils.get_logger(
		logpath=os.path.join("logs", f"train_dose_cond_{experimentID}{tag}{getattr(args, 'log_suffix', '')}.log"),
		filepath=os.path.abspath(__file__))
	logger.info(" ".join(sys.argv))

	data_obj = load_data(args, device)
	logger.info(f"train batches: {data_obj['n_train_batches']} | test batches: {data_obj['n_test_batches']}")

	obsrv_std = torch.Tensor([args.obsrv_std if args.obsrv_std is not None
							  else args.noise_weight]).to(device)
	z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.0]).to(device))
	model = create_dose_conditioned_model(args, data_obj["input_dim"], z0_prior, obsrv_std, device)
	model.blank_formulation = args.blank_formulation
	if args.blank_formulation:
		print("encoder: formulation indicator static[:,1] blanked")
	if args.init_from:
		_sd = torch.load(args.init_from, map_location=device, weights_only=False)['state_dict']
		_m, _u = model.load_state_dict(_sd, strict=False)
		print(f"initialised from {args.init_from}"
		      + (f" | missing {len(_m)} unexpected {len(_u)}" if (_m or _u) else ""))
	if args.freeze:
		_tot = _frz = 0; _want = [m.strip() for m in args.freeze.split(',') if m.strip()]
		for _n, _p in model.named_parameters():
			_tot += _p.numel()
			if _n.split('.')[0] in _want:
				_p.requires_grad_(False); _frz += _p.numel()
		print(f"froze {args.freeze}: {_frz:,}/{_tot:,} params ({100*_frz/_tot:.0f}%) held fixed")
	if args.learn_obsrv_std:
		from lib.calibration import enable_learnable_obsrv_std
		enable_learnable_obsrv_std(model, init=args.obsrv_std)
	n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	logger.info(f"dose-conditioned model | cond_mode={args.cond_mode} | encoder_dose={args.encoder_dose} "
				f"| direction={args.direction} | iwae={args.iwae} | trainable params={n_params}")

	if args.load is not None:
		utils.get_ckpt_model(ckpt_path, model, device)

	optimizer = optim.Adamax(model.parameters(), lr=args.lr)
	num_batches = data_obj["n_train_batches"]

	writer = None
	if args.tensorboard:
		from torch.utils.tensorboard import SummaryWriter
		writer = SummaryWriter(os.path.join(args.save, "runs_dosecond", str(experimentID)))

	smoothed_test_mse = float('inf')
	best_smoothed_test_mse = float('inf')
	early_stop_counter = 0
	n_iters_to_viz = getattr(args, 'eval_every', 10) or 10
	start = time.time()

	for itr in range(1, num_batches * (args.niters + 1)):
		optimizer.zero_grad()
		_decay = 0.999 ** (1.0 / num_batches) if args.lr_per_epoch else 0.999
		utils.update_learning_rate(optimizer, decay_rate=_decay, lowest=args.lr / 10)

		epoch = itr // num_batches
		if epoch < args.kl_warmup_epochs:
			kl_coef = 0.
		else:
			kl_coef = args.kl_coef_max * (1 - 0.99 ** (epoch - args.kl_warmup_epochs))

		batch_dict = utils.get_next_batch_film(data_obj["train_dataloader"])
		train_res = model.compute_dose_cond_losses(batch_dict,
			n_traj_samples=args.n_traj_samples, kl_coef=kl_coef,
			max_out=data_obj["max_out"], direction=args.direction, iwae=args.iwae)
		train_res["loss"].backward()
		optimizer.step()

		if itr % (n_iters_to_viz * num_batches) != 0:
			continue

		with torch.no_grad():
			test_res = evaluate(model, data_obj, args, kl_coef)

		logger.info(
			'Epoch {:04d} [DoseCond Test] | Loss {:.6f} | V1 NLL (self) {:.6f} | '
			'V2 NLL (counterfactual) {:.6f} | KL {:.4f} | rmse_auc {:.6f}'.format(
				epoch, test_res["loss"], test_res["rec_loss_v1"],
				test_res["rec_loss_v2"], test_res["kl_loss"], test_res["rmse_auc"]))
		logger.info("KL coef: {}".format(kl_coef))
		logger.info("Train loss (one batch): {}".format(float(train_res["loss"].detach())))

		current_raw_mse = test_res["mse"]
		logger.info("Test MSE (raw): {:.4f}".format(current_raw_mse))
		if smoothed_test_mse == float('inf'):
			smoothed_test_mse = current_raw_mse
		else:
			smoothed_test_mse = (current_raw_mse * args.smoothing_factor
								 + smoothed_test_mse * (1 - args.smoothing_factor))
		logger.info(f"Test MSE (smoothed, alpha={args.smoothing_factor}): {smoothed_test_mse:.4f}")
		for i, v in enumerate(test_res["mse_cond"]):
			logger.info("Test MSE group_{}: {:.4f}".format(i, v))

		if smoothed_test_mse < best_smoothed_test_mse:
			best_smoothed_test_mse = smoothed_test_mse
			early_stop_counter = 0
			logger.info(f"New best smoothed test MSE: {best_smoothed_test_mse:.4f}. Saving to {best_ckpt_path}")
			torch.save({'args': args, 'state_dict': model.state_dict(), 'epoch': epoch,
						'raw_test_mse_at_best': current_raw_mse,
						'smoothed_test_mse': best_smoothed_test_mse}, best_ckpt_path)
		else:
			early_stop_counter += 1
			logger.info(f"No improvement. Early stopping counter: {early_stop_counter}/{args.patience}")

		if writer is not None:
			writer.add_scalar('Loss/train_total_loss', float(train_res["loss"]), itr)
			writer.add_scalar('Loss/test_total_loss', test_res["loss"], itr)
			writer.add_scalar('MSE/test_raw', current_raw_mse, itr)
			writer.add_scalar('MSE/rmse_auc', test_res["rmse_auc"], itr)
			writer.flush()

		torch.save({'args': args, 'state_dict': model.state_dict()}, ckpt_path)

		# Late in training the metric wanders as much between adjacent checkpoints
		# of ONE run as it does between seeds, so a single fixed-epoch read-out is
		# noisy.  --ckpt-dense-from N switches to --ckpt-dense-every spacing past
		# epoch N, giving enough late checkpoints to average or take a median over.
		_stride = getattr(args, 'ckpt_every', 0)
		if getattr(args, 'ckpt_dense_from', 0) > 0 and epoch >= args.ckpt_dense_from:
			_stride = max(1, getattr(args, 'ckpt_dense_every', 0) or args.ckpt_every)
		if getattr(args, 'ckpt_every', 0) > 0 and epoch % _stride == 0:
			_traj_dir = os.path.join(os.path.dirname(ckpt_path), 'traj')
			os.makedirs(_traj_dir, exist_ok=True)
			_base = os.path.basename(ckpt_path).replace('.ckpt', '')
			torch.save({'args': args, 'state_dict': model.state_dict(), 'epoch': epoch},
			           os.path.join(_traj_dir, f'{_base}_ep{epoch:06d}.ckpt'))

		_min_itr = 2000 * num_batches if args.lr_per_epoch else 2000
		if early_stop_counter >= args.patience and itr > _min_itr:
			logger.info("Early stopping triggered.")
			break

	torch.save({'args': args, 'state_dict': model.state_dict()}, ckpt_path)
	if writer is not None:
		writer.close()

	logger.info(f"Training complete in {time.time() - start:.1f}s.")
	print(f"Last checkpoint : {ckpt_path}")
	if tag:
		print(f"Evaluate it with: python3 test_dose_cond.py --experiment {args.experiment} "
			  f"--data-dir {args.data_dir} --save {args.save} --tag '{tag}'")
	if best_smoothed_test_mse != float('inf'):
		print(f"Best checkpoint : {best_ckpt_path} (smoothed test MSE {best_smoothed_test_mse:.4f})")


if __name__ == '__main__':
	main()
