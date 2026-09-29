###########################
# Priority 4 -- head-to-head against Lu et al.'s neural-PK submodule: TRAINING
#
# Standalone. Reuses only the dataset loader from train_dose_cond.py so both
# baselines see byte-identical data, and writes to its own folder,
#   <save>/exp_lupk_run/<exp>/experiment_lupk_<id>[_best].ckpt
# so no previously reported result can be touched.
###########################

import os
import sys
import time
import argparse

import numpy as np
import torch
import torch.optim as optim

import lib.utils as utils
from lib.lu_neural_pk import create_lu_pk_model
from train_dose_cond import load_data
from lib.read_tacro import set_static_hematocrit


def build_parser():
	p = argparse.ArgumentParser("Lu et al. neural-PK baseline (revision Priority 4)")
	# --- experiment / data (same interface as train_dose_cond.py) ---
	p.add_argument('--experiment', type=str, required=True)
	p.add_argument('--data-dir', type=str, default='./results/exp_film_run')
	p.add_argument('--save', type=str, default='./results/')
	p.add_argument('--load', type=str, default=None)
	p.add_argument('--consistent-scaling', action='store_true')
	# --- optimisation (their procedure: plain L2, standard optimiser) ---
	p.add_argument('--niters', type=int, default=6000, help="Number of epochs.")
	p.add_argument('--lr', type=float, default=1e-3)
	p.add_argument('-b', '--batch-size', type=int, default=512)
	p.add_argument('--seed', type=int, default=42)
	p.add_argument('--patience', type=int, default=10000)
	p.add_argument('--smoothing_factor', type=float, default=0.99)
	p.add_argument('--weight-decay', type=float, default=0.0)
	# --- architecture ---
	p.add_argument('--param-dim', type=int, default=8,
		help="Width of the static per-patient parameter vector the encoder emits.")
	p.add_argument('--enc-hidden', type=int, default=64)
	p.add_argument('--enc-layers', type=int, default=1)
	p.add_argument('--dyn-hidden', type=int, default=64)
	p.add_argument('--dyn-layers', type=int, default=2)
	p.add_argument('--n-euler', type=int, default=240,
		help="Forward-Euler steps across the prediction window.")
	p.add_argument('--readout', type=str, default='identity', choices=['identity', 'linear'],
		help="'identity' reads the concentration coordinate directly, as in their design.")
	p.add_argument('--init', type=str, default='history', choices=['history', 'encoder', 'zero'],
		help="Where the simulated trajectory starts. 'history' (default) simulates the "
			 "full run-in of prior doses from zero, so the state entering the observation "
			 "window is the model's own steady state. 'encoder' lets the encoder set the "
			 "initial state. 'zero' starts from zero at t=0 with a single dose -- "
			 "mis-specified for this steady-state dataset, kept only to reproduce "
			 "earlier runs.")
	p.add_argument('--n-prior-doses', type=int, default=6,
		help="Number of prior administrations in the run-in (nbr_ss in the generator).")
	p.add_argument('--use-static', action='store_true',
		help="Also give the encoder the static covariates (dose, formulation, CYP3A5) "
			 "that our model receives. OFF by default, because their encoder reads only "
			 "the four dynamic features -- turn it on as a fairness check.")
	# --- training protocol ---
	p.add_argument('--train-mode', type=str, default='reconstruct',
		choices=['reconstruct', 'counterfactual'],
		help="'reconstruct' is their original procedure: each visit is encoded from its "
			 "own observations and reconstructed at its own dose, and the V1->V2 "
			 "counterfactual is a purely test-time operation. 'counterfactual' also "
			 "trains the cross terms, so the comparison cannot be accused of "
			 "withholding the target task.")
	p.add_argument('--select-on', type=str, default='counterfactual',
		choices=['counterfactual', 'reconstruction'],
		help="Metric used to pick the best checkpoint. Default matches what "
			 "run_models.py and train_dose_cond.py do, so every row of the table gets "
			 "the same treatment.")
	p.add_argument('--tensorboard', action='store_true')
	# --- opt-in additions (scenario-4 benchmark, 2026-09-23); defaults reproduce earlier runs ---
	p.add_argument('--static-dim', type=int, default=3,
		help="Static channels: 3 = [dose, formulation, CYP]; 4 adds hematocrit/35 (scenario 4). "
			 "Only reaches the encoder with --use-static.")
	p.add_argument('--eval-every', type=int, default=10,
		help="Epochs between in-training test evaluations (log line, _best selection). That "
			 "evaluation refits the test scaler (handoff landmine 1.1) and is for monitoring only.")
	p.add_argument('--ckpt-every', type=int, default=0,
		help="If >0, save a fixed-epoch checkpoint every N epochs into a 'traj/' subfolder. "
			 "Report these, not _best (which is selected on test data).")
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


LUPK_TAG_SPEC = [
	('train_mode', 'reconstruct', 'mode'),
	('use_static', False, lambda v: 'static'),
	('init', 'history', 'init'),
	('readout', 'identity', 'readout'),
	('select_on', 'counterfactual', 'sel'),
	('param_dim', 8, 'pdim'),
	('enc_hidden', 64, 'ench'),
	('dyn_hidden', 64, 'dynh'),
	('consistent_scaling', False, lambda v: 'jointscale'),
	('static_dim', 3, 'sdim'),   # appended last: existing file names are unchanged
]


def evaluate(model, data_obj, args):
	keys = ["loss", "mse_v1", "mse_v2", "mse_counterfactual", "rmse_auc"]
	totals = {k: 0.0 for k in keys}
	n = data_obj["n_test_batches"]
	for _ in range(n):
		batch_dict = utils.get_next_batch_film(data_obj["test_dataloader"])
		res = model.compute_l2_losses(batch_dict, train_mode=args.train_mode,
			max_out=data_obj["max_out"])
		for k in keys:
			v = res[k]
			totals[k] += float(v.item() if torch.is_tensor(v) else v)
	return {k: v / n for k, v in totals.items()}


def main():
	args = build_parser().parse_args()
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	experimentID = args.load if args.load is not None else args.experiment

	ckpt_dir = os.path.join(args.save, "exp_lupk_run", str(args.experiment))
	utils.makedirs(ckpt_dir)
	tag = config_tag(args, LUPK_TAG_SPEC)
	ckpt_path = os.path.join(ckpt_dir, f"experiment_lupk_{experimentID}{tag}.ckpt")
	best_ckpt_path = ckpt_path.replace(".ckpt", "_best.ckpt")

	utils.makedirs("logs/")
	logger = utils.get_logger(
		logpath=os.path.join("logs", f"train_lu_pk_{experimentID}{tag}.log"),
		filepath=os.path.abspath(__file__))
	logger.info(" ".join(sys.argv))

	# hematocrit channel is opt-in, exactly as in run_models.py / train_dose_cond.py
	set_static_hematocrit(int(args.static_dim) >= 4)
	data_obj = load_data(args, device)
	logger.info(f"train batches: {data_obj['n_train_batches']} | test batches: {data_obj['n_test_batches']}")

	model = create_lu_pk_model(args, device)
	n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	logger.info(f"Lu et al. neural-PK | param_dim={args.param_dim} | n_euler={args.n_euler} "
				f"| readout={args.readout} | init={args.init} | use_static={args.use_static} "
				f"| train_mode={args.train_mode} | trainable params={n_params}")

	if args.load is not None:
		checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
		model.load_state_dict(checkpoint['state_dict'])

	optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	num_batches = data_obj["n_train_batches"]

	writer = None
	if args.tensorboard:
		from torch.utils.tensorboard import SummaryWriter
		writer = SummaryWriter(os.path.join(args.save, "runs_lupk", str(experimentID)))

	smoothed = float('inf')
	best_smoothed = float('inf')
	early_stop_counter = 0
	n_iters_to_viz = args.eval_every
	start = time.time()

	for itr in range(1, num_batches * (args.niters + 1)):
		optimizer.zero_grad()
		batch_dict = utils.get_next_batch_film(data_obj["train_dataloader"])
		train_res = model.compute_l2_losses(batch_dict, train_mode=args.train_mode,
			max_out=data_obj["max_out"], diagnostics=False)
		train_res["loss"].backward()
		optimizer.step()

		if itr % num_batches == 0 and args.ckpt_every > 0 and (itr // num_batches) % args.ckpt_every == 0:
			_traj_dir = os.path.join(ckpt_dir, 'traj')
			os.makedirs(_traj_dir, exist_ok=True)
			_base = os.path.basename(ckpt_path).replace('.ckpt', '')
			torch.save({'args': args, 'state_dict': model.state_dict(), 'epoch': itr // num_batches},
					   os.path.join(_traj_dir, f'{_base}_ep{itr // num_batches:06d}.ckpt'))

		if itr % (n_iters_to_viz * num_batches) != 0:
			continue

		epoch = itr // num_batches
		with torch.no_grad():
			test_res = evaluate(model, data_obj, args)

		logger.info(
			'Epoch {:04d} [LuPK Test] | L2 {:.6f} | V1 recon MSE {:.6f} | V2 recon MSE {:.6f} '
			'| V1->V2 counterfactual MSE {:.6f} | rmse_auc {:.6f}'.format(
				epoch, test_res["loss"], test_res["mse_v1"], test_res["mse_v2"],
				test_res["mse_counterfactual"], test_res["rmse_auc"]))
		logger.info("Train loss (one batch): {}".format(float(train_res["loss"].detach())))

		if args.select_on == 'counterfactual':
			current = test_res["mse_counterfactual"]
		else:
			current = 0.5 * (test_res["mse_v1"] + test_res["mse_v2"])
		smoothed = current if smoothed == float('inf') else (
			current * args.smoothing_factor + smoothed * (1 - args.smoothing_factor))
		logger.info(f"Selection metric ({args.select_on}) raw {current:.4f} | smoothed {smoothed:.4f}")

		if smoothed < best_smoothed:
			best_smoothed = smoothed
			early_stop_counter = 0
			logger.info(f"New best smoothed metric: {best_smoothed:.4f}. Saving to {best_ckpt_path}")
			torch.save({'args': args, 'state_dict': model.state_dict(), 'epoch': epoch,
						'raw_metric_at_best': current, 'smoothed_metric': best_smoothed},
					   best_ckpt_path)
		else:
			early_stop_counter += 1
			logger.info(f"No improvement. Early stopping counter: {early_stop_counter}/{args.patience}")

		if writer is not None:
			writer.add_scalar('Loss/train_l2', float(train_res["loss"]), itr)
			writer.add_scalar('Loss/test_l2', test_res["loss"], itr)
			writer.add_scalar('MSE/counterfactual', test_res["mse_counterfactual"], itr)
			writer.add_scalar('MSE/rmse_auc', test_res["rmse_auc"], itr)
			writer.flush()

		torch.save({'args': args, 'state_dict': model.state_dict()}, ckpt_path)

		if early_stop_counter >= args.patience and itr > 2000:
			logger.info("Early stopping triggered.")
			break

	torch.save({'args': args, 'state_dict': model.state_dict()}, ckpt_path)
	if writer is not None:
		writer.close()

	logger.info(f"Training complete in {time.time() - start:.1f}s.")
	print(f"Last checkpoint : {ckpt_path}")
	if tag:
		print(f"Evaluate it with: python3 test_lu_pk.py --experiment {args.experiment} "
			  f"--data-dir {args.data_dir} --save {args.save} --tag '{tag}'")
	if best_smoothed != float('inf'):
		print(f"Best checkpoint : {best_ckpt_path} (smoothed {args.select_on} {best_smoothed:.4f})")


if __name__ == '__main__':
	main()
