#!/usr/bin/env python3
"""How nonlinear is a trained 50-unit decoder on the latent states it actually sees?
Runs encoder + latent ODE on test series, collects z(t) on the 100-point grid and the decoder output y(z),
then fits the best AFFINE map y ~ W z + b on those states. Reports R^2 of the affine fit, the residual RMS
(in the model's transformed output space and as a % of the output range), and tanh saturation
(share of hidden pre-activations with |a| > 2).
    decoder_linearity.py <plain|film> <ckpt> [n_batches]"""
import sys, numpy as np, torch
from torch.distributions.normal import Normal
sys.path.insert(0, '.')
import lib.utils as utils
from lib.read_tacro import set_static_hematocrit
from lib.create_latent_ode_model import create_LatentODE_model
kind, ck_path = sys.argv[1], sys.argv[2]; nb = int(sys.argv[3]) if len(sys.argv) > 3 else 2
dev = torch.device('cpu'); torch.manual_seed(0)
ck = torch.load(ck_path, map_location=dev, weights_only=False); args = ck['args']
for k, d in (('obsrv_std', None), ('static_dim', 3), ('n_train_series', None), ('encoder_dose_zero', False), ('blank_formulation', False)):
    if not hasattr(args, k): setattr(args, k, d)
set_static_hematocrit(int(args.static_dim) >= 4)
model = create_LatentODE_model(args, 1, Normal(torch.Tensor([0.]), torch.Tensor([1.])),
                               torch.Tensor([args.obsrv_std or 0.01]), dev, classif_per_tp=False, n_labels=1)
model.load_state_dict(ck['state_dict']); model.eval()
for k in ('encoder_dose_zero', 'blank_formulation', 'film_no_z0_cond'): setattr(model, k, getattr(args, k, False))
Z = []
with torch.no_grad():
    if kind == 'plain':
        from lib.parse_datasets import parse_datasets
        data = parse_datasets(args, dev)
        for _ in range(nb):
            b = utils.get_next_batch(data['test_dataloader'])
            tp = utils.linspace_vector(b['tp_to_predict'][0], torch.tensor(24.), 100)
            _, info = model.get_reconstruction(tp, b['observed_data'], b['observed_tp'], dose=b['dose'],
                                               static=b['static'], mask=b['observed_mask'], n_traj_samples=10)
            Z.append(info['latent_traj'].reshape(-1, info['latent_traj'].shape[-1]))
    else:
        from lib.read_tacro import extract_gen_tac_film, TacroFilmDataset, collate_fn_tacro_film
        from torch.utils.data import DataLoader
        e = args.experiment
        _, pin = extract_gen_tac_film(file_path=[f"results/exp_film_run/{e}/virtual_cohort_film_train.csv"])
        d, _ = extract_gen_tac_film(file_path=[f"results/exp_film_run/{e}/virtual_cohort_film_test.csv"], scale=pin)
        ld = DataLoader(TacroFilmDataset(d), batch_size=200, shuffle=False, collate_fn=lambda x: collate_fn_tacro_film(x, dev))
        for i, b in enumerate(ld):
            if i >= nb: break
            tp = utils.linspace_vector(b["tp_to_predict_v1"][0], torch.tensor(24.), 100)
            x, s_ = model._encoder_input(b["observed_data_v1"], b["dose_v1"], b["static_v1"])
            mu, std = model.encoder_z0(x, b["observed_tp_v1"], static=s_, run_backwards=True)
            z0 = utils.sample_standard_gaussian(mu.repeat(10, 1, 1), std.abs().repeat(10, 1, 1))
            sol = model.diffeq_solver(z0, tp); Z.append(sol.reshape(-1, sol.shape[-1]))
    Z = torch.cat(Z)
    y = model.decoder(Z).squeeze(-1).numpy()
    if getattr(model.decoder, 'residual', False):
        pre = model.decoder.mlp[0](Z).numpy()
        ylin = model.decoder.lin(Z).squeeze(-1).numpy(); ymlp = model.decoder.mlp(Z).squeeze(-1).numpy()
        print(f"   residual: std of linear branch {ylin.std():.4f} | std of MLP branch {ymlp.std():.4f} "
              f"(ratio {ymlp.std()/ylin.std():.2f})")
    else:
        pre = model.decoder.decoder[0](Z).numpy() if len(model.decoder.decoder) > 1 else None
Zn = Z.numpy(); X = np.hstack([Zn, np.ones((len(Zn), 1))])
coef, *_ = np.linalg.lstsq(X, y, rcond=None); res = y - X @ coef
r2 = 1 - res.var() / y.var()
print(f"{kind:5} {ck_path.split('/')[-1][:60]}")
print(f"   states {len(Zn)} | affine-fit R^2 {r2:.5f} | residual RMS {np.sqrt(np.mean(res**2)):.4f} "
      f"({100*np.sqrt(np.mean(res**2))/(y.max()-y.min()):.2f}% of output range) | "
      + (f"tanh |a|>2: {100*np.mean(np.abs(pre) > 2):.1f}% of hidden activations" if pre is not None else "linear decoder"))
