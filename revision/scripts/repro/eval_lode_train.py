#!/usr/bin/env python3
"""TRAINING-series variant of eval_lode.py (for the training-set recalibration factor).
Score a PLAIN latent ODE (run_models.py --latent-ode, PK_Tacro loader) on the AUC of the observed
visit, reproducing the committed test_model.py computation: prediction grid linspace(t0, 24, 100),
100 posterior samples, inverse Box-Cox, concentrations after index 50 zeroed for Prograf (12 h interval),
x max_out, AUC = trapezoid of the sample-mean curve; reference = true AUC. Unlike test_model.py it
scores ALL test batches (not only the first one) and keeps per-patient values.

    eval_lode.py <ckpt> <out.json>
Series ids are '<patient>_<visit>'; V1 and V2 are reported separately (both are factual here)."""
import sys, json, numpy as np, torch
from scipy.special import inv_boxcox
from torch.distributions.normal import Normal
sys.path.insert(0, '.')
import lib.utils as utils
try:
    from lib.read_tacro import set_static_hematocrit
except ImportError:                       # committed (paper-era) lib: no hematocrit channel
    set_static_hematocrit = lambda flag: None
from lib.parse_datasets import parse_datasets
from lib.create_latent_ode_model import create_LatentODE_model
ck_path, out = sys.argv[1], sys.argv[2]
dev = torch.device('cpu'); torch.manual_seed(0); np.random.seed(0)
ck = torch.load(ck_path, map_location=dev, weights_only=False); args = ck['args']
for _k, _d in (('obsrv_std', None), ('static_dim', 3), ('n_train_series', None), ('encoder_dose_zero', False), ('blank_formulation', False)):
    if not hasattr(args, _k): setattr(args, _k, _d)      # checkpoints written by older code
set_static_hematocrit(int(getattr(args, 'static_dim', 3)) >= 4)
data_obj = parse_datasets(args, dev)
obsrv_std = torch.Tensor([args.obsrv_std if args.obsrv_std is not None else 0.01]).to(dev)
z0_prior = Normal(torch.Tensor([0.0]).to(dev), torch.Tensor([1.]).to(dev))
model = create_LatentODE_model(args, data_obj['input_dim'], z0_prior, obsrv_std, dev, classif_per_tp=False, n_labels=1)
model.load_state_dict(ck['state_dict']); model.eval()
for k in ('encoder_dose_zero', 'blank_formulation'): setattr(model, k, getattr(args, k, False))
max_out = float(data_obj['max_out']['max_out'][0]); lam = float(data_obj['max_out']['best_lambda'][0])
ids, t_auc, p_auc = [], [], []
for _ in range(data_obj['n_train_batches']):
    b = utils.get_next_batch(data_obj['train_dataloader'])
    tp = utils.linspace_vector(b['tp_to_predict'][0], torch.tensor(24.), 100).to(dev)
    with torch.no_grad():
        rec, _ = model.get_reconstruction(tp, b['observed_data'], b['observed_tp'], dose=b['dose'],
                                          static=b['static'], mask=b['observed_mask'], n_traj_samples=100)
    r = np.nan_to_num(inv_boxcox(rec.numpy(), lam), nan=0.0)
    r[:, b['static'][:, 1].bool().numpy(), 50:, :] = 0
    pred = np.trapezoid(r.mean(0)[..., 0] * max_out, tp.numpy(), axis=-1)
    ids += [str(x) for x in b['patient_id']]; t_auc += (b['auc_red'].numpy().reshape(-1) * max_out).tolist(); p_auc += pred.tolist()
ids = np.array(ids); t_auc = np.array(t_auc); p_auc = np.array(p_auc)
res = {'ckpt': ck_path, 'n_series': int(len(ids))}
for v in ('1', '2'):
    m = np.array([(i.endswith('_' + v) if '_' in i else v == '1') for i in ids])
    if not m.any(): continue
    r = p_auc[m] / t_auc[m] - 1; a = np.abs(r); k = a <= np.quantile(a, 0.99)
    res[f'v{v}'] = dict(n=int(m.sum()), rmspe=100*float(np.sqrt(np.mean(r**2))), rmspe_trim1=100*float(np.sqrt(np.mean(r[k]**2))),
                        median_abs=100*float(np.median(a)), bias_median=100*float(np.median(r)), mpe=100*float(np.mean(r)),
                        n_gt100=int((a > 1).sum()))
res['per_series'] = {'id': ids.tolist(), 'true_auc': t_auc.tolist(), 'pred_auc': p_auc.tolist()}
json.dump(res, open(out, 'w'))
print(json.dumps({k: v for k, v in res.items() if k != 'per_series'}, indent=1))
