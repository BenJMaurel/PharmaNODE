#!/usr/bin/env python3
"""Visual check: can you see the OT-FiLM vs dose-cond difference in the curves?

Reproduces the harness prediction path exactly (test_film_matched.py /
test_dose_cond.py): hematocrit channel set from the checkpoint's static_dim, scaler
pinned to the TRAINING cohort, test split only, posterior mean over 100 draws,
per-patient nRMSE = RMSE / mean(truth) at the 12 target times against the noisy
observed Visit-2 curve.  Writes per-patient arrays to an .npz so the figure can be
restyled without re-running the models.

    curves_s4.py <root> <epoch6> <seed> <test_cohort> <out.npz>
"""
import sys, os, numpy as np, torch
sys.path.insert(0, '.')
from torch.utils.data import DataLoader
import lib.utils as utils
from lib.read_tacro import extract_gen_tac_film, TacroFilmDataset, collate_fn_tacro_film, set_static_hematocrit
from test_dose_cond import unscale_data, calculate_auc
from probe_entanglement import load_film_model
from lib.dose_conditioned import create_dose_conditioned_model

root, ep, seed, test_cohort, out = sys.argv[1:6]
TRAIN = "confound_vc00_s4"
dev = torch.device("cpu")
torch.manual_seed(0); np.random.seed(0)

ck = {"film": f"{root}/film_s{seed}/exp_film_run/{TRAIN}/traj/experiment_film_{TRAIN}__noz0_sig0.05_sc141.27_dech50_sel-mse_v2_ep{ep}.ckpt",
      "dc":   f"{root}/dc_s{seed}/exp_dosecond_run/{TRAIN}/traj/experiment_dosecond_{TRAIN}__sig-0.05_ep{ep}.ckpt"}
sd = {a: int(getattr(torch.load(p, map_location='cpu', weights_only=False)['args'], 'static_dim', 3)) for a, p in ck.items()}
assert sd["film"] == sd["dc"], sd
set_static_hematocrit(sd["film"] >= 4)

R = "results/exp_film_run"
pin = extract_gen_tac_film(file_path=[f"{R}/{TRAIN}/virtual_cohort_film_train.csv"])[1]
data_all, scaler = extract_gen_tac_film(file_path=[f"{R}/{test_cohort}/virtual_cohort_film_train.csv",
                                                   f"{R}/{test_cohort}/virtual_cohort_film_test.csv"], scale=pin)
tr_ids = set(extract_gen_tac_film(file_path=[f"{R}/{test_cohort}/virtual_cohort_film_train.csv"], scale=pin)[0].keys())
ev = {k: v for k, v in data_all.items() if k not in tr_ids}
ids = list(ev.keys())
b = next(iter(DataLoader(TacroFilmDataset(ev), batch_size=4000, shuffle=False,
                         collate_fn=lambda x: collate_fn_tacro_film(x, dev))))
dense = utils.linspace_vector(b["tp_to_predict_v1"][0], torch.tensor(24.0), 100)

def run(arch):
    if arch == "film":
        m, _ = load_film_model(ck[arch], dev)
        f = lambda tp2: m.get_reconstruction_extrapolation(
            data_v1=b["observed_data_v1"], time_steps_v1=b["observed_tp_v1"], time_steps_v2=tp2,
            dose_v1=b["dose_v1"], dose_v2=b["dose_v2"], time_steps_to_predict_v1=tp2,
            static_v1=b["static_v1"], delta_t=b["delta_t"], t_v1=b["t_v1"], n_traj_samples=100)[0]
    else:
        c = torch.load(ck[arch], map_location='cpu', weights_only=False); a = c['args']
        m = create_dose_conditioned_model(a, 1, torch.distributions.Normal(torch.tensor([0.]), torch.tensor([1.])),
                                          torch.Tensor([getattr(a, 'noise_weight', 0.01)]), dev)
        m.load_state_dict(c['state_dict']); m.eval()
        f = lambda tp2: m.get_reconstruction_counterfactual(
            data_v1=b["observed_data_v1"], time_steps_v1=b["observed_tp_v1"], dose_v1=b["dose_v1"],
            dose_v2=b["dose_v2"], tp_to_predict_v1=tp2, tp_to_predict_v2=tp2,
            static_v1=b["static_v1"], n_traj_samples=100)[0]
    with torch.no_grad():
        dense_pred = unscale_data(f(dense).mean(0).squeeze(-1), scaler)
        sparse_pred = unscale_data(f(b["tp_to_predict_v2"]).mean(0).squeeze(-1), scaler)
    return np.asarray(dense_pred), np.asarray(sparse_pred)

y = np.asarray(unscale_data(b["data_to_predict_v2"].squeeze(-1).clone(), scaler))
res = {}
for arch in ("film", "dc"):
    d, s = run(arch)
    res[f"{arch}_dense"], res[f"{arch}_sparse"] = d, s
    res[f"{arch}_nrmse"] = np.sqrt(np.mean((s - y) ** 2, axis=1)) / np.mean(y, axis=1)
    print(f"{arch}: mean per-patient nRMSE {100*res[f'{arch}_nrmse'].mean():.2f}%")
np.savez(out, ids=np.array(ids), y=y, tp=np.asarray(b["tp_to_predict_v2"]), dense_tp=np.asarray(dense),
         obs_v1=np.asarray(unscale_data(b["observed_data_v1"].squeeze(-1).clone(), scaler)),
         obs_tp_v1=np.asarray(b["observed_tp_v1"]), d1=np.asarray(b["dose_v1"]).ravel() * 8,
         d2=np.asarray(b["dose_v2"]).ravel() * 8, prograf=np.asarray(b["static_v1"][:, 1]).astype(bool), **res)
print("wrote", out)
