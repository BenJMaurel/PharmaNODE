#!/usr/bin/env python3
"""Lu training failures: numerical (explicit-Euler instability) or optimisation (bad weights)?
Re-evaluates a checkpoint at the training step count (n_euler) and at 10x finer steps, same data path and
scaler pinning as test_lu_pk.py. If the error vanishes with finer steps the failure is the solver; if it is
unchanged the learned weights themselves are bad. Also reports the largest |dt * df/dx| eigenvalue proxy
(spectral norm of the state Jacobian times dt) along the V1 trajectories: > 2 means forward Euler is unstable.
Usage: euler_refine.py <ckpt> <experiment> <scale_from>"""
import sys, numpy as np, torch
sys.path.insert(0, '.')
from torch.utils.data import DataLoader
import lib.utils as utils
from lib.read_tacro import extract_gen_tac_film, TacroFilmDataset, collate_fn_tacro_film, set_static_hematocrit
from lib.lu_neural_pk import create_lu_pk_model
from test_dose_cond import unscale_data, calculate_auc
torch.set_num_threads(1)
ck, exp, sf = sys.argv[1:4]
C = torch.load(ck, map_location='cpu', weights_only=False); A = C['args']
set_static_hematocrit(int(getattr(A, 'static_dim', 3)) >= 4)
d = f'results/exp_film_run'
pin = extract_gen_tac_film(file_path=[f'{d}/{sf}/virtual_cohort_film_train.csv'])[1]
allp, sc = extract_gen_tac_film(file_path=[f'{d}/{exp}/virtual_cohort_film_train.csv', f'{d}/{exp}/virtual_cohort_film_test.csv'], scale=pin)
tr = set(extract_gen_tac_film(file_path=[f'{d}/{exp}/virtual_cohort_film_train.csv'], scale=pin)[0].keys())
ev = {k: v for k, v in allp.items() if k not in tr}
dl = DataLoader(TacroFilmDataset(ev), batch_size=2000, shuffle=False, collate_fn=lambda x: collate_fn_tacro_film(x, 'cpu'))
m = create_lu_pk_model(A, 'cpu'); m.load_state_dict(C['state_dict']); m.eval()
n0 = m.n_euler
out = []
for n in (n0, 10 * n0):
    m.n_euler = n; t1, p1, t2, p2 = [], [], [], []; nonfinite = 0
    with torch.no_grad():
        for b in dl:
            tp = utils.linspace_vector(b["tp_to_predict_v1"][0], torch.tensor(24.0), 100)
            q1, q2, _ = m.predict_counterfactual(conc_v1=b["observed_data_v1"], times_v1=b["observed_tp_v1"],
                dose_v1=b["dose_v1"], dose_v2=b["dose_v2"], tp_v1=tp, tp_v2=tp, static=b["static_v1"])
            nonfinite += int((~torch.isfinite(q1)).sum() + (~torch.isfinite(q2)).sum())
            q1 = unscale_data(q1.squeeze(-1), sc); q2 = unscale_data(q2.squeeze(-1), sc)
            pg = b["static_v1"][:, 1].bool().numpy(); q1[pg, 50:] = 0; q2[pg, 50:] = 0
            t1 += list((b["auc_red_v1"] * sc[0]).numpy()); t2 += list((b["auc_red_v2"] * sc[0]).numpy())
            p1 += list(calculate_auc(q1, tp.numpy(), b["static_v1"])); p2 += list(calculate_auc(q2, tp.numpy(), b["static_v1"]))
    e = lambda t, p: 100 * np.median(np.abs(np.array(p) / np.array(t) - 1))
    b_ = lambda t, p: 100 * np.median(np.array(p) / np.array(t) - 1)
    out.append(f"n_euler {n:5d}: V1 med|err| {e(t1,p1):6.1f} (bias {b_(t1,p1):+6.1f})  V2 med|err| {e(t2,p2):6.1f} (bias {b_(t2,p2):+6.1f})  non-finite {nonfinite}")
print(f"{ck.split('/')[-1][-40:]}  [{exp}, n={len(t1)}]"); print("  " + "\n  ".join(out))
