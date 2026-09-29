#!/usr/bin/env python3
"""Dose response of the Lu et al. port at ep6000: median predicted AUC across doses with the encoder
output frozen (first 300 held-out patients of confound_vc00_s4_pnoise, scaler pinned to the N=100
training cohort), and the elasticity d log AUC / d log dose between 1 and 8 mg, against the true
elasticity from the same patients' paired V1/V2 truth. Output: results/ndm/lu_dose_elasticity.txt"""
import os, sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch, numpy as np
from torch.utils.data import DataLoader
import lib.utils as utils
from lib.read_tacro import extract_gen_tac_film, TacroFilmDataset, collate_fn_tacro_film, set_static_hematocrit
from lib.lu_neural_pk import create_lu_pk_model
from test_dose_cond import unscale_data, calculate_auc
set_static_hematocrit(True)
import os  # env overrides (defaults = the unwindowed paper-noise runs); windowed: LU_T=..._win_n100 LU_B=..._win LU_ROOT=results/s4_pnoise_win LU_PFX=lu-
T = os.environ.get('LU_T', 'confound_vc00_s4_pnoise_n100'); B = os.environ.get('LU_B', 'confound_vc00_s4_pnoise')
ROOT = os.environ.get('LU_ROOT', 'results/s4_pnoise_lu'); PFX = os.environ.get('LU_PFX', '')
pin = extract_gen_tac_film(file_path=[f'results/exp_film_run/{T}/virtual_cohort_film_train.csv'])[1]
tr, te = (f'results/exp_film_run/{B}/virtual_cohort_film_{s}.csv' for s in ('train', 'test'))
d_all, sc = extract_gen_tac_film(file_path=[tr, te], scale=pin)
tid = set(extract_gen_tac_film(file_path=[tr], scale=pin)[0].keys())
b = next(iter(DataLoader(TacroFilmDataset({k: v for k, v in d_all.items() if k not in tid}), batch_size=300,
                         collate_fn=lambda x: collate_fn_tacro_film(x, torch.device('cpu')))))
dense = utils.linspace_vector(b['tp_to_predict_v1'][0], torch.tensor(24.0), 100)
DOSES = (0.25, 1, 2, 4, 8, 12)
for var, tag in (('faithful', '__sdim-4'), ('static', '__mode-counterfactual_static_sdim-4')):
    els, meds = [], []
    for s in (1, 2, 3):
        ck = torch.load(f'{ROOT}/{PFX}{var}_n100_s{s}/exp_lupk_run/{T}/traj/experiment_lupk_{T}{tag}_ep006000.ckpt',
                        weights_only=False)
        m = create_lu_pk_model(ck['args'], torch.device('cpu')); m.load_state_dict(ck['state_dict']); m.eval()
        with torch.no_grad():
            th, x0 = m.encode(b['observed_data_v1'], b['observed_tp_v1'], b['dose_v1'],
                              static=b['static_v1'] if m.encoder.n_static else None)
            itv = m.interval_from_static(b['static_v1'])
            auc = {}
            for mg in DOSES:
                p = unscale_data(m.simulate(th, torch.full_like(b['dose_v1'], mg / 8.0), dense, interval=itv, x0=x0).squeeze(-1), sc)
                p[b['static_v1'][:, 1].bool().numpy(), 50:] = 0
                auc[mg] = np.array(calculate_auc(p, dense.numpy(), b['static_v1']))
        els.append(np.median(np.log(auc[8] / auc[1]) / np.log(8)))
        meds.append([np.median(auc[k]) for k in DOSES])
    print(f"{var:9s} elasticity 1->8 mg per seed {' '.join(f'{e:.2f}' for e in els)} (mean {np.mean(els):.2f}); "
          f"median AUC at {DOSES} mg (mean of seeds): {' '.join(f'{x:.0f}' for x in np.mean(meds, 0))}")
t1, t2, d1, d2 = (b[k].numpy() for k in ('auc_red_v1', 'auc_red_v2', 'dose_v1', 'dose_v2'))
ok = np.abs(np.log(d2 / d1)) > 0.3
print(f"true      elasticity (paired V1/V2 truth, same 300 patients, |log dose ratio| > 0.3): "
      f"median {np.median(np.log(t2[ok] / t1[ok]) / np.log(d2[ok] / d1[ok])):.2f}")
