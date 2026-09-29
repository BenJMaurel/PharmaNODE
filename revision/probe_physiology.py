###########################
# Physiological-decodability probe
#
# A latent-space quality criterion that does NOT go through the decoder: how well
# does the learned representation linearly encode the patient's TRUE generating PK
# parameters? A latent that recovers the physiology is better structured in a sense
# independent of downstream predictive accuracy -- which is why equal RMSPE between
# two models says nothing about this.
#
# Targets recorded per patient by the generator:
#   K_12, K_21  distribution rate constants  -- used by the dynamics in every scenario
#   HT          hematocrit                   -- enters CL (sc.2) and Vmax (sc.3)
#   K_ELIM      = CL/Vc                      -- NEGATIVE CONTROL under scenario 3:
#               CL is sampled but never used there (elimination is Michaelis-Menten),
#               so K_ELIM is unidentifiable from the data and every model should fail
#               on it. If a probe "recovers" it, the probe is leaking.
#   CYP, formulation -- SANITY CHECKS only: both are handed to the encoder in the
#               static vector, so high scores are expected and prove nothing.
#
# Purely additive.
###########################

import os, sys, json, argparse
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
import torch
from torch.utils.data import DataLoader

import lib.utils as utils
from lib.read_tacro import extract_gen_tac_film, TacroFilmDataset, collate_fn_tacro_film
from probe_entanglement import (load_film_model, encode_mu, film_transport,
                                regression_probe, classification_probe)

CONT = ['K_12', 'K_21', 'HT', 'K_ELIM']
DISC = ['CYP', 'ST']
NOTE = {'K_ELIM': 'partly identifiable (CL unused in sc.3, but Vc is)',
        'CYP': 'sanity check (given to the encoder)',
        'ST': 'sanity check (given to the encoder)'}


def build_parser():
    p = argparse.ArgumentParser("Probe how well a latent encodes true PK physiology")
    p.add_argument('--experiment', type=str, required=True)
    p.add_argument('--model', type=str, required=True, choices=['film', 'dosecond', 'lupk'])
    p.add_argument('--representation', type=str, default='z_base',
                   choices=['z_base', 'z_new', 'theta'],
                   help="'z_base' = Visit-1 posterior mean (film/dosecond); "
                        "'z_new' = FiLM-transported state; 'theta' = Lu's static vector.")
    p.add_argument('--tag', type=str, default='')
    p.add_argument('--ckpt', type=str, default=None)
    p.add_argument('--save', type=str, default='./results/')
    p.add_argument('--data-dir', type=str, default='./results/exp_film_run')
    p.add_argument('--eval-split', type=str, default='all', choices=['test', 'all'])
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out-json', type=str, default=None)
    return p


def default_ckpt(cli):
    d = {'film': ('exp_film_run', 'experiment_film'),
         'dosecond': ('exp_dosecond_run', 'experiment_dosecond'),
         'lupk': ('exp_lupk_run', 'experiment_lupk')}[cli.model]
    return os.path.join(cli.save, d[0], str(cli.experiment),
                        f"{d[1]}_{cli.experiment}{cli.tag}_best.ckpt")


def get_representation(cli, batch, device):
    path = cli.ckpt or default_ckpt(cli)
    if not os.path.exists(path):
        print(f"Checkpoint not found: {path}", file=sys.stderr); sys.exit(1)
    if cli.model == 'film':
        m, _ = load_film_model(path, device)
        z = encode_mu(m, batch["observed_data_v1"], batch["observed_tp_v1"],
                      batch["dose_v1"], batch["static_v1"])
        if cli.representation == 'z_new':
            z = film_transport(m, z, batch["dose_v1"], batch["dose_v2"],
                               batch["delta_t"], batch["t_v1"])
        return z.detach().cpu().numpy(), path
    ck = torch.load(path, map_location=device, weights_only=False)
    if cli.model == 'dosecond':
        from lib.dose_conditioned import create_dose_conditioned_model
        prior = torch.distributions.Normal(torch.Tensor([0.]).to(device), torch.Tensor([1.]).to(device))
        m = create_dose_conditioned_model(ck['args'], 1, prior,
                torch.Tensor([getattr(ck['args'], 'noise_weight', .01)]).to(device), device)
        m.load_state_dict(ck['state_dict']); m.eval()
        mu, _ = m.encode(batch["observed_data_v1"], batch["observed_tp_v1"],
                         batch["dose_v1"], static=batch["static_v1"])
        return mu.squeeze(0).detach().cpu().numpy(), path
    from lib.lu_neural_pk import create_lu_pk_model
    m = create_lu_pk_model(ck['args'], device)
    m.load_state_dict(ck['state_dict']); m.eval()
    static = batch["static_v1"] if m.encoder.n_static else None
    theta, _ = m.encode(batch["observed_data_v1"], batch["observed_tp_v1"],
                        batch["dose_v1"], static=static)
    return theta.detach().cpu().numpy(), path


def main():
    cli = build_parser().parse_args()
    device = torch.device("cpu")
    tr = os.path.join(cli.data_dir, cli.experiment, "virtual_cohort_film_train.csv")
    te = tr.replace('train', 'test')
    data, _ = extract_gen_tac_film(file_path=[tr, te])
    if cli.eval_split == 'test':
        train_ids = set(extract_gen_tac_film(file_path=[tr])[0].keys())
        data = {k: v for k, v in data.items() if k not in train_ids}
    loader = DataLoader(TacroFilmDataset(data), batch_size=8000, shuffle=False,
                        collate_fn=lambda x: collate_fn_tacro_film(x, device))
    batch = next(iter(loader))
    pids = [int(x) for x in batch["patient_ids"]]

    raw = pd.concat([pd.read_csv(tr), pd.read_csv(te)])
    truth = raw.groupby('ID')[CONT + DISC].first()
    Y = {t: truth.loc[pids, t].values.astype(float) for t in CONT + DISC}

    with torch.no_grad():
        X, path = get_representation(cli, batch, device)
    raw_obs = batch["observed_data_v1"].reshape(len(pids), -1).detach().cpu().numpy()

    label = cli.label if hasattr(cli, 'label') else f"{cli.model}:{cli.representation}"
    print(f"\n{label}  ({X.shape[1]}-dim, {len(pids)} patients)   ckpt: {os.path.basename(path)}")
    print(f"{'target':<10}{'probe R2':>10}{'raw-data R2':>13}{'gain':>8}   note")
    print('-' * 72)
    rows = []
    for t in CONT:
        if np.std(Y[t]) < 1e-12:
            print(f"{t:<10}{'constant in this scenario -- not identifiable, skipped':>44}")
            rows.append({'target': t, 'skipped': 'constant'})
            continue
        r = regression_probe(X, Y[t], n_splits=cli.folds, seed=cli.seed)['best_r2']
        rr = regression_probe(raw_obs, Y[t], n_splits=cli.folds, seed=cli.seed)['best_r2']
        rows.append({'target': t, 'r2': r, 'raw_r2': rr, 'gain': r - rr})
        print(f"{t:<10}{r:>10.3f}{rr:>13.3f}{r-rr:>+8.3f}   {NOTE.get(t,'')}")
    for t in DISC:
        c = classification_probe(X, Y[t], n_splits=cli.folds, seed=cli.seed)
        rows.append({'target': t, 'bal_acc': c['balanced_accuracy'], 'chance': c['chance']})
        print(f"{t:<10}{c['balanced_accuracy']:>10.3f}{'(chance '+format(c['chance'],'.2f')+')':>13}"
              f"{'':>8}   {NOTE.get(t,'')}")
    print('-' * 72)
    print("'gain' = how much the latent adds over probing the raw Visit-1 observations.")

    out = {'model': cli.model, 'representation': cli.representation,
           'experiment': cli.experiment, 'checkpoint': path,
           'dim': int(X.shape[1]), 'n_patients': len(pids), 'rows': rows}
    if cli.out_json is None:
        cli.out_json = os.path.join('results', 'revision', str(cli.experiment),
                                    f'physio_{cli.model}_{cli.representation}{cli.tag}.json')
    utils.makedirs(os.path.dirname(cli.out_json) or '.')
    json.dump(out, open(cli.out_json, 'w'), indent=2)
    print(f"written to {cli.out_json}")


if __name__ == '__main__':
    main()
