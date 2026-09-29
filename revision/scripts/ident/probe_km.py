"""Can the learned latent predict Km?

probe_physiology.py only scores the parameters the generator writes to disk, and Km is
not one of them. Km and Vmax are replayed here from the generator's seed, then probed
the same way: cross-validated regression from the Visit-1 posterior mean.

Every target is reported against a raw-observation baseline (the same regression run on
the 3 concentrations the encoder is given). The latent only deserves credit for the
GAIN over that baseline -- hematocrit, for instance, is handed to the encoder in the
static vector, so a high score there proves nothing.

  python3 probe_km.py --cohort confound_vc00_s4 --scenario 4 --seeds "2 3 4"
"""
import argparse, random, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader

sys.path.insert(0, '/Users/benjaminmaurel/Documents/PharmaNODE')
from lib.read_tacro import (extract_gen_tac_film, TacroFilmDataset,
                            collate_fn_tacro_film, set_static_hematocrit)
from probe_physiology import get_representation, regression_probe
from gen_tacro_film import TacrolimusPK, POPULATION_PARAMS as PP


def replay(n, scenario, seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    rows = []
    for pid in range(1, n + 1):
        form = random.choice(['Prograf', 'Advagraf'])
        cyp = random.choice(['expresser', 'non_expresser'])
        ht = random.uniform(25.0, 45.0) if scenario in (2, 4) else 35.0
        pk = TacrolimusPK(formulation=form, hematocrit=ht, distribution_type='log_normal',
                          cyp_status=cyp, scenario=scenario)
        pk._sample_individual_parameters()
        r = {k: float(v) for k, v in pk.individual_params.items()}
        r.update(ID=pid, HT=ht)
        rows.append(r)
    return pd.DataFrame(rows).set_index('ID')


class NS:                      # minimal stand-in for the probe's argparse namespace
    def __init__(self, **kw): self.__dict__.update(kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', default='confound_vc00_s4')
    ap.add_argument('--scenario', type=int, default=4)
    ap.add_argument('--seeds', default='2 3 4')
    ap.add_argument('--static-dim', type=int, default=4)
    ap.add_argument('--root', default='results/s4')
    ap.add_argument('--film-tag', default='__noz0_sig0.05_sc141.27_dech50_sel-mse_v2_ep006000')
    ap.add_argument('--dc-tag', default='__sig-0.05_ep006000')
    a = ap.parse_args()

    set_static_hematocrit(a.static_dim >= 4)
    base = f'/Users/benjaminmaurel/Documents/PharmaNODE/results/exp_film_run/{a.cohort}'
    tr, te = f'{base}/virtual_cohort_film_train.csv', f'{base}/virtual_cohort_film_test.csv'
    data, _ = extract_gen_tac_film(file_path=[tr, te])
    train_ids = set(extract_gen_tac_film(file_path=[tr])[0].keys())
    data = {k: v for k, v in data.items() if k not in train_ids}      # held-out only
    loader = DataLoader(TacroFilmDataset(data), batch_size=8000, shuffle=False,
                        collate_fn=lambda x: collate_fn_tacro_film(x, torch.device('cpu')))
    batch = next(iter(loader))
    pids = [int(x) for x in batch['patient_ids']]

    n_tot = len(pd.read_csv(f'{base}/confound_truth.csv'))
    T = replay(n_tot, a.scenario)
    tv_km = PP['theta_Km'] * ((T.HT / 35.0) ** PP['theta_Km_HT']) if a.scenario == 4 else PP['theta_Km']
    targets = {
        'log Km':        np.log(T.loc[pids, 'Km'].values),
        'eta_Km':        np.log(T.loc[pids, 'Km'].values) - np.log(np.asarray(tv_km)[[p - 1 for p in pids]]),
        'log Vmax':      np.log(T.loc[pids, 'Vmax'].values),
        'log Vc':        np.log(T.loc[pids, 'Vc'].values),
        'HT (given)':    T.loc[pids, 'HT'].values,
        'log CL (inert)': np.log(T.loc[pids, 'CL'].values),
    }

    raw = batch['observed_data_v1'].reshape(len(pids), -1).numpy()
    print(f'cohort {a.cohort} (scenario {a.scenario}) | {len(pids)} held-out patients | static_dim={a.static_dim}')
    print('cross-validated R^2 from the Visit-1 posterior mean; "raw" = same probe on the 3 observations\n')

    for arch, mdl, sub, tag in (('FiLM', 'film', 'exp_film_run', a.film_tag),
                                ('dose-cond', 'dosecond', 'exp_dosecond_run', a.dc_tag)):
        acc = {k: [] for k in targets}
        rawacc = {k: [] for k in targets}
        for s in a.seeds.split():
            pre = 'film' if mdl == 'film' else 'dc'
            ck = (f'/Users/benjaminmaurel/Documents/PharmaNODE/{a.root}/{pre}_s{s}/{sub}/'
                  f'{a.cohort}/traj/experiment_{"film" if mdl=="film" else "dosecond"}_{a.cohort}{tag}.ckpt')
            cli = NS(ckpt=ck, model=mdl, representation='z_base', save='', experiment=a.cohort, tag='')
            with torch.no_grad():
                X, _ = get_representation(cli, batch, torch.device('cpu'))
            for k, y in targets.items():
                acc[k].append(regression_probe(X, y, n_splits=5, seed=0)['best_r2'])
        for k, y in targets.items():
            rawacc[k] = [regression_probe(raw, y, n_splits=5, seed=0)['best_r2']]
        print(f'--- {arch} (z_base, {X.shape[1]}-dim, seeds {a.seeds}) ---')
        print(f"{'target':16s} {'latent R2':>16s} {'raw-obs R2':>11s} {'gain':>8s}")
        for k in targets:
            v = np.array(acc[k]); r = rawacc[k][0]
            print(f'{k:16s} {v.mean():7.3f}+-{v.std(ddof=1):5.3f} {r:11.3f} {v.mean()-r:+8.3f}')
        print()


if __name__ == '__main__':
    main()
