"""How much of the visit-2 error is an information limit rather than a model limit?

E[AUC2 | inputs] is the Bayes-optimal point predictor under squared error. We estimate it
directly by regression on the 8000 training patients (where the true AUC is known) and
score on the same 2000 held-out patients the networks are scored on. Each feature set is
a different amount of information, so the comparison localises the bottleneck:

  A population    dose + formulation + CYP only, no patient measurement
  B encoder       + the 3 concentrations the encoder actually receives (t = 0, 1, 3 h)
  C full curve    + all 12 visit-1 concentrations out to 24 h (same single dose)
  D physiology    + the individual's true k_elim, k_12, k_21, Vc

If B is far above C, the 3-point window is the bottleneck -> observe more timepoints.
If B is close to C, one dose is the bottleneck however densely you sample it.
D is what perfect inference of the linear PK parameters would buy; whatever remains is
carried by the saturation parameters (Km, Vmax), which a single dose cannot identify.
"""
import argparse, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

OBS_T = [0., 0.33, 0.67, 1., 1.5, 2., 3., 4., 6., 9., 12., 24.]
ENC_T = [0., 1., 3.]


def build(cohort, root='/Users/benjaminmaurel/Documents/PharmaNODE/results/exp_film_run'):
    truth = pd.read_csv(f'{root}/{cohort}/confound_truth.csv')
    frames = []
    for split, f in (('train', 'virtual_cohort_film_train.csv'), ('test', 'virtual_cohort_film_test.csv')):
        d = pd.read_csv(f'{root}/{cohort}/{f}'); d['__split'] = split
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df['DVn'] = pd.to_numeric(df['DV'], errors='coerce')
    df['TIMEn'] = pd.to_numeric(df['TIME'], errors='coerce')
    obs = df[df.DVn.notna()].copy()

    rows = []
    for (pid, visit), g in obs.groupby(['ID', 'VISIT']):
        g = g.sort_values('TIMEn')
        t = g.TIMEn.values - g.TIMEn.values[0]
        v = g.DVn.values
        rec = {'ID': pid, 'VISIT': visit, 'split': g['__split'].iloc[0],
               'AUC': float(g.AUC.iloc[0]), 'DRUG': 1 if g.DRUG.iloc[0] == 'Prograf' else 0,
               'CYP': int(g.CYP.iloc[0]),
               'k_elim': float(g.K_ELIM.iloc[0]), 'k_12': float(g.K_12.iloc[0]), 'k_21': float(g.K_21.iloc[0])}
        for j, tt in enumerate(OBS_T):
            rec[f'c{j}'] = float(v[np.argmin(np.abs(t - tt))])
        rows.append(rec)
    wide = pd.DataFrame(rows)

    v1 = wide[wide.VISIT == 1].set_index('ID')
    v2 = wide[wide.VISIT == 2].set_index('ID')
    tr = truth.set_index('ID')
    ids = sorted(set(v1.index) & set(v2.index) & set(tr.index))
    out = pd.DataFrame(index=ids)
    out['split'] = v1.loc[ids, 'split']
    out['AUC1'] = v1.loc[ids, 'AUC']; out['AUC2'] = v2.loc[ids, 'AUC']
    out['d1'] = tr.loc[ids, 'd1']; out['d2'] = tr.loc[ids, 'd2']
    out['DRUG'] = v1.loc[ids, 'DRUG']; out['CYP'] = v1.loc[ids, 'CYP']
    for c in ('k_elim', 'k_12', 'k_21'):
        out[c] = v1.loc[ids, c]
    out['Vc'] = tr.loc[ids, 'confound_par'] if (tr.confound_param == 'Vc').all() else np.nan
    for j in range(len(OBS_T)):
        out[f'c{j}'] = v1.loc[ids, f'c{j}']
    # true per-individual parameters, replayed from the generator's seed (seed=0)
    tp = pd.read_csv('/tmp/true_params_10k.csv').set_index('ID')
    for c in ('Ktr', 'Vmax', 'Km'):
        out[c] = tp.loc[ids, c]
    return out


ENC_IDX = [OBS_T.index(t) for t in ENC_T]
FEATS = {
    'A population': ['d1', 'd2', 'DRUG', 'CYP'],
    'B encoder (3 obs)': ['d1', 'd2', 'DRUG', 'CYP'] + [f'c{j}' for j in ENC_IDX],
    'C full v1 curve (12 obs)': ['d1', 'd2', 'DRUG', 'CYP'] + [f'c{j}' for j in range(len(OBS_T))],
    'D + true linear PK': ['d1', 'd2', 'DRUG', 'CYP'] + [f'c{j}' for j in range(len(OBS_T))]
                           + ['k_elim', 'k_12', 'k_21', 'Vc'],
    'E + true Km, Vmax too': ['d1', 'd2', 'DRUG', 'CYP'] + [f'c{j}' for j in range(len(OBS_T))]
                           + ['k_elim', 'k_12', 'k_21', 'Vc', 'Ktr', 'Vmax', 'Km'],
    'F 3 obs + true Km,Vmax': ['d1', 'd2', 'DRUG', 'CYP'] + [f'c{j}' for j in ENC_IDX]
                           + ['Vmax', 'Km'],
}


def rmspe(t, p): return float(np.sqrt(np.mean(((t - p) / t) ** 2)) * 100)
def mpe(t, p):   return float(np.mean((t - p) / t) * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', default='confound_vc00_10k')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    d = build(a.cohort)
    tr = d[d.split == 'train']; te = d[d.split == 'test']
    print(f'cohort {a.cohort}: {len(tr)} train / {len(te)} test patients\n')
    print(f"{'information given':28s} {'V1 AUC RMSPE':>13s} {'V2 AUC RMSPE':>13s} {'V2 MPE':>8s}")
    print('-' * 66)
    for lab, cols in FEATS.items():
        res = {}
        for tgt in ('AUC1', 'AUC2'):
            m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                              random_state=a.seed)
            # fit in log space: the metric is relative, so relative error is what to minimise
            m.fit(tr[cols].values, np.log(tr[tgt].values))
            res[tgt] = np.exp(m.predict(te[cols].values))
        print(f'{lab:28s} {rmspe(te.AUC1.values, res["AUC1"]):13.2f} '
              f'{rmspe(te.AUC2.values, res["AUC2"]):13.2f} {mpe(te.AUC2.values, res["AUC2"]):8.2f}')

    print('\nfor reference, the trained networks on this same test split (control-trained):')
    print(f"{'  dose-cond':28s} {4.03:13.2f} {16.13:13.2f}")
    print(f"{'  ours/FiLM':28s} {8.44:13.2f} {19.06:13.2f}")


if __name__ == '__main__':
    main()
