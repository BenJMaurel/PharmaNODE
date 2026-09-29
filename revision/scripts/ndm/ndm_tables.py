#!/usr/bin/env python3
"""Numbers for the npj Digital Medicine manuscript (ndm.tex).

Every value quoted in ndm.tex is printed by this script from the per-patient arrays of the
existing eval JSONs / EBE CSVs (landmine 1.3: no raw AUC RMSPE as a headline).

Scenario-4 control cohort, same 1000 held-out test patients (physiology) throughout -- NB the lo/hi cohorts redraw d1 for 85% of them (handoff section 10) -- networks at a fixed
epoch (ep6000, traj/ checkpoints, scaler pinned to the training cohort by the harness).
Relative error r = pred/true - 1 on the AUC.

Cell metrics
  med   median |r| (%)                     bias  median r (%)
  w20   % of patients with |r| <= 20 %      trm   RMSPE after dropping the worst 1 % (%)
  n100  number of patients with |r| > 100 % nrm   curve nRMSE (%), MEAN over patients
  cov   empirical coverage of the nominal 95 % predictive interval (networks, V2 only)

Usage:  python scripts/ndm/ndm_tables.py  > results/ndm/tables.txt
"""
import os, json, glob
import numpy as np, pandas as pd

R = "results"
RNG = np.random.default_rng(0)


def metrics(t, p):
    t, p = np.asarray(t, float), np.asarray(p, float)
    r = p / t - 1
    a = np.abs(r)
    keep = a <= np.quantile(a, 0.99)
    return dict(med=100 * np.median(a), bias=100 * np.median(r), w20=100 * np.mean(a <= 0.20),
                trm=100 * np.sqrt(np.mean(r[keep] ** 2)), n100=int((a > 1).sum()), abs=a)


def net_cell(path, visit):
    if not os.path.exists(path):
        return None
    j = json.load(open(path))
    pp = j["per_patient"]
    m = metrics(pp[f"true_auc_{visit}"], pp[f"pred_auc_{visit}"])
    m["nrm"] = j[visit]["nrmse_pct"]
    m["cov"] = (100 * j["calibration_v2"]["intervals"]["0.95"]["coverage"]
                if visit == "v2" and "calibration_v2" in j else np.nan)
    return m


def lode_cell(path):
    """plain latent ODE evaluator (scripts/repro/eval_lode.py): V1 only, first 1000 series."""
    if not os.path.exists(path):
        return None
    j = json.load(open(path))
    ps = j["per_series"]
    v1 = np.array([i.endswith("_1") for i in ps["id"]])   # ids '<patient>_<visit>', visits interleaved
    assert v1.sum() == j["v1"]["n"]
    m = metrics(np.array(ps["true_auc"])[v1], np.array(ps["pred_auc"])[v1])
    assert abs(m["med"] - j["v1"]["median_abs"]) < 1e-6, "does not reproduce eval_lode.py's own V1 median"
    m["nrm"], m["cov"] = np.nan, np.nan
    return m


def ebe_cell(tag, reg, n=1000):
    f = f"{R}/s4/ebe/popebe_{tag}_n{n}.csv"
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    if reg == "v1":
        t, p, nr = d.auc_true_v1, d.auc_v1, d.nrmse_v1
    else:
        if f"auc_{reg}" not in d:
            return None
        t, p, nr = d[f"auc_true_{reg}"], d[f"auc_{reg}"], d[f"nrmse_{reg}"]
    m = metrics(t, p)
    m["nrm"], m["cov"] = 100 * nr.mean(), np.nan
    return m


def agg(cells):
    cells = [c for c in cells if c is not None]
    if not cells:
        return None
    out = {"n": len(cells)}
    for k in ("med", "bias", "w20", "trm", "n100", "nrm", "cov"):
        v = np.array([c[k] for c in cells], float)
        out[k], out[k + "_sd"], out[k + "_seeds"] = np.nanmean(v), (np.nanstd(v, ddof=1) if len(v) > 1 else 0.0), v
    out["abs_mean_over_seeds"] = np.mean([c["abs"] for c in cells], axis=0)
    return out


def fmt(a, show_cov=False):
    if a is None:
        return "   (not available)"
    s = (f"med {a['med']:5.1f}±{a['med_sd']:3.1f}  bias {a['bias']:+6.1f}  w20 {a['w20']:5.1f}  "
         f"trm {a['trm']:6.1f}  n>100% {a['n100']:5.1f}  nRMSE {a['nrm']:6.1f}")
    if show_cov and not np.isnan(a["cov"]):
        s += f"  cov95 {a['cov']:4.1f}"
    s += f"   [n={a['n']}; med per seed {' '.join(f'{x:.1f}' for x in a['med_seeds'])}]"
    return s


def paired(a, b, label):
    """patient-level comparison of seed-averaged |error|, a vs b (same 1000 patients)."""
    if a is None or b is None:
        return f"  {label}: not available"
    da, db = a["abs_mean_over_seeds"], b["abs_mean_over_seeds"]
    diff = 100 * (np.median(da) - np.median(db))
    boots = []
    n = len(da)
    for _ in range(2000):
        i = RNG.integers(0, n, n)
        boots.append(100 * (np.median(da[i]) - np.median(db[i])))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    frac = 100 * np.mean(da < db)
    try:
        from scipy.stats import wilcoxon
        pv = wilcoxon(da, db).pvalue
    except Exception:
        pv = np.nan
    seedd = a["med_seeds"] - b["med_seeds"] if len(a["med_seeds"]) == len(b["med_seeds"]) else None
    s = (f"  {label}: diff of medians of seed-averaged |err| {diff:+.2f} pt "
         f"[95% patient-bootstrap {lo:+.2f}, {hi:+.2f}], first closer for {frac:.1f}% of patients, "
         f"Wilcoxon p = {pv:.2g}")
    if seedd is not None:
        s += f"; per-seed diff of medians {' '.join(f'{x:+.1f}' for x in seedd)}"
    return s


REGS = [("v1", "in", "V1 factual (observed dose)"), ("v2", "in", "V2 counterfactual, in range 1-8 mg"),
        ("v2", "hi", "V2 counterfactual, above range 10-12 mg"), ("v2", "lo", "V2 counterfactual, below range 0.25-0.5 mg")]

# ---------------- paper-level noise, N = 100 training patients -------------------------------
PN = {
    "OT-FiLM residual decoder (proposed)": f"{R}/s4_pnoise_decres/eval/ep006000_film_n100_s{{s}}_on_{{reg}}.json",
    "dose-cond residual decoder": f"{R}/s4_pnoise_decres/eval_dc/ep006000_dc_n100_s{{s}}_on_{{reg}}.json",
    # scripts/s4/chain_lu_pnoise.sh; faithful = Lu et al.'s procedure (dynamic features only, reconstruct loss),
    # static = + covariates (dose, formulation, CYP, HT) in the encoder + cross-visit (counterfactual) loss
    "Lu et al. neural-PK, faithful": f"{R}/s4_pnoise_lu/eval/ep006000_lu-faithful_n100_s{{s}}_on_{{reg}}.json",
    "Lu et al. neural-PK, +covariates +cross-visit": f"{R}/s4_pnoise_lu/eval/ep006000_lu-static_n100_s{{s}}_on_{{reg}}.json",
    "ablation: OT-FiLM 50-unit tanh decoder": f"{R}/s4_pnoise/eval/ep006000_film_n100_s{{s}}_on_{{reg}}.json",
    "ablation: OT-FiLM linear decoder": f"{R}/s4_pnoise_dech0/eval/ep006000_film_n100_s{{s}}_on_{{reg}}.json",
    "ablation: dose-cond linear decoder (default)": f"{R}/s4_pnoise/eval/ep006000_dc_n100_s{{s}}_on_{{reg}}.json",
    "ablation: OT-FiLM factual-only objective (dech50)": f"{R}/s4_pnoise_factonly/eval/ep006000_film_n100_s{{s}}_on_{{reg}}.json",
    "ablation: OT-FiLM sigma 0.01 (dech50)": f"{R}/s4_pnoise_sig001/eval/ep006000_film_n100_s{{s}}_on_{{reg}}.json",
    "ablation: OT-FiLM sigma 0.217 (dech50)": f"{R}/s4_pnoise_sig0217/eval/ep006000_film_n100_s{{s}}_on_{{reg}}.json",
}
PN_EBE = [("reference: EBE, true simulator model", "true_pnoise"),
          ("reference: NLME fitted (Monolix, true structure+covariates), N=100", "mlx_pnoise_n100"),
          ("reference: NLME fitted (Monolix, L1 covariates), N=100", "mlx_l1_pnoise_n100")]

# ---------------- paper-level noise, WINDOWED cohort (handoff 1.12), N = 100 -------------------
# scripts/s4/chain_table1_window.sh: visit-1 Cavg restricted to 3-25 ng/mL, 1100 test patients, lo/hi keep visit 1
PW = {
    "OT-FiLM residual decoder (proposed)": f"{R}/s4_pnoise_win/eval/ep006000_film_n100_s{{s}}_on_{{reg}}.json",
    "dose-cond residual decoder": f"{R}/s4_pnoise_win/eval/ep006000_dc_n100_s{{s}}_on_{{reg}}.json",
    "Lu et al. neural-PK, faithful": f"{R}/s4_pnoise_win/eval/ep006000_lu-faithful_n100_s{{s}}_on_{{reg}}.json",
    "Lu et al. neural-PK, +covariates +cross-visit": f"{R}/s4_pnoise_win/eval/ep006000_lu-static_n100_s{{s}}_on_{{reg}}.json",
}
# all 1100 test patients (the _n1000 files covered 1000 of them), scripts/s4/refs_window_n1100.sh
PW_EBE = [("reference: EBE, true simulator model", "true_pnoise_win", 1100),
          ("reference: NLME fitted (Monolix, true structure+covariates), N=100", "mlx_pnoise_win_n100", 1100),
          ("reference: NLME fitted (Monolix, L1 covariates), N=100", "mlx_l1_pnoise_win_n100", 1100)]

# ---------------- low noise, N = 800 training patients ---------------------------------------
LN = {
    "OT-FiLM residual decoder (proposed)": f"{R}/s4_decres/eval/ep006000_film_s{{s}}_vc00_on_vc00{{sfx}}.json",
    "dose-cond residual decoder": f"{R}/s4_decres/eval_dc/ep006000_dc_s{{s}}_on_{{reg}}.json",  # chain_dc_decres.sh naming
    "Lu et al. neural-PK, faithful": f"{R}/s4_lu/eval/ep006000_lu-faithful_s{{s}}_on_{{reg}}.json",   # not run
    "Lu et al. neural-PK, +covariates +cross-visit": f"{R}/s4_lu/eval/ep006000_lu-static_s{{s}}_on_{{reg}}.json",
    "ablation: OT-FiLM 50-unit tanh decoder": f"{R}/s4/eval/ep006000_film_s{{s}}_vc00_on_vc00{{sfx}}.json",
    "ablation: OT-FiLM linear decoder": f"{R}/s4_dech0/eval/ep006000_film_s{{s}}_vc00_on_vc00{{sfx}}.json",
    "ablation: dose-cond linear decoder (default)": f"{R}/s4/eval/ep006000_dc_s{{s}}_vc00_on_vc00{{sfx}}.json",
}
LN_EBE = [("reference: EBE, true simulator model", "true"),
          ("reference: NLME fitted (Monolix, true structure+covariates), N=800", "mlx_n800")]
SFX = {"in": "", "hi": "hi", "lo": "lo"}


def run_block(title, table, ebes, seeds, path_kw):
    print("=" * 110)
    print(title)
    res = {}
    for visit, reg, lab in REGS:
        print(f"\n-- {lab}")
        for name, pat in table.items():
            cells = [net_cell(pat.format(s=s, reg=reg, sfx=SFX[reg]), visit) for s in seeds]
            a = agg(cells)
            res[(name, visit, reg)] = a
            print(f"  {name:55s} {fmt(a, show_cov=(visit == 'v2'))}")
        if path_kw == "pn" and visit == "v1":
            for name, root in (("ablation: plain latent ODE, linear decoder (factual only)", "lode_s4pn"),
                               ("ablation: plain latent ODE, 50-unit decoder (factual only)", "lode_s4pn_dech50")):
                a = agg([lode_cell(f"{R}/{root}/eval/s{s}_final.json") for s in (1, 2, 3)])
                res[(name, visit, reg)] = a
                print(f"  {name:55s} {fmt(a)}")
        for name, tag, *n in ebes:
            e = ebe_cell(tag, "v1" if visit == "v1" else reg, *n)
            a = agg([e]) if e is not None else None
            res[(name, visit, reg)] = a
            print(f"  {name:55s} {fmt(a)}")
    print("\n-- paired, patient level (proposed vs competitors)")
    for visit, reg, lab in REGS:
        a = res.get(("OT-FiLM residual decoder (proposed)", visit, reg))
        for comp in ("dose-cond residual decoder", "Lu et al. neural-PK, faithful",
                     "Lu et al. neural-PK, +covariates +cross-visit"):
            print(paired(a, res.get((comp, visit, reg)), f"{lab} | proposed vs {comp}"))
    return res


def recal_block(title="paper noise N=100, residual decoders", cohort="confound_vc00_s4_pnoise_n100", table=None,
                runs=(("film", "OT-FiLM residual decoder (proposed)", "paper_decres/film"),
                      ("dc", "dose-cond residual decoder", "paper_decres/dc"),
                      ("faithful", "Lu et al. neural-PK, faithful", "paper_lu/faithful"),
                      ("static", "Lu et al. neural-PK, +covariates +cross-visit", "paper_lu/static"))):
    """Training-set level recalibration (same rule as scripts/s4/recal_tables.py): c = 1/median(pred/true)
    of V1 AUC over the run's own 100 training patients; every test prediction is multiplied by c."""
    table = PN if table is None else table
    print("=" * 110)
    print(f"TRAINING-SET RECALIBRATION, {title} (median |AUC err| %, before -> after)")
    auc_tr = np.sort(pd.read_csv(f"{R}/exp_film_run/{cohort}/virtual_cohort_film_train.csv")
                     .query("VISIT == 1").groupby("ID").AUC.first().values)
    for arch, name, rpfx in runs:
        pat = table[name]
        cs, rows = [], {k: ([], []) for k in ("V1", "V2 in", "V2 hi", "V2 lo")}
        for s in (1, 2, 3):
            fa = f"{R}/recal/{rpfx}_s{s}_all.json"
            if not os.path.exists(fa):
                continue
            pa = json.load(open(fa))["per_patient"]
            tt, pp = np.array(pa["true_auc_v1"]), np.array(pa["pred_auc_v1"])
            assert np.allclose(np.sort(tt[:100]), auc_tr, rtol=1e-3), f"{fa}: first 100 rows are not the training set"
            c = 1.0 / np.median(pp[:100] / tt[:100])
            cs.append(c)
            for k, reg, vis in (("V1", "in", "v1"), ("V2 in", "in", "v2"), ("V2 hi", "hi", "v2"), ("V2 lo", "lo", "v2")):
                f = pat.format(s=s, reg=reg)
                if not os.path.exists(f):
                    continue
                j = json.load(open(f))["per_patient"]
                t, p = np.array(j[f"true_auc_{vis}"]), np.array(j[f"pred_auc_{vis}"])
                rows[k][0].append(100 * np.median(np.abs(p / t - 1)))
                rows[k][1].append(100 * np.median(np.abs(c * p / t - 1)))
        if not cs:
            print(f"  {arch:8s} (not available)")
            continue
        print(f"  {arch:8s} factors c = {' '.join(f'{c:.3f}' for c in cs)}")
        for k, (b, a) in rows.items():
            if b:
                print(f"     {k:6s} {np.mean(b):6.1f} -> {np.mean(a):6.1f}   (after, per seed {' '.join(f'{x:.1f}' for x in a)})")


if __name__ == "__main__":
    run_block("PAPER-LEVEL RESIDUAL NOISE (0.71 + 0.113 C), N = 100 training patients, 3 seeds, ep6000",
              PN, PN_EBE, (1, 2, 3), "pn")
    run_block("LOW RESIDUAL NOISE (0.03 + 0.03 C), N = 800 training patients, 4 seeds, ep6000",
              LN, LN_EBE, (1, 2, 3, 4), "ln")
    recal_block()
    run_block("PAPER-LEVEL NOISE, WINDOWED COHORT (visit-1 Cavg 3-25 ng/mL; 1100 test patients), N = 100, 3 seeds, ep6000",
              PW, PW_EBE, (1, 2, 3), "pw")
    recal_block("WINDOWED paper noise N=100, residual decoders", "confound_vc00_s4_pnoise_win_n100", PW,
                (("film", "OT-FiLM residual decoder (proposed)", "paper_win/film"),
                 ("dc", "dose-cond residual decoder", "paper_win/dc"),
                 ("faithful", "Lu et al. neural-PK, faithful", "paper_win/lu-faithful"),
                 ("static", "Lu et al. neural-PK, +covariates +cross-visit", "paper_win/lu-static")))
