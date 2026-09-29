#!/usr/bin/env python3
"""Does the learned transport do more than proportional dose scaling?

The strongest form of the reviewer's "simple exposure baseline" is not a
trapezoid over sparse points -- it is to take the model's OWN visit-1 AUC
prediction and scale it by the dose ratio:

    AUC_2_hat = AUC_1_model * (d_2 / d_1)

If this matches the model's visit-2 prediction, the FiLM transport has learned
nothing beyond proportionality and could be replaced by a multiplication.  The
comparison uses the harness's own per-patient outputs, so the model column is
exactly the published number.
"""
import argparse, glob, json, os, subprocess, sys, tempfile
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib.read_tacro import extract_gen_tac_film, TacroFilmDataset, collate_fn_tacro_film  # noqa: E402

rmspe = lambda t, p: float(np.sqrt(np.mean(((t - p) / t) ** 2)) * 100)  # noqa: E731
mpe = lambda t, p: float(np.mean((t - p) / t) * 100)                     # noqa: E731

COHORTS = {
    "90000":          ("linear PK (sc. 2)",  "results/doselaw_dech50/90000/exp_film_run/90000/*_best.ckpt"),
    "93000":          ("Michaelis-Menten",   "results/doselaw_dech50/93000/exp_film_run/93000/*_best.ckpt"),
    "confound_km00":  ("wide, control",      "results/seeds/s1wide/exp_film_run/confound_km00/traj/*_ep003000.ckpt"),
    "confound_km09":  ("wide, confounded",   "results/seeds/s1wide/exp_film_run/confound_km09/traj/*_ep003000.ckpt"),
    "confound_kmn00": ("narrow, control",    "results/seeds/s1narrow/exp_film_run/confound_kmn00/traj/*_ep003000.ckpt"),
    "confound_kmn09": ("narrow, confounded", "results/seeds/s1narrow/exp_film_run/confound_kmn09/traj/*_ep003000.ckpt"),
}


def ratios(exp, data_dir):
    tr = os.path.join(data_dir, exp, "virtual_cohort_film_train.csv")
    te = os.path.join(data_dir, exp, "virtual_cohort_film_test.csv")
    allp, sc = extract_gen_tac_film(file_path=[tr, te])
    train_ids = set(extract_gen_tac_film(file_path=[tr])[0].keys())
    test = {k: v for k, v in allp.items() if k not in train_ids}
    b = next(iter(DataLoader(TacroFilmDataset(test), batch_size=4000, shuffle=False,
                             collate_fn=lambda x: collate_fn_tacro_film(x, torch.device("cpu")))))
    return (b["dose_v2"] / b["dose_v1"]).cpu().numpy(), (b["auc_red_v2"] * sc[0]).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="./results/exp_film_run")
    a = ap.parse_args()
    print(f"{'cohort':<17}{'setting':<21}{'our model':>12}{'our V1 x d2/d1':>17}{'ratio':>8}")
    print(f"{'':<38}{'V2 RMSPE':>12}{'V2 RMSPE':>17}{'':>8}")
    print("-" * 76)
    for exp, (label, pat) in COHORTS.items():
        ck = glob.glob(pat)
        if not ck:
            print(f"{exp:<17}{label:<21}  (no checkpoint)"); continue
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out = f.name
        subprocess.run([sys.executable, "test_film_matched.py", "--experiment", exp,
                        "--data-dir", a.data_dir, "--ckpt", ck[0], "--eval-split", "test",
                        "--out-json", out], capture_output=True, text=True)
        pp = json.load(open(out))["per_patient"]
        os.unlink(out)
        t2 = np.array(pp["true_auc_v2"]); p2 = np.array(pp["pred_auc_v2"]); p1 = np.array(pp["pred_auc_v1"])
        r, t2b = ratios(exp, a.data_dir)
        assert len(r) == len(t2) and np.allclose(np.sort(t2b), np.sort(t2), rtol=1e-3), f"{exp}: patient order mismatch"
        model, null = rmspe(t2, p2), rmspe(t2, p1 * r)
        print(f"{exp:<17}{label:<21}{model:>12.2f}{null:>17.2f}{null/model:>8.2f}x")


if __name__ == "__main__":
    main()
