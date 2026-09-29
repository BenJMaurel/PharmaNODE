"""Where every seed's checkpoints and training logs live.

The seed set was built over several sessions and two historical names break
the pattern: narrow seed 2 is saved under results/seeds/s2 with log suffix
__s2, and wide seed 2's logs use __s2w.  Anything that iterates over seeds
should resolve paths here rather than re-deriving them.
"""
import glob
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# arm -> (cohort prefix, seeds trained)
ARMS = {
    "wide":   ("km",  (1, 2, 3, 4, 5, 6)),
    "narrow": ("kmn", (1, 2, 3, 4, 5, 6, 7, 8, 9)),
}
# MODELS is the pinned comparator set that check_reported.py and
# analyse_confound.py iterate over; adding to it would invalidate the pinned
# values, so the analytic-W2 retrain lives in ALL_MODELS instead and is opted
# into explicitly with --models.
MODELS = ("dc", "film")
ALL_MODELS = ("dc", "dcd", "film", "filmd", "w2", "wr0.548", "wr2.0")
MODEL_NAME = {"dc": "dose-cond", "dcd": "dose-cond (dense)",
              "film": "ours", "filmd": "classic FiLM (dense)",
              "w2": "ours + w2ana",
              "wr0.548": "w2ana + ridge 0.548", "wr2.0": "w2ana + ridge 2.0"}
# "filmd" is classic FiLM retrained with --ckpt-dense-* so it can be compared to the
# ridge arms on the same 11-point 5000-6000 window.  Identical flags otherwise; saved
# apart from results/seeds/ because the filename tag is the same.
# Explicit-ridge sweep arms live in results/ridge/, wide arm only.
RIDGE_LAMBDAS = ("0.548", "2.0")
# w2 checkpoints come from the --w2-analytic-moments retrain, saved outside
# results/seeds/ so no earlier run can be overwritten.
W2_ARMS = {"wide": (1, 2, 3, 4, 5, 6, 7, 8, 9), "narrow": (1, 2, 3, 4, 5, 6, 7, 8, 9)}
REGIMES = ("in", "lo", "hi")                  # in support, below range, above range
REGIME_NAME = {"in": "in support", "lo": "below range", "hi": "above range"}


def cohorts(arm):
    """(confounded, control) training cohorts of an arm."""
    p = ARMS[arm][0]
    return f"confound_{p}09", f"confound_{p}00"


def eval_cohort(cohort, regime):
    return cohort if regime == "in" else f"{cohort}_{regime}"


def seed_dir(arm, seed):
    return "s2" if (arm, seed) == ("narrow", 2) else f"s{seed}{arm}"


def log_suffix(arm, seed):
    if seed == 2:
        return {"wide": "s2w", "narrow": "s2"}[arm]
    return f"s{seed}{arm}"


def checkpoint(arm, seed, model, trained, epoch):
    """Repo-relative path of the fixed-epoch trajectory checkpoint.  Exactly one must exist."""
    # epoch="best" selects the run's validation-selected checkpoint instead of a
    # fixed-epoch trajectory dump.  Both families select on the same criterion
    # (--select-on mse_v2 in the wide arm), so the comparison stays matched.
    if model == "dcd":
        pat = os.path.join("results", "dosecond_dense", f"dc_s{seed}{arm}", "exp_dosecond_run",
                           trained, ("*_best.ckpt" if epoch == "best"
                                     else os.path.join("traj", f"*_ep{int(epoch):06d}.ckpt")))
        hits = sorted(glob.glob(os.path.join(REPO, pat)))
        if len(hits) != 1:
            raise FileNotFoundError(f"{arm} seed {seed} dcd {trained} ep{epoch}: expected "
                                    f"exactly one checkpoint matching {pat}, found {len(hits)}")
        return os.path.relpath(hits[0], REPO)
    if model == "filmd":
        pat = os.path.join("results", "classic_dense", f"classic_s{seed}{arm}", "exp_film_run",
                           trained, ("*_best.ckpt" if epoch == "best"
                                     else os.path.join("traj", f"*_ep{int(epoch):06d}.ckpt")))
        hits = sorted(glob.glob(os.path.join(REPO, pat)))
        if len(hits) != 1:
            raise FileNotFoundError(f"{arm} seed {seed} filmd {trained} ep{epoch}: expected "
                                    f"exactly one checkpoint matching {pat}, found {len(hits)}")
        return os.path.relpath(hits[0], REPO)
    if model.startswith("wr"):
        lam = model[2:]
        pat = os.path.join("results", "ridge", f"ridge_s{seed}{arm}_wr{lam}", "exp_film_run",
                           trained, ("*_best.ckpt" if epoch == "best"
                                     else os.path.join("traj", f"*_ep{int(epoch):06d}.ckpt")))
        hits = sorted(glob.glob(os.path.join(REPO, pat)))
        if len(hits) != 1:
            raise FileNotFoundError(f"{arm} seed {seed} {model} {trained} ep{epoch}: expected "
                                    f"exactly one checkpoint matching {pat}, found {len(hits)}")
        return os.path.relpath(hits[0], REPO)
    if model == "w2":
        pat = (os.path.join("results", "w2ana", f"w2ana_s{seed}{arm}", "exp_film_run", trained,
                            "*_w2ana_best.ckpt") if epoch == "best" else
               os.path.join("results", "w2ana", f"w2ana_s{seed}{arm}", "exp_film_run", trained,
                            "traj", f"*_w2ana_ep{int(epoch):06d}.ckpt"))
        hits = sorted(glob.glob(os.path.join(REPO, pat)))
        if len(hits) != 1:
            raise FileNotFoundError(f"{arm} seed {seed} w2 {trained} ep{epoch}: expected exactly "
                                    f"one checkpoint matching {pat}, found {len(hits)}")
        return os.path.relpath(hits[0], REPO)
    base = os.path.join("results", "seeds", seed_dir(arm, seed))
    if model == "dc":
        pat = (os.path.join(base, "exp_dosecond_run", trained, f"experiment_dosecond_{trained}*_best.ckpt")
               if epoch == "best" else
               os.path.join(base, "exp_dosecond_run", trained, "traj",
                            f"experiment_dosecond_{trained}_ep{int(epoch):06d}.ckpt"))
    else:
        pat = (os.path.join(base, "exp_film_run", trained, "*_best.ckpt") if epoch == "best"
               else os.path.join(base, "exp_film_run", trained, "traj", f"*_ep{int(epoch):06d}.ckpt"))
    hits = sorted(glob.glob(os.path.join(REPO, pat)))
    if len(hits) != 1:
        raise FileNotFoundError(f"{arm} seed {seed} {model} {trained} ep{epoch}: expected exactly "
                                f"one checkpoint matching {pat}, found {len(hits)}")
    return os.path.relpath(hits[0], REPO)


def train_log(arm, seed, model, trained):
    if model == "filmd":
        return os.path.join(REPO, "logs", f"run_models_{trained}__classic_s{seed}{arm}.log")
    if model.startswith("wr"):
        return os.path.join(REPO, "logs",
                            f"run_models_{trained}__ridge_s{seed}{arm}_wr{model[2:]}.log")
    if model == "w2":
        return os.path.join(REPO, "logs", f"run_models_{trained}__w2ana_s{seed}{arm}.log")
    script = "run_models" if model == "film" else "train_dose_cond"
    return os.path.join(REPO, "logs", f"{script}_{trained}__{log_suffix(arm, seed)}.log")
