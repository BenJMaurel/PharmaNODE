#!/usr/bin/env python3
"""Run an evaluation harness with every RNG seeded.

test_film_matched.py and test_dose_cond.py draw posterior samples from torch's
global RNG without seeding it, so evaluating one checkpoint twice gives
slightly different numbers (over 137 accidental repeat evaluations during the
revision: median 0.05, max 0.55 percentage points).  This shim seeds random,
numpy and torch, then executes the harness unchanged as __main__.  Nothing in
the harness is modified.

    python3 seeded_eval.py [--eval-seed N] <harness.py> [harness args ...]
"""
import os
import random
import runpy
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    argv = sys.argv[1:]
    seed = 0
    if argv[:1] == ["--eval-seed"]:
        seed, argv = int(argv[1]), argv[2:]
    if not argv:
        sys.exit(__doc__)
    script, args = argv[0], argv[1:]
    os.chdir(REPO)
    sys.path.insert(0, REPO)          # the harnesses import lib.* and each other
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    sys.argv = [script] + args
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
