#!/usr/bin/env python3
"""Run the paper's (committed, unseeded) gen_tacro.py with fixed random seeds, without editing it.
Must be run from inside a worktree of the paper commit.  gen_seeded.py <data_seed> <gen_tacro args...>"""
import sys, random, runpy
import numpy as np, torch
sys.path.insert(0, ".")          # as "python gen_tacro.py" would: the worktree's paper-era lib
seed = int(sys.argv[1])
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
sys.argv = ['gen_tacro.py'] + sys.argv[2:]
runpy.run_path('gen_tacro.py', run_name='__main__')
