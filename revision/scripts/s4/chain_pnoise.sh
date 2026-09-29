#!/bin/bash
# Paper-noise experiment (Benjamin 2026-09-21): is the EBE's advantage a low-noise artefact?
# After confound_vc00_s4_pnoise exists: N=100 subset, Monolix true-covariate + L1 fits with EBE,
# true-model EBE floor, and FiLM/dose-cond N=100 seeds 1-3 (main config, 6000 epochs) under the 800% cap.
cd "$(dirname "$0")/../.."
say () { echo "[$(date '+%F %T')] $*"; }
B=confound_vc00_s4_pnoise; PY=/opt/miniconda3/bin/python3.12
while pgrep -f "gen_tacro_confound.py --exp $B " >/dev/null || [ ! -f results/exp_film_run/$B/noise.json ]; do sleep 60; done
say "cohort $B ready"
$PY scripts/nsweep/make_subsets.py $B "100"
$PY - <<PY
import pandas as pd
a = set(pd.read_csv("results/exp_film_run/${B}_n100/virtual_cohort_film_train.csv").ID)
b = set(pd.read_csv("results/exp_film_run/confound_vc00_s4_n100/virtual_cohort_film_train.csv").ID)
print("same 100 training patients as the low-noise N=100:", a == b, len(a))
PY
$PY scripts/monolix/build_monolix_data.py ${B}_n100 results/monolix/data/${B}_n100.csv
COHORT=$B scripts/monolix/fit_and_ebe.sh 100 2 > logs/monolix_pnoise_n100.log 2>&1 &
COHORT=$B VARIANT=l1 scripts/monolix/fit_and_ebe.sh 100 2 > logs/monolix_l1_pnoise_n100.log 2>&1 &
say "Monolix true + L1 fits launched"
EBE_COHORT=$B nice -n 5 $PY scripts/monolix/ebe_popmodel.py true 1000 0 results/s4/ebe/popebe_true_pnoise_n1000.csv > logs/popebe_true_pnoise.log 2>&1 &
say "true-model EBE floor launched"
sleep 180
NSWEEP_ROOT=results/s4_pnoise NSWEEP_BASE=$B scripts/nsweep/sched_generic.sh 800 "1 2 3" "100" 6000
wait; say "chain done (networks still training)"
