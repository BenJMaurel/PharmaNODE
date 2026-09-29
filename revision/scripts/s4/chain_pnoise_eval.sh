#!/bin/bash
# Follow-up of chain_pnoise.sh: score the paper-noise N=100 networks (seeds 1-3) at ep6000 in range as soon
# as each seed is ready; once the _lo/_hi cohorts exist, redo the three EBEs with all regimes
# (deterministic: in-range columns unchanged) and add the lo/hi network scores (eval skips existing JSONs).
cd "$(dirname "$0")/../.."
say () { echo "[$(date '+%F %T')] $*"; }
B=confound_vc00_s4_pnoise; PY=/opt/miniconda3/bin/python3.12
F=__noz0_sig0.05_sc141.27_dech50_sel-mse_v2_ep006000.ckpt; D=__sig-0.05_ep006000.ckpt; T=${B}_n100
for s in 1 2 3; do
  while [ ! -f results/s4_pnoise/film_n100_s$s/exp_film_run/$T/traj/experiment_film_$T$F ] || \
        [ ! -f results/s4_pnoise/dc_n100_s$s/exp_dosecond_run/$T/traj/experiment_dosecond_$T$D ]; do sleep 120; done
  say "seed $s at ep6000; scoring (regimes that exist)"
  NSWEEP_ROOT=results/s4_pnoise NSWEEP_BASE=$B scripts/nsweep/eval_nsweep.sh 100 $s 006000 3
done
say "IN_RANGE_NETWORKS_DONE"
while [ ! -f results/exp_film_run/${B}_lo/noise.json ] || [ ! -f results/exp_film_run/${B}_hi/noise.json ]; do sleep 60; done
say "lo/hi cohorts ready; rerunning EBEs with all regimes"
EBE_COHORT=$B $PY scripts/monolix/ebe_popmodel.py true 1000 0 results/s4/ebe/popebe_true_pnoise_n1000.csv > logs/popebe_true_pnoise.log 2>&1 &
EBE_COHORT=$B $PY scripts/monolix/ebe_popmodel.py results/monolix/fit_pnoise_n100/estimates.json 1000 0 results/s4/ebe/popebe_mlx_pnoise_n100_n1000.csv > logs/popebe_mlx_pnoise_n100.log 2>&1 &
EBE_COHORT=$B $PY scripts/monolix/ebe_popmodel.py results/monolix/fit_l1_pnoise_n100/estimates.json 1000 0 results/s4/ebe/popebe_mlx_l1_pnoise_n100_n1000.csv > logs/popebe_mlx_l1_pnoise_n100.log 2>&1 &
for s in 1 2 3; do NSWEEP_ROOT=results/s4_pnoise NSWEEP_BASE=$B scripts/nsweep/eval_nsweep.sh 100 $s 006000 2; done
wait; say "ALL_PNOISE_EVAL_DONE"
