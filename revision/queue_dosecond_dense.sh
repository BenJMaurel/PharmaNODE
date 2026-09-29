#!/bin/bash
# dose-cond retrained with dense late checkpoints.
#
# WHY: the headline confounding result (dose-cond DiD +9.46 vs ours +5.05) reads
# dose-cond at a SINGLE epoch, while the FiLM side now has an 11-point median.
# Today showed single-epoch reads on this metric move by up to 3.7 pp for reasons
# that have nothing to do with the model, and the spikes are one-sided (upward).
# So the headline gap could be partly an artefact of how dose-cond was measured.
# This puts both sides of that comparison on the same footing.
#
# Flags identical to the original dose-cond runs except --ckpt-dense-*, which only
# adds save points.  Saved under results/dosecond_dense/ -- the filename tag matches
# the originals, so writing into results/seeds/ would overwrite them.
set -u
cd "$(dirname "$0")"
SEEDS=${@:-"1 2 3 4 5 6"}
MAXJOBS=${MAXJOBS:-11}
LOG=results/dosecond_dense_queue.log
mkdir -p results/dosecond_dense
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
say () { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
busy () { pgrep -f 'run_models\.py|train_dose_cond\.py' | wc -l | tr -d ' '; }
say "dose-cond dense retrain: seeds $SEEDS, wide arm, 6000 epochs"
for s in $SEEDS; do
  for c in confound_km09 confound_km00; do
    tag="dc_s${s}wide"; R="results/dosecond_dense/${tag}"
    if ls "$R/exp_dosecond_run/$c/traj/"*_ep006000.ckpt >/dev/null 2>&1; then
      say "  skip seed $s $c (already done)"; continue
    fi
    while [ "$(busy)" -ge "$MAXJOBS" ]; do sleep 60; done
    mkdir -p "$R/exp_dosecond_run/$c"
    nohup python3 train_dose_cond.py --experiment "$c" --data-dir ./results/exp_film_run \
        --niters 6000 -b 512 -l 10 --lr 1e-2 --seed "$s" --patience 1000000 \
        --save "./$R/" --ckpt-every 1000 --ckpt-dense-from 5000 --ckpt-dense-every 100 \
        --log-suffix "__${tag}" >/dev/null 2>&1 &
    say "  launched seed $s $c (pid $!), jobs now $(busy)"
    sleep 3
  done
done
say "all queued"
wait
say "dose-cond dense retrain complete"
