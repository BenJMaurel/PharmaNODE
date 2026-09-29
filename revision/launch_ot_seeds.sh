#!/bin/bash
# Three more seeds of the OT ablation (--film-self-consistency 0) on the two
# confounding cohorts. Their with-OT counterparts are results/seeds/s{4,5,6}wide,
# which exist, so this gives a paired comparison at n=6.
# Waits for the running stage to drain first.
set -u
cd /Users/benjaminmaurel/Documents/PharmaNODE
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
LOG=results/revision_chain.log
say () { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
say "OT ablation seeds 4-6: waiting for the machine to drain"
while [ "$(pgrep -f 'run_models\.py|train_dose_cond\.py' | wc -l | tr -d ' ')" -gt 0 ]; do sleep 120; done
say "machine free; launching 6 runs"
for c in confound_km09 confound_km00; do
  for s in 4 5 6; do
    R="results/ablation_noOT/${c}_s${s}"
    [ -d "$R" ] && { say "  $R exists, skipping"; continue; }
    mkdir -p "$R/exp_film_run/$c"
    nohup python3 run_models.py --niters 6000 -n 200 -s 40 -l 10 --dataset PK_Tacro --latent-ode \
        --use_film --noise-weight 0.01 --max-t 5. -b 512 --seed "$s" --experiment "$c" \
        --film-no-z0-cond --obsrv-std 0.217 --film-self-consistency 0 --decoder-hidden 50 \
        --patience 1000000 --save "./$R/" --ckpt-every 1000 --log-suffix "__noOT_${c}_s${s}" \
        >/dev/null 2>&1 &
    say "  launched $c seed $s (pid $!)"
  done
done
wait
say "OT ablation seeds 4-6 complete"
