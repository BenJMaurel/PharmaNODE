#!/bin/bash
# Queue further w2ana seeds, launching each cell only as a core frees so the
# machine stays at MAXJOBS without oversubscribing.
#   ./queue_w2ana_seeds.sh 4 5 6 7 8 9
set -u
cd /Users/benjaminmaurel/Documents/PharmaNODE
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
MAXJOBS=${MAXJOBS:-12}
LOG=results/w2ana_queue.log
say () { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
busy () { pgrep -f 'run_models\.py|train_dose_cond\.py' | wc -l | tr -d ' '; }
for s in "$@"; do
  for arm in wide narrow; do
    if [ "$arm" = wide ]; then cells="confound_km09 confound_km00"; sel="--select-on mse_v2"
    else cells="confound_kmn09 confound_kmn00"; sel=""; fi
    tag="w2ana_s${s}${arm}"; R="results/w2ana/${tag}"
    for c in $cells; do
      [ -f "$R/exp_film_run/$c/experiment_film_${c}"*w2ana*_best.ckpt ] 2>/dev/null && continue
      while [ "$(busy)" -ge "$MAXJOBS" ]; do sleep 60; done
      mkdir -p "$R/exp_film_run/$c"
      nohup python3 run_models.py --niters 6000 -n 200 -s 40 -l 10 --dataset PK_Tacro \
          --latent-ode --use_film --noise-weight 0.01 --max-t 5. -b 512 --seed "$s" \
          --experiment "$c" --film-no-z0-cond --obsrv-std 0.217 --film-self-consistency 7.5 \
          --decoder-hidden 50 --w2-analytic-moments $sel --patience 1000000 \
          --save "./$R/" --ckpt-every 1000 --log-suffix "__${tag}" >/dev/null 2>&1 &
      say "  launched seed $s $arm $c (pid $!), jobs now $(busy)"
      sleep 3
    done
  done
done
say "seeds $* all queued"
wait
say "w2ana seeds $* complete"
