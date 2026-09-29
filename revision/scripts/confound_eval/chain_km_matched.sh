#!/bin/bash
# Matched rerun of the scenario-3 Km confounding experiment (handoff §7.2 / §9.12.2), 2026-09-23.
# The published DiD (dose-cond +7.62 vs OT-FiLM +4.11, wide arm) compared FiLM at obsrv-std 0.217 / sc 7.5 /
# 50-unit readout with dose-cond at 0.01 / linear readout. Here both use the CURRENT main config:
#   OT-FiLM   : --obsrv-std 0.05 --film-self-consistency 141.27 --decoder-hidden 50 --decoder-residual
#   dose-cond : --obsrv-std 0.05 --decoder-hidden 50 --decoder-residual        (same likelihood, same readout)
# Remaining asymmetries (by design / known): FiLM has the transport + its penalties and a 3-sample IWAE bound;
# dose-cond uses the ELBO (no --iwae in any launch script). static-dim 3 (scenario 3, HT fixed at 35).
# Cohorts confound_km09 (rho 0.9) / confound_km00 (rho 0), same patients, 800 train / 200 test; seeds 1-3;
# 3000 epochs (the published common budget), traj ckpt every 300; 1 thread each, at most MAXJ concurrent.
# Scoring: ep 2400/2700/3000, each model on the four test sets (km09, km00 own test splits = the published
# protocol; km09_big, km00_big = 1000 new patients), scaler pinned to the TRAINING cohort, RNG seeded.
# Log: logs/chain_km_matched.log. Do not edit while running (bash reads scripts incrementally).
cd /Users/benjaminmaurel/Documents/PharmaNODE
PY=/opt/miniconda3/bin/python3.12
ROOT=${ROOT:-results/km_matched}; E=${EVAL:-$ROOT/eval}; CAP=${CAP:-0}   # CAP>0: also wait until total %cpu + 100 <= CAP
SEEDS="${SEEDS:-1 2 3}"; NIT=${NIT:-3000}; MAXJ=${MAXJ:-10}; EPS="${EPS:-002400 002700 003000}"
mkdir -p $E logs
say () { echo "[$(date '+%F %T')] $*"; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cpu () { ps -A -o pcpu=,stat= | awk '$2 !~ /T/ {s+=$1} END {printf "%d", s}'; }
throttle () { while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 20; done
              [ "$CAP" -gt 0 ] && while [ $(( $(cpu) + 100 )) -gt "$CAP" ]; do sleep 60; done; return 0; }

PIDF=logs/$(basename $ROOT).pids; : > $PIDF
for arch in film dc; do                       # FiLM first: slower per epoch
  for s in $SEEDS; do
    for c in confound_km09 confound_km00; do
      throttle
      R=$ROOT/s$s; mkdir -p $R/exp_film_run/$c $R/exp_dosecond_run/$c
      if [ $arch = film ]; then
        nohup nice -n 5 $PY run_models.py --niters $NIT -n 200 -s 40 -l 10 --dataset PK_Tacro \
          --latent-ode --use_film --noise-weight 0.01 --max-t 5. -b 512 --seed $s --experiment $c \
          --film-no-z0-cond --obsrv-std 0.05 --film-self-consistency 141.27 --decoder-hidden 50 --decoder-residual \
          --select-on mse_v2 --patience 1000000 --save ./$R/ --eval-every 300 --ckpt-every 300 \
          --log-suffix __kmm_s$s > logs/train_${LOGTAG:-kmm}_film_${c#confound_}_s$s.log 2>&1 &
      else
        nohup nice -n 5 $PY train_dose_cond.py --experiment $c --data-dir ./results/exp_film_run \
          --niters $NIT -b 512 -l 10 --lr 1e-2 --seed $s --patience 1000000 --obsrv-std 0.05 \
          --decoder-hidden 50 --decoder-residual --save ./$R/ --eval-every 300 --ckpt-every 300 \
          --log-suffix __kmm_s$s > logs/train_${LOGTAG:-kmm}_dc_${c#confound_}_s$s.log 2>&1 &
      fi
      echo "$!" >> $PIDF
      say "launched $arch $c seed $s (pid $!)"
      [ "$CAP" -gt 0 ] && sleep 90 || sleep 3     # let %cpu register before the next cap check
    done
  done
done
wait
say "training finished"

ckpt () {  # arch seed cohort epoch -> the single traj checkpoint, or empty
  local sub=exp_film_run; [ $1 = dc ] && sub=exp_dosecond_run
  ls $ROOT/s$2/$sub/$3/traj/*_ep$4.ckpt 2>/dev/null
}
n_ok=0; n_miss=0
for ep in $EPS; do
  for arch in film dc; do
    H=test_film_matched.py; [ $arch = dc ] && H=test_dose_cond.py
    for s in $SEEDS; do
      for tr in confound_km09 confound_km00; do
        CK=$(ckpt $arch $s $tr $ep)
        if [ "$(echo "$CK" | grep -c ckpt)" != 1 ]; then say "MISSING/AMBIGUOUS ckpt $arch s$s $tr ep$ep: '$CK'"; n_miss=$((n_miss+1)); continue; fi
        for ev in confound_km09 confound_km00 confound_km09_big confound_km00_big; do
          out=$E/ep${ep}_${arch}_s${s}_${tr#confound_}_on_$(echo ${ev#confound_} | tr -d _).json
          [ -f $out ] && continue
          throttle
          nice -n 5 $PY scripts/confound_eval/seeded_eval.py --eval-seed 0 $H --experiment $ev \
            --data-dir ./results/exp_film_run --ckpt $CK --scale-from $tr --eval-split test \
            --label kmm_${arch}_s${s}_${tr#confound_}_on_${ev#confound_}_ep$ep --out-json $out \
            > logs/eval_${LOGTAG:-kmm}_${arch}_s${s}_${tr#confound_}_on_${ev#confound_}_ep$ep.log 2>&1 &
          n_ok=$((n_ok+1))
        done
      done
    done
  done
done
wait
say "scored ($n_ok evaluations launched, $n_miss missing checkpoints); $(ls $E | wc -l | tr -d ' ') JSON files"
KM_EVAL=$E KM_EPS="$EPS" $PY scripts/confound_eval/km_matched_did.py > $ROOT/did_summary.txt 2>&1 && say "summary -> $ROOT/did_summary.txt"
say "KM_MATCHED_DONE"
