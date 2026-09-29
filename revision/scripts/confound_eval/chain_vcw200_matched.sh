#!/bin/bash
# N=200 VERSION (Benjamin, 2026-09-24: 'start with 200 patients to quicken the experiments'): nested subsets
# confound_vc0{9,0}_win_n200 (make_subsets seed 0 -> contained in the n800 subsets), full-batch steps. Starts once the
# windowed Table 1 chain has LAUNCHED its last run (it keeps priority on the cores; the CPU cap arbitrates the rest).
# --- header of chain_vcw_matched.sh ---
# WINDOWED rerun of the matched Vc confounding experiment (Benjamin, 2026-09-24): same as chain_vc_matched.sh but on
# cohorts restricted to a clinically plausible visit-1 exposure (Cavg 3-25 ng/mL, handoff 1.12), made by
# scripts/confound_eval/gen_vc_window.sh. NB the window induces corr(d1, log Vc) ~ +0.49 in the CONTROL arm, so the
# contrast is strong vs moderate confounding (~0.92 vs ~0.49), not 0.9 vs 0. Starts only after GEN_VC_WIN_DONE and
# TABLE1_WIN_TRAINED (Table 1 has priority). Labels: vcw09 / vcw00 (train), vcw09 / vcw00 / vcw09big / vcw00big (test).
# --- original header follows ---
# Matched Vc confounding experiment, three architectures (Benjamin, 2026-09-24): the SHORTCUT test.
# Vc is identifiable from the curve (probe R2 ~0.86), so the Bayes-optimal use of a confounded dose is small; a model
# damaged much more than that when the dose-Vc association is withdrawn has taken the shortcut. (The Km arm, where Km
# is not identifiable, measures transportability instead: there using the dose is the optimal inference.)
# Cohorts confound_vc09 (rho 0.9 on log Vc) / confound_vc00, same patients, 800 train / 200 test (+ vc09_big / vc00_big,
# 1000 new patients), scenario 3, --static-dim 3. Seeds 1-3, 6000 epochs, traj every 300, 1 thread each.
#   OT-FiLM   : --obsrv-std 0.05 --film-self-consistency 141.27 --decoder-hidden 50 --decoder-residual
#   dose-cond : --obsrv-std 0.05 --decoder-hidden 50 --decoder-residual
#   Lu et al. : "same task" variant, --static-dim 3 --use-static --train-mode counterfactual (port defaults otherwise)
# Launch order Lu -> FiLM -> dose-cond (longest first), at most MAXJ concurrent and total %cpu + 100 <= CAP.
# Scoring at ep 2400/2700/3000 and 5400/5700/6000 on vc09, vc00, vc09_big, vc00_big, scaler pinned to the training
# cohort, seeded -> results/vc_matched/eval/ep<EP>_<film|dc|lu>_s<seed>_<vc09|vc00>_on_<...>.json. FiLM + dose-cond are
# scored as soon as they finish (partial summaries), Lu when it finishes; summaries via km_matched_did.py with KM_PFX=vc
# -> results/vc_matched/did_summary_ep{3000,6000}.txt. Log logs/chain_vc_matched.log. Do not edit while running.
cd /Users/benjaminmaurel/Documents/PharmaNODE
PY=/opt/miniconda3/bin/python3.12
ROOT=results/vcw200_matched; LROOT=results/vcw200_matched_lu; E=$ROOT/eval
SEEDS="${SEEDS:-1 2 3}"; NIT=${NIT:-6000}; MAXJ=${MAXJ:-10}; CAP=${CAP:-1000}
EPS="002400 002700 003000 005400 005700 006000"
mkdir -p $E logs
say () { echo "[$(date '+%F %T')] $*"; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cpu () { ps -A -o pcpu=,stat= | awk '$2 !~ /T/ {s+=$1} END {printf "%d", s}'; }
throttle () { while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 20; done
              while [ $(( $(cpu) + 100 )) -gt "$CAP" ]; do sleep 60; done; return 0; }
COH="confound_vc09_win_n200 confound_vc00_win_n200"
lab () { case $1 in confound_vc09_win_n200) echo vcw09 ;; confound_vc00_win_n200) echo vcw00 ;;
                     confound_vc09_big_win) echo vcw09big ;; confound_vc00_big_win) echo vcw00big ;; esac; }
until grep -q GEN_VC_WIN_DONE logs/gen_vc_win.log 2>/dev/null; do sleep 120; done
for b in confound_vc09_win confound_vc00_win; do
  [ -d results/exp_film_run/${b}_n200 ] || $PY scripts/nsweep/make_subsets.py $b "200" 0 || { say "SUBSET FAILED $b"; exit 1; }
done
until grep -q "launched dc seed 3" logs/chain_table1_window.log 2>/dev/null; do sleep 120; done
for c in $COH; do [ -d results/exp_film_run/$c ] || { say "missing cohort $c"; exit 1; }; done
say "cohorts ready and Table 1 fully launched -> starting"

PID_LU=""; PID_FD=""; : > logs/vcw200_matched.pids
for arch in lu film dc; do
  for s in $SEEDS; do
    for c in $COH; do
      throttle
      L=logs/train_vcw200_${arch}_$(lab $c)_s$s.log
      if [ $arch = lu ]; then
        R=$LROOT/s$s; mkdir -p $R
        nohup nice -n 5 $PY -W ignore train_lu_pk.py --experiment $c --data-dir ./results/exp_film_run \
          --niters $NIT -b 512 --seed $s --save ./$R/ --static-dim 3 --use-static --train-mode counterfactual \
          --ckpt-every 300 --eval-every 300 > $L 2>&1 &
        PID_LU="$PID_LU $!"
      elif [ $arch = film ]; then
        R=$ROOT/s$s; mkdir -p $R/exp_film_run/$c
        nohup nice -n 5 $PY run_models.py --niters $NIT -n 200 -s 40 -l 10 --dataset PK_Tacro \
          --latent-ode --use_film --noise-weight 0.01 --max-t 5. -b 512 --seed $s --experiment $c \
          --film-no-z0-cond --obsrv-std 0.05 --film-self-consistency 141.27 --decoder-hidden 50 --decoder-residual \
          --select-on mse_v2 --patience 1000000 --save ./$R/ --eval-every 300 --ckpt-every 300 \
          --log-suffix __vcw200_s$s > $L 2>&1 &
        PID_FD="$PID_FD $!"
      else
        R=$ROOT/s$s; mkdir -p $R/exp_dosecond_run/$c
        nohup nice -n 5 $PY train_dose_cond.py --experiment $c --data-dir ./results/exp_film_run \
          --niters $NIT -b 512 -l 10 --lr 1e-2 --seed $s --patience 1000000 --obsrv-std 0.05 \
          --decoder-hidden 50 --decoder-residual --save ./$R/ --eval-every 300 --ckpt-every 300 \
          --log-suffix __vcw200_s$s > $L 2>&1 &
        PID_FD="$PID_FD $!"
      fi
      echo "$!" >> logs/vcw200_matched.pids
      say "launched $arch $c seed $s (pid $!)"
      sleep 90                                  # let %cpu register before the next cap check
    done
  done
done

ckpt () {  # arch seed cohort epoch -> the single traj checkpoint, or empty
  case $1 in
    lu)   ls $LROOT/s$2/exp_lupk_run/$3/traj/*_ep$4.ckpt 2>/dev/null ;;
    film) ls $ROOT/s$2/exp_film_run/$3/traj/*_ep$4.ckpt 2>/dev/null ;;
    dc)   ls $ROOT/s$2/exp_dosecond_run/$3/traj/*_ep$4.ckpt 2>/dev/null ;;
  esac
}
score () {  # archs...   (waits only for its own evaluations, not for training jobs still running)
  local n_ok=0 n_miss=0 EV=""
  for ep in $EPS; do
    for arch in "$@"; do
      case $arch in lu) H=test_lu_pk.py ;; film) H=test_film_matched.py ;; dc) H=test_dose_cond.py ;; esac
      for s in $SEEDS; do
        for tr in $COH; do
          CK=$(ckpt $arch $s $tr $ep)
          if [ "$(echo "$CK" | grep -c ckpt)" != 1 ]; then say "MISSING/AMBIGUOUS ckpt $arch s$s $tr ep$ep: '$CK'"; n_miss=$((n_miss+1)); continue; fi
          for ev in confound_vc09_win_n200 confound_vc00_win_n200 confound_vc09_big_win confound_vc00_big_win; do
            out=$E/ep${ep}_${arch}_s${s}_$(lab $tr)_on_$(lab $ev).json
            [ -f $out ] && continue
            throttle
            nice -n 5 $PY scripts/confound_eval/seeded_eval.py --eval-seed 0 $H --experiment $ev \
              --data-dir ./results/exp_film_run --ckpt $CK --scale-from $tr --eval-split test \
              --label vcw200_${arch}_s${s}_$(lab $tr)_on_$(lab $ev)_ep$ep --out-json $out \
              > logs/eval_vcw200_${arch}_s${s}_$(lab $tr)_on_$(lab $ev)_ep$ep.log 2>&1 &
            EV="$EV $!"; n_ok=$((n_ok+1))
          done
        done
      done
    done
  done
  [ -n "$EV" ] && wait $EV
  say "scored $* ($n_ok evaluations launched, $n_miss missing checkpoints)"
}
summarise () {
  for set in "002400 002700 003000:3000" "005400 005700 006000:6000"; do
    KM_PFX=vcw KM_EVAL=$E KM_EPS="${set%%:*}" $PY scripts/confound_eval/km_matched_did.py > $ROOT/did_summary_ep${set##*:}.txt 2>&1
  done
  say "summaries -> $ROOT/did_summary_ep{3000,6000}.txt"
}

wait $PID_FD
say "FiLM + dose-cond training finished"
score film dc
summarise
say "VCW200_FD_DONE"
wait $PID_LU
say "Lu training finished"
score lu
summarise
say "VCW200_MATCHED_DONE"
