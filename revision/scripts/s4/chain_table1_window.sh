#!/bin/bash
# Table 1 of ndm.tex on the WINDOWED benchmark (handoff 1.12, Benjamin 2026-09-24): scenario 4, paper noise, visit-1
# average concentration restricted to 3-25 ng/mL, N = 100 training patients, same protocol as the current Table 1.
#   cohorts   confound_vc00_s4_pnoise_win{,_lo,_hi} (made by the other session's scripts/s4/gen_window_all.sh)
#   networks  OT-FiLM residual, dose-cond residual, Lu faithful, Lu same-task; seeds 1-3; 6000 epochs; traj every 300
#   refs      Monolix fit (true covariates, L1) on the 100 training patients + EBE; true-model EBE
#   scoring   ep6000 on in / lo / hi with the scaler pinned to the training cohort, seeded ->
#             results/s4_pnoise_win/eval/ep006000_<film|dc|lu-faithful|lu-static>_n100_s<seed>_on_<in|lo|hi>.json,
#             training cohort (--eval-split all) -> results/recal/paper_win/<arch>_s<seed>_all.json
# Markers in logs/chain_table1_window.log: TABLE1_WIN_TRAINED (the Vc chain waits on it), TABLE1_WIN_DONE.
# Do not edit while running.
cd /Users/benjaminmaurel/Documents/PharmaNODE
PY=/opt/miniconda3/bin/python3.12
B=confound_vc00_s4_pnoise_win; T=${B}_n100; R=results/s4_pnoise_win; RC=results/recal/paper_win
SEEDS="1 2 3"; NIT=6000; MAXJ=${MAXJ:-10}; CAP=${CAP:-1000}
mkdir -p $R/eval $RC logs
say () { echo "[$(date '+%F %T')] $*"; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cpu () { ps -A -o pcpu=,stat= | awk '$2 !~ /T/ {s+=$1} END {printf "%d", s}'; }
throttle () { while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 20; done
              while [ $(( $(cpu) + 100 )) -gt "$CAP" ]; do sleep 60; done; return 0; }

for c in $B ${B}_lo ${B}_hi; do
  while pgrep -f "gen_tacro_confound.py --exp $c " >/dev/null || [ ! -f results/exp_film_run/$c/noise.json ]; do sleep 60; done
  say "cohort $c ready: $(grep -c . results/exp_film_run/$c/confound_truth.csv) truth rows"
done
[ -d results/exp_film_run/$T ] || $PY scripts/nsweep/make_subsets.py $B "100" || { say "SUBSET FAILED"; exit 1; }
$PY - <<PY
import pandas as pd
R = "results/exp_film_run"
b = pd.read_csv(f"{R}/$B/confound_truth.csv"); bt = b[b.split == "test"].set_index("ID")
for reg in ("lo", "hi"):
    t = pd.read_csv(f"{R}/${B}_{reg}/confound_truth.csv"); tt = t[t.split == "test"].set_index("ID")
    same = set(tt.index) == set(bt.index)
    d1 = same and (tt.loc[bt.index, "d1"].values == bt.d1.values).all()
    print(f"pairing {reg}: n_test {len(tt)} vs {len(bt)}, same ids {same}, same d1 {d1}, d2 {sorted(tt.d2.unique())}")
PY

# ---- references (background): Monolix true-covariate + L1 fits on the 100 training patients, true-model EBE ----
$PY scripts/monolix/build_monolix_data.py $T results/monolix/data/$T.csv > logs/mlx_data_win.log 2>&1
COHORT=$B scripts/monolix/fit_and_ebe.sh 100 2 > logs/monolix_pnoise_win_n100.log 2>&1 &
COHORT=$B VARIANT=l1 scripts/monolix/fit_and_ebe.sh 100 2 > logs/monolix_l1_pnoise_win_n100.log 2>&1 &
EBE_COHORT=$B nice -n 5 $PY scripts/monolix/ebe_popmodel.py true 1000 0 results/s4/ebe/popebe_true_pnoise_win_n1000.csv \
  > logs/popebe_true_pnoise_win.log 2>&1 &
REFS="$(jobs -p | tr '\n' ' ')"
say "references launched (pids $REFS)"
sleep 120

# ---- networks: longest first ----
TRAIN=""
for arch in lu-static lu-faithful film dc; do
  for s in $SEEDS; do
    throttle
    D=$R/${arch}_n100_s$s; mkdir -p $D
    L=logs/train_t1win_${arch}_s$s.log
    case $arch in
      lu-static)   nohup nice -n 5 $PY -W ignore train_lu_pk.py --experiment $T --data-dir ./results/exp_film_run --niters $NIT \
                     -b 512 --seed $s --save ./$D/ --static-dim 4 --use-static --train-mode counterfactual \
                     --ckpt-every 300 --eval-every 300 > $L 2>&1 & ;;
      lu-faithful) nohup nice -n 5 $PY -W ignore train_lu_pk.py --experiment $T --data-dir ./results/exp_film_run --niters $NIT \
                     -b 512 --seed $s --save ./$D/ --static-dim 4 --ckpt-every 300 --eval-every 300 > $L 2>&1 & ;;
      film)        mkdir -p $D/exp_film_run/$T
                   nohup nice -n 5 $PY run_models.py --niters $NIT -n 200 -s 40 -l 10 --dataset PK_Tacro --latent-ode --use_film \
                     --noise-weight 0.01 --max-t 5. -b 512 --seed $s --experiment $T --film-no-z0-cond --obsrv-std 0.05 \
                     --film-self-consistency 141.27 --decoder-hidden 50 --decoder-residual --select-on mse_v2 --patience 1000000 \
                     --static-dim 4 --save ./$D/ --eval-every 300 --ckpt-every 300 --log-suffix __t1win_s$s > $L 2>&1 & ;;
      dc)          mkdir -p $D/exp_dosecond_run/$T
                   nohup nice -n 5 $PY train_dose_cond.py --experiment $T --data-dir ./results/exp_film_run --niters $NIT -b 512 \
                     -l 10 --lr 1e-2 --seed $s --patience 1000000 --obsrv-std 0.05 --static-dim 4 --decoder-hidden 50 \
                     --decoder-residual --save ./$D/ --eval-every 300 --ckpt-every 300 --log-suffix __t1win_s$s > $L 2>&1 & ;;
    esac
    TRAIN="$TRAIN $!"; echo "$!" >> logs/t1win.pids
    say "launched $arch seed $s (pid $!)"; sleep 90
  done
done
wait $TRAIN
say "TABLE1_WIN_TRAINED"

# ---- scoring at ep6000 ----
EV=""
for arch in film dc lu-faithful lu-static; do
  case $arch in film) H=test_film_matched.py; sub=exp_film_run ;; dc) H=test_dose_cond.py; sub=exp_dosecond_run ;;
                *) H=test_lu_pk.py; sub=exp_lupk_run ;; esac
  for s in $SEEDS; do
    CK=$(ls $R/${arch}_n100_s$s/$sub/$T/traj/*_ep006000.ckpt 2>/dev/null)
    [ "$(echo "$CK" | grep -c ckpt)" = 1 ] || { say "MISSING ckpt $arch s$s: '$CK'"; continue; }
    for reg in in lo hi; do
      ex=$B; [ $reg != in ] && ex=${B}_$reg
      throttle
      nice -n 5 $PY scripts/confound_eval/seeded_eval.py --eval-seed 0 $H --experiment $ex --data-dir ./results/exp_film_run \
        --ckpt $CK --scale-from $T --eval-split test --label t1win_${arch}_s${s}_$reg \
        --out-json $R/eval/ep006000_${arch}_n100_s${s}_on_$reg.json > logs/eval_t1win_${arch}_s${s}_$reg.log 2>&1 &
      EV="$EV $!"
    done
    throttle
    nice -n 5 $PY scripts/confound_eval/seeded_eval.py --eval-seed 0 $H --experiment $T --data-dir ./results/exp_film_run \
      --ckpt $CK --scale-from $T --eval-split all --label t1win_${arch}_all_s$s \
      --out-json $RC/${arch}_s${s}_all.json > logs/recal_t1win_${arch}_s$s.log 2>&1 &
    EV="$EV $!"
  done
done
[ -n "$EV" ] && wait $EV
say "scored: $(ls $R/eval | wc -l | tr -d ' ') eval files"
wait $REFS 2>/dev/null
say "references: $(ls results/s4/ebe/ | grep -c pnoise_win) EBE files"
say "TABLE1_WIN_DONE"
