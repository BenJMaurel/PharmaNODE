#!/bin/bash
# Non-Gaussian scenario, part 2 -- networks at N=100, seeds 1-3: FiLM (residual decoder), dose-cond (residual
# decoder), plain latent ODE (linear decoder, HT in the encoder). Launched one at a time under an 800% CPU cap,
# then scored 3 jobs at a time (test in/lo/hi + training cohort for the recalibration factor). Restart-safe:
# finished runs and existing JSONs are skipped. Run AFTER the reboot (scripts/s4/after_reboot.sh).
cd /Users/benjaminmaurel/Documents/PharmaNODE
B=confound_vc00_s4_tslow; T=${B}_n100; L=s4tslow_lode_n100; R=results/s4_tslow; PY=/opt/miniconda3/bin/python3.12
FT=__noz0_sig0.05_sc141.27_dech50_sel-mse_v2_decres_ep006000.ckpt; DT=__dech-50_decres_sig-0.05_ep006000.ckpt
cpu () { ps -A -o pcpu=,stat= | awk '$2 !~ /T/ {s+=$1} END {printf "%d", s}'; }
room () { while [ $(( $(cpu) + 100 )) -gt 800 ]; do sleep 120; done; }
while [ ! -d results/exp_film_run/$T ] || [ ! -f results/$L/virtual_cohort_test.csv ]; do sleep 60; done
TH="OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1"
for s in 1 2 3; do
  if [ ! -f $R/film_n100_s$s/exp_film_run/$T/traj/experiment_film_$T$FT ]; then
    room; mkdir -p $R/film_n100_s$s/exp_film_run/$T
    env $TH nohup nice -n 5 $PY run_models.py --niters 6000 -n 200 -s 40 -l 10 --dataset PK_Tacro --latent-ode --use_film \
      --noise-weight 0.01 --max-t 5. -b 512 --seed $s --experiment $T --film-no-z0-cond --obsrv-std 0.05 \
      --film-self-consistency 141.27 --decoder-hidden 50 --decoder-residual --select-on mse_v2 --patience 1000000 \
      --static-dim 4 --save ./$R/film_n100_s$s/ --eval-every 300 --ckpt-every 300 --log-suffix "__tslow_s$s" \
      > logs/train_tslow_film_s$s.log 2>&1 &
    echo "[$(date '+%F %T')] film seed $s (pid $!)"; sleep 120
  fi
  if [ ! -f $R/dc_n100_s$s/exp_dosecond_run/$T/traj/experiment_dosecond_$T$DT ]; then
    room; mkdir -p $R/dc_n100_s$s/exp_dosecond_run/$T
    env $TH nohup nice -n 5 $PY train_dose_cond.py --experiment $T --data-dir ./results/exp_film_run --niters 6000 -b 512 \
      -l 10 --lr 1e-2 --seed $s --patience 1000000 --obsrv-std 0.05 --static-dim 4 --decoder-hidden 50 --decoder-residual \
      --save ./$R/dc_n100_s$s/ --eval-every 300 --ckpt-every 300 --log-suffix "__tslow_s$s" > logs/train_tslow_dc_s$s.log 2>&1 &
    echo "[$(date '+%F %T')] dose-cond seed $s (pid $!)"; sleep 120
  fi
  if ! grep -q "Training complete" logs/train_tslow_lode_s$s.log 2>/dev/null; then
    room; mkdir -p $R/lode_s$s/$L
    env $TH nohup nice -n 5 $PY run_models.py --niters 6000 -n 200 -s 40 -l 10 --dataset PK_Tacro --latent-ode \
      --noise-weight 0.01 --max-t 5. -b 512 --seed $s --experiment $L --static-dim 4 --n-train-series 200 \
      --save ./$R/lode_s$s/ > logs/train_tslow_lode_s$s.log 2>&1 &
    echo "[$(date '+%F %T')] plain latent ODE seed $s (pid $!)"; sleep 120
  fi
done
# ---- wait for all 9, then score ----
for s in 1 2 3; do
  while [ ! -f $R/film_n100_s$s/exp_film_run/$T/traj/experiment_film_$T$FT ] || \
        [ ! -f $R/dc_n100_s$s/exp_dosecond_run/$T/traj/experiment_dosecond_$T$DT ] || \
        ! grep -q "Training complete" logs/train_tslow_lode_s$s.log 2>/dev/null; do sleep 180; done
done
echo "[$(date '+%F %T')] all 9 runs done; scoring"
mkdir -p $R/eval results/recal/tslow; J=$R/score_jobs.txt; : > $J
for s in 1 2 3; do
  FC=$R/film_n100_s$s/exp_film_run/$T/traj/experiment_film_$T$FT; DC=$R/dc_n100_s$s/exp_dosecond_run/$T/traj/experiment_dosecond_$T$DT
  for reg in "" _lo _hi; do nm=${reg#_}; nm=${nm:-in}
    [ -f $R/eval/ep006000_film_n100_s${s}_on_$nm.json ] || echo "$TH nice -n 10 $PY test_film_matched.py --experiment $B$reg --scale-from $T --ckpt $FC --label tslow_film_s$s$reg --out-json $R/eval/ep006000_film_n100_s${s}_on_$nm.json > logs/eval_tslow_film_s$s$reg.log 2>&1" >> $J
    [ -f $R/eval/ep006000_dc_n100_s${s}_on_$nm.json ] || echo "$TH nice -n 10 $PY test_dose_cond.py --experiment $B$reg --scale-from $T --ckpt $DC --label tslow_dc_s$s$reg --out-json $R/eval/ep006000_dc_n100_s${s}_on_$nm.json > logs/eval_tslow_dc_s$s$reg.log 2>&1" >> $J
  done
  [ -f results/recal/tslow/film_s${s}_all.json ] || echo "$TH nice -n 10 $PY test_film_matched.py --experiment $T --scale-from $T --eval-split all --ckpt $FC --label tslow_film_all_s$s --out-json results/recal/tslow/film_s${s}_all.json > logs/recal_tslow_film_s$s.log 2>&1" >> $J
  [ -f results/recal/tslow/dc_s${s}_all.json ] || echo "$TH nice -n 10 $PY test_dose_cond.py --experiment $T --scale-from $T --eval-split all --ckpt $DC --label tslow_dc_all_s$s --out-json results/recal/tslow/dc_s${s}_all.json > logs/recal_tslow_dc_s$s.log 2>&1" >> $J
  LC=$R/lode_s$s/$L/experiment_$L.ckpt
  [ -f $R/eval/lode_s${s}_final.json ] || echo "$TH nice -n 10 $PY scripts/repro/eval_lode.py $LC $R/eval/lode_s${s}_final.json > logs/eval_tslow_lode_s$s.log 2>&1" >> $J
  [ -f $R/eval/lode_s${s}_final_train.json ] || echo "$TH nice -n 10 $PY scripts/repro/eval_lode_train.py $LC $R/eval/lode_s${s}_final_train.json > logs/eval_tslow_lode_train_s$s.log 2>&1" >> $J
done
scripts/s4/run_jobs.sh $J 3
echo "TSLOW_NET_DONE $(date '+%F %T')"
