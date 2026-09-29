#!/bin/bash
# Hidden-saturation experiment (handoff 11.3), launched 28/09 night. Train = confound_vc00_s4_sat30_win_n100 (k = 30,
# where linear ties MM on BIC). Test = a NEW wide population: per-patient k log-uniform in [1, 30] (seed 1, dose seed
# 1235, IDs offset by 10000 so they can never collide with training IDs), windowed 3-25 ng/mL, paper noise.
# Arms: MAP-BE linear (the BIC-selected model), MAP-BE MM (estimated on the same train), plain latent ODE x 3 seeds.
cd /Users/benjaminmaurel/Documents/PharmaNODE
PY=/opt/miniconda3/bin/python3.12; RX=results/exp_film_run; TR=confound_vc00_s4_sat30_win_n100; TE=confound_vc00_s4_satwide_win
E=s4sat30_lode_n100; R=results/sat; TH="OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1"
say() { echo "[$(date '+%F %T')] $*"; }
mkdir -p $R/ebe $R/eval logs
if [ ! -f $RX/$TE/confound_truth.csv ]; then
  env $TH nice -n 5 $PY gen_tacro_confound.py --exp $TE --num_patients 2000 --rho 0.0 --test-fraction 0.99 --seed 1 \
    --dose-seed 1235 --id-offset 10000 --confound-param Vc --scenario 4 --dose-grid 1,2,3,4,5,6,7,8 --prop-sd 0.113 \
    --add-sd 0.71 --v1-cavg-window 3,25 --sat-logunif 1,30 > logs/gen_satwide.log 2>&1 || { say "GEN FAILED"; exit 1; }
fi
grep -h "rejected\|train .* test" logs/gen_satwide.log
mkdir -p results/$E
cp -n $RX/$TR/virtual_cohort_film_train.csv results/$E/virtual_cohort_train.csv
cp -n $RX/$TE/virtual_cohort_film_test.csv results/$E/virtual_cohort_test.csv
$PY -c "
import pandas as pd; a=pd.read_csv('results/$E/virtual_cohort_train.csv'); b=pd.read_csv('results/$E/virtual_cohort_test.csv')
assert not set(a.ID) & set(b.ID), 'ID collision'; print('latent ODE data: train', a.ID.nunique(), 'patients / test', b.ID.nunique(), 'patients, no ID overlap')" || exit 1
for s in 1 2 3; do
  grep -q "Training complete" logs/train_sat_lode_s$s.log 2>/dev/null && continue
  mkdir -p $R/lode_s$s/$E
  env $TH nohup nice -n 5 $PY run_models.py --niters 6000 -n 200 -s 40 -l 10 --dataset PK_Tacro --latent-ode \
    --noise-weight 0.01 --max-t 5. -b 512 --seed $s --experiment $E --static-dim 4 --n-train-series 200 \
    --save ./$R/lode_s$s/ > logs/train_sat_lode_s$s.log 2>&1 &
  say "plain latent ODE seed $s (pid $!)"; sleep 60
done
# ---- MAP-BE arms (fast) ----
$PY scripts/sat/lin_to_mm_spec.py $R/calib/sat30_lin/estimates.json $R/calib/sat30_lin/estimates_as_mm.json
for arm in lin mm; do
  f=$R/calib/sat30_mm/estimates.json; [ $arm = lin ] && f=$R/calib/sat30_lin/estimates_as_mm.json
  [ -f $R/ebe/popebe_${arm}_satwide.csv ] || env $TH EBE_COHORT=$TE nice -n 5 $PY scripts/monolix/ebe_popmodel.py $f 100000 0 \
    $R/ebe/popebe_${arm}_satwide.csv > logs/ebe_satwide_$arm.log 2>&1 || say "EBE $arm FAILED"
  say "EBE $arm done"
done
# ---- wait for the networks, then score ----
for s in 1 2 3; do until grep -q "Training complete" logs/train_sat_lode_s$s.log 2>/dev/null; do sleep 300; done; done
say "training done; scoring"
for s in 1 2 3; do
  [ -f $R/eval/lode_s$s.json ] || env $TH nice -n 5 $PY scripts/repro/eval_lode.py $R/lode_s$s/$E/experiment_$E.ckpt \
    $R/eval/lode_s$s.json > logs/eval_sat_lode_s$s.log 2>&1 &
done; wait
$PY scripts/sat/sat_table.py | tee $R/sat_table.txt
say "SATEXP_DONE"
