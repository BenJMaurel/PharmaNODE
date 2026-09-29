#!/bin/bash
# Non-Gaussian scenario (Student-t etas + 30% slow absorbers, paper noise), part 1 -- statistical side.
# Waits for the three cohorts, builds the N=100 subset (= all 100 training patients) and the Monolix data,
# then: Monolix fit (log-normal, true covariate model) + EBE on the 1000 test patients, and the EBE with the
# generator's nominal Gaussian prior. Restart-safe (skips outputs that exist).
cd /Users/benjaminmaurel/Documents/PharmaNODE
B=confound_vc00_s4_tslow; PY=/opt/miniconda3/bin/python3.12
for c in $B ${B}_lo ${B}_hi; do while [ ! -f results/exp_film_run/$c/noise.json ]; do sleep 60; done; done
echo "[$(date '+%F %T')] cohorts ready"
[ -d results/exp_film_run/${B}_n100 ] || $PY scripts/nsweep/make_subsets.py $B "100"
[ -f results/monolix/data/${B}_n100.csv ] || $PY scripts/monolix/build_monolix_data.py ${B}_n100 results/monolix/data/${B}_n100.csv
E=s4tslow_lode_n100; mkdir -p results/$E
cp -n results/exp_film_run/${B}_n100/virtual_cohort_film_train.csv results/$E/virtual_cohort_train.csv
cp -n results/exp_film_run/${B}_n100/virtual_cohort_film_test.csv results/$E/virtual_cohort_test.csv
[ -f results/s4/ebe/popebe_true_tslow_n1000.csv ] || \
  EBE_COHORT=$B nice -n 5 $PY scripts/monolix/ebe_popmodel.py true 1000 0 results/s4/ebe/popebe_true_tslow_n1000.csv > logs/popebe_true_tslow.log 2>&1 &
[ -f results/s4/ebe/popebe_mlx_tslow_n100_n1000.csv ] || \
  COHORT=$B scripts/monolix/fit_and_ebe.sh 100 2 > logs/monolix_tslow_n100.log 2>&1
wait
echo "TSLOW_STAT_DONE $(date '+%F %T')"
