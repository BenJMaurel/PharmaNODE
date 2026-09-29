#!/bin/bash
# Wait for the lo/hi cohorts, check pairing, score both architectures, then draw the
# out-of-support curves.  Writes OOD_EVAL_DONE to logs/s4_ood_chain.log at the end.
cd "$(dirname "$0")/../.."
PY=/opt/miniconda3/bin/python3.12
while pgrep -f "gen_tacro_confound.py --exp confound_vc00_s4_(lo|hi)" >/dev/null || \
      [ ! -f results/exp_film_run/confound_vc00_s4_lo/virtual_cohort_film_test.csv ] || \
      [ ! -f results/exp_film_run/confound_vc00_s4_hi/virtual_cohort_film_test.csv ]; do sleep 30; done
echo "[$(date '+%H:%M')] cohorts ready"
$PY scripts/s4/check_ood_pairing.py || { echo "ABORT: pairing failed"; exit 1; }
scripts/s4/eval_s4_ood.sh results/s4 006000 "1 2 3 4" 3
echo "[$(date '+%H:%M')] evaluations done; drawing curves"
for r in lo hi; do
  ( OMP_NUM_THREADS=1 nice -n 5 $PY scripts/s4/curves_s4.py results/s4 006000 1 confound_vc00_s4_$r \
      results/s4/curves/ep006000_s1_vc00_on_vc00$r.npz > logs/curves_s4_$r.log 2>&1 ) &
done; wait
echo "[$(date '+%H:%M')] OOD_EVAL_DONE"
