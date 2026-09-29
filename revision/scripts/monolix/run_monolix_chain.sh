#!/bin/bash
# Sequential Monolix fits for the patient-count sweep, each followed by the MAP-BE of the
# 1000 test patients under the ESTIMATED population model.  Waits for the N=100 test fit,
# and stops if it did not finish cleanly.
set -u
cd "$(dirname "$0")/../.."
PY=/opt/miniconda3/bin/python3.12
say () { echo "[$(date '+%F %T')] $*"; }
while pgrep -f "fit_s4_mm.R results/monolix/data/confound_vc00_s4_n100.csv" >/dev/null; do sleep 60; done
grep -q FIT_DONE results/monolix/fit_n100/fit.log || { say "N=100 fit did not finish cleanly -- chain stopped"; exit 1; }
say "N=100 fit done; EBE on 1000 test patients"
nice -n 5 $PY scripts/monolix/ebe_popmodel.py results/monolix/fit_n100/estimates.json 1000 0 results/s4/ebe/popebe_mlx_n100_n1000.csv > logs/popebe_mlx_n100.log 2>&1
for n in 200 400 800; do
  d=results/monolix/data/confound_vc00_s4_n$n.csv; [ $n = 800 ] && d=results/monolix/data/confound_vc00_s4.csv
  o=results/monolix/fit_n$n; mkdir -p $o
  say "fitting N=$n"
  nice -n 5 timeout 21600 Rscript scripts/monolix/fit_s4_mm.R $d $o 2 > $o/fit.log 2>&1
  grep -q FIT_DONE $o/fit.log || { say "N=$n fit failed -- see $o/fit.log"; continue; }
  say "N=$n fit done ($(grep -oE 'SAEM done after [0-9.]+ min' $o/fit.log)); EBE"
  nice -n 5 $PY scripts/monolix/ebe_popmodel.py $o/estimates.json 1000 0 results/s4/ebe/popebe_mlx_n${n}_n1000.csv > logs/popebe_mlx_n$n.log 2>&1
done
say "MONOLIX_CHAIN_DONE"
