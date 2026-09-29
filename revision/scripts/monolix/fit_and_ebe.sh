#!/bin/bash
# One sweep size: SAEM fit of the true-structure model, then MAP-BE of the 1000 test
# patients under the estimated population model.
#   scripts/monolix/fit_and_ebe.sh <N> [threads]      (N=800 uses the full cohort)
#   scripts/monolix/fit_and_ebe.sh 100 wait           (N=100 already fitting: only wait, then EBE)
#   VARIANT=l1 scripts/monolix/fit_and_ebe.sh <N> [threads]   (fit_s4_mm.R variant l1; own dirs/files)
set -u
cd "$(dirname "$0")/../.."
N=${1:?N}; TH=${2:-2}; PY=/opt/miniconda3/bin/python3.12
VAR=${VARIANT:-true}; T=""; [ "$VAR" != true ] && T="${VAR}_"
# COHORT (default confound_vc00_s4) selects the data; a non-default cohort adds its suffix to every name
COH=${COHORT:-confound_vc00_s4}; SFX=${COH#confound_vc00_s4}; SFX=${SFX#_}; [ -n "$SFX" ] && T="${T}${SFX}_"
o=results/monolix/fit_${T}n$N
say () { echo "[$(date '+%F %T')] N=$N ${T}$*"; }
if [ "$TH" = wait ]; then
  while pgrep -f "fit_s4_mm.R results/monolix/data/confound_vc00_s4_n100.csv" >/dev/null; do sleep 60; done
else
  d=results/monolix/data/${COH}_n$N.csv; [ "$N" = 800 ] && d=results/monolix/data/${COH}.csv
  [ -e "$o/estimates.json" ] && { say "estimates already exist -- not refitting"; } || {
    mkdir -p $o; say "fitting ($TH threads)"
    nice -n 5 timeout 64800 Rscript scripts/monolix/fit_s4_mm.R $d $o $TH $VAR > $o/fit.log 2>&1; }
fi
grep -q FIT_DONE $o/fit.log || { say "fit did not finish cleanly -- see $o/fit.log"; exit 1; }
say "fit done ($(grep -oE 'SAEM done after [0-9.]+ min' $o/fit.log)); EBE on 1000 test patients"
EBE_COHORT=$COH nice -n 5 $PY scripts/monolix/ebe_popmodel.py $o/estimates.json 1000 0 results/s4/ebe/popebe_mlx_${T}n${N}_n1000.csv > logs/popebe_mlx_${T}n$N.log 2>&1 \
  && say "EBE done -> results/s4/ebe/popebe_mlx_${T}n${N}_n1000.csv" || say "EBE failed -- see logs/popebe_mlx_${T}n$N.log"
