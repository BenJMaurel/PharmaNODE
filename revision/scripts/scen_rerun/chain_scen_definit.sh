#!/bin/bash
# 3-scenario MAP-BE redo with Monolix DEFAULT initial values (Benjamin, 2026-09-26), after the main rerun.
# Gate: SCEN_RERUN_DONE in logs/chain_scen_rerun.log (or pid $RERUN_PID gone) AND VCW_MATCHED_DONE in
# logs/chain_vcw_matched.log (or pid $CHAIN_PID gone), so the N=800 scoring is not starved again.
# Validation first: the 22/09 scenario-1 dataset (id 91101) with INIT_MODE=default SIGMA_MODE=sd must reproduce the
# committed pipeline's own fit and MAP-BE output (results/paper_repro/s1/results/91101); the comparison is logged,
# the chain stops only if the R step fails. Then 10 workers over seeds 1..100 x scenarios 1-3, cap-gated, restartable
# (DONE_definit markers). Summary -> results/scen_rerun/summary_definit.txt. Markers DEFINIT_CHECK_DONE, SCEN_DEFINIT_DONE.
cd /Users/benjaminmaurel/Documents/PharmaNODE
CAP=${CAP:-1000}; NSLOT=${NSLOT:-10}; RERUN_PID=${RERUN_PID:-57736}; CHAIN_PID=${CHAIN_PID:-20198}; SEEDS=${SEEDS:-$(seq 1 100)}
PY=/opt/miniconda3/bin/python3.12; MLX=/Applications/monolixSuite2024R1.app/Contents/Resources/monolixSuite
say () { echo "[$(date '+%F %T')] $*"; }
cpu () { ps -A -o pcpu=,stat= | awk '$2 !~ /T/ {s+=$1} END {printf "%d", s}'; }
throttle () { while [ $(( $(cpu) + 100 )) -gt "$CAP" ]; do sleep 60; done; }
say "waiting for the main rerun (SCEN_RERUN_DONE / pid $RERUN_PID) and the N=800 chain (VCW_MATCHED_DONE / pid $CHAIN_PID)"
until { grep -q SCEN_RERUN_DONE logs/chain_scen_rerun.log 2>/dev/null || ! kill -0 $RERUN_PID 2>/dev/null; } && \
      { grep -q VCW_MATCHED_DONE logs/chain_vcw_matched.log 2>/dev/null || ! kill -0 $CHAIN_PID 2>/dev/null; }; do sleep 120; done
if [ "${SKIP_CHECK:-0}" = 1 ]; then say "SKIP_CHECK=1: validation skipped (passed 27/09 02:41: identical fit and MAP-BE output)"; else
say "gate open -> validation on the 22/09 dataset 91101 (default init, SIGMA as SDs = committed pipeline)"
V=results/scen_rerun/definit_check; rm -rf $V; mkdir -p $V; O=results/paper_repro/s1/results/91101
cp $O/virtual_cohort_train.csv $O/virtual_cohort_test.csv $V/
throttle
( cd results/paper_repro/w01 && MLX_THREADS=1 OMP_NUM_THREADS=1 INIT_MODE=default REUSE_FIT=0 SIGMA_MODE=sd OUT_TAG=check \
  nice -n 5 Rscript ../../../scripts/scen_rerun/all_run_tacro_rerun_init.r --virtual_cohort ../../../$V/virtual_cohort_test.csv \
  --output_dir ../../../$V --experiment 91101 --cores 1 --monolix_path "$MLX" > ../../../$V/r_check.log 2>&1 ) \
  || { say "VALIDATION R STEP FAILED -> chain aborted (see $V/r_check.log)"; exit 1; }
$PY - <<PYEOF > $V/validation.txt 2>&1
cat $V/validation.txt
import pandas as pd, numpy as np
a=pd.read_csv('$V/2_test_tacro/populationParameters.txt').set_index('parameter').value
b=pd.read_csv('$O/2_test_tacro/populationParameters.txt').set_index('parameter').value
print('validation: population estimates, new vs 22/09 committed-pipeline fit:')
for k in ('CL_pop','Vc_pop','KTR_pop','Q_pop','Vp_pop','a','b'): print(f'   {k:8s} {a[k]:10.4g} {b[k]:10.4g}')
x=pd.read_csv('$V/tacro_mapbayest_auc_check.csv'); y=pd.read_csv(sorted(__import__('glob').glob('$O/tacro_mapbayest_auc_*.csv'))[0])
m=x.merge(y,on='ID'); r=lambda d,c: 100*np.sqrt(np.mean((d[c]/d.AUC_observed_x-1)**2))
print(f'validation: MAP-BE RMSPE new {r(m,"auc_ipred_x"):.2f} vs 22/09 {r(m,"auc_ipred_y"):.2f} (n={len(m)}); max |dAUC| {float((m.auc_ipred_x-m.auc_ipred_y).abs().max()):.3g}')
PYEOF
say "DEFINIT_CHECK_DONE"
fi
JOBS=(); for s in $SEEDS; do for sc in 1 2 3; do JOBS+=("$sc $s"); done; done
say "${#JOBS[@]} jobs over $NSLOT slots"
worker () {
  local k=$1 w=results/paper_repro/w$(printf %02d $(( $1 + 1 ))) i
  for (( i = k; i < ${#JOBS[@]}; i += NSLOT )); do
    set -- ${JOBS[$i]}
    [ -f results/scen_rerun/runs/s$1_seed$(printf %03d $2)/DONE_definit ] && continue
    throttle
    bash scripts/scen_rerun/run_definit.sh $w $1 $2
  done
}
for (( k = 0; k < NSLOT; k++ )); do worker $k & sleep 60; done
wait
say "finished: $(ls results/scen_rerun/runs/*/DONE_definit 2>/dev/null | wc -l | tr -d ' ') DONE_definit, $(ls results/scen_rerun/runs/*/FAILED_definit_* 2>/dev/null | wc -l | tr -d ' ') FAILED"
$PY scripts/scen_rerun/summarise.py > results/scen_rerun/summary_definit.txt 2>&1
say "summary -> results/scen_rerun/summary_definit.txt"
say "SCEN_DEFINIT_DONE"
