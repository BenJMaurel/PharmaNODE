#!/bin/bash
# Held-out learning curve of the matched Km runs: every model at ep 600/1200/1800/2100 on km09_big and km00_big
# (1000 new patients), scaler pinned to the training cohort, seeded -> results/km_matched/eval_curve/. The
# ep2400/2700/3000 cells already exist in results/km_matched/eval/. 3 evaluations at a time. Log logs/km_curve.log.
cd /Users/benjaminmaurel/Documents/PharmaNODE
PY=/opt/miniconda3/bin/python3.12; O=results/km_matched/eval_curve; mkdir -p $O
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
throttle () { while [ "$(jobs -rp | wc -l)" -ge "${MAXJ:-3}" ]; do sleep 10; done; }
for ep in 000600 001200 001800 002100; do
  for arch in film dc; do
    H=test_film_matched.py; sub=exp_film_run; [ $arch = dc ] && { H=test_dose_cond.py; sub=exp_dosecond_run; }
    for s in 1 2 3; do for tr in confound_km09 confound_km00; do
      CK=$(ls results/km_matched/s$s/$sub/$tr/traj/*_ep$ep.ckpt)
      for ev in confound_km09_big confound_km00_big; do
        out=$O/ep${ep}_${arch}_s${s}_${tr#confound_}_on_$(echo ${ev#confound_} | tr -d _).json
        [ -f $out ] && continue; throttle
        nice -n 5 $PY scripts/confound_eval/seeded_eval.py --eval-seed 0 $H --experiment $ev --data-dir ./results/exp_film_run \
          --ckpt $CK --scale-from $tr --eval-split test --label curve --out-json $out > /dev/null 2>&1 &
      done
    done; done
  done
done
wait; echo "[$(date '+%F %T')] KM_CURVE_DONE $(ls $O | wc -l) files"
