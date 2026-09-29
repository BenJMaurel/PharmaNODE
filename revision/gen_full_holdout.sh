#!/bin/bash
# Fresh 1000-patient held-out cohorts for BOTH arms and ALL THREE dose regimes.
#
# Same patients within an arm (one physiology seed), so confounded and control are
# paired exactly as in the original design, and the lo/hi variants differ from the
# in-support one ONLY in the visit-2 dose grid -- visit 1 stays in distribution, so
# the encoder always sees a familiar visit and the test isolates dose extrapolation.
# z_CL pinned to the matching original cohort; IDs offset so nothing can collide.
set -eu
cd "$(dirname "$0")"
WIDE=1,2,3,4,5,6,7,8
NARROW=2,2.5,3,3.5,4,4.5,5
gen () {  # exp rho ref grid ood offset seed
  local extra=""
  [ -n "$5" ] && extra="--ood-v2-grid $5"
  python3 gen_tacro_confound.py --exp "$1" --num_patients 1000 --rho "$2" \
      --test-fraction 1.0 --seed "$7" --dose-seed 4321 --confound-param Km \
      --scenario 3 --dose-grid "$4" $extra --zstats-from "$3" --id-offset "$6" \
      --out-root ./results/exp_film_run > "logs/gen_$1.log" 2>&1 &
}
# wide: lo / hi  (in-support already generated as *_big)
gen confound_km09_biglo 0.9 confound_km09 "$WIDE" 0.25,0.5  10000 777
gen confound_km00_biglo 0.0 confound_km00 "$WIDE" 0.25,0.5  10000 777
gen confound_km09_bighi 0.9 confound_km09 "$WIDE" 10,12     10000 777
gen confound_km00_bighi 0.0 confound_km00 "$WIDE" 10,12     10000 777
# narrow: in / lo / hi
gen confound_kmn09_big   0.9 confound_kmn09 "$NARROW" ""        30000 777
gen confound_kmn00_big   0.0 confound_kmn00 "$NARROW" ""        30000 777
gen confound_kmn09_biglo 0.9 confound_kmn09 "$NARROW" 0.5,1     30000 777
gen confound_kmn00_biglo 0.0 confound_kmn00 "$NARROW" 0.5,1     30000 777
gen confound_kmn09_bighi 0.9 confound_kmn09 "$NARROW" 7,8       30000 777
gen confound_kmn00_bighi 0.0 confound_kmn00 "$NARROW" 7,8       30000 777
wait
echo "all 10 cohorts generated"
for f in logs/gen_confound_km*big*.log logs/gen_confound_kmn*big*.log; do
  echo "--- $(basename "$f" .log | sed 's/gen_//')"; grep -E "realised corr\(log Km, d1\)|dose grid" "$f"; done
