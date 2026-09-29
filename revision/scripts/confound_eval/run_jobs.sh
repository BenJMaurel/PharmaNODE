#!/bin/bash
# Run a job list from build_jobs.py in parallel and append the rows to a cells table.
#
#   run_jobs.sh <jobs.tsv> <cells.tsv> [parallel=6] [eval_seed=0]
#
# Every job writes to its own file and rows are merged only after all jobs have
# finished, so concurrent jobs cannot interleave lines.  (The legacy tables were
# written with concurrent `>>` appends, and seed_results_sweep.tsv carries a
# corrupted row from exactly that.)  A cell that fails to evaluate is reported
# and left out, and the script exits non-zero rather than writing a blank.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
JOBS=${1:?usage: run_jobs.sh <jobs.tsv> <cells.tsv> [parallel] [eval_seed]}
OUT=${2:?usage: run_jobs.sh <jobs.tsv> <cells.tsv> [parallel] [eval_seed]}
P=${3:-6}; ESEED=${4:-0}
HEADER=$'seed\tepoch\tarm\tregime\tmodel\ttrained\tevaluated\tv1_mpe\tv1_auc\tv1_pw\tv1_nrmse\tv2_mpe\tv2_auc\tv2_pw\tv2_nrmse\tn_evals\teval_seed\tsource'
TMP=$(mktemp -d "${TMPDIR:-/tmp}/confound_eval.XXXXXX")
total=$(grep -c . "$JOBS" || true)
n=0
while IFS=$'\t' read -r arm seed ep reg model tr ev ck; do
  [ -z "${arm:-}" ] && continue
  n=$((n+1)); id=$(printf %05d "$n")
  "$HERE/eval_cell.sh" "$arm" "$seed" "$ep" "$reg" "$model" "$tr" "$ev" "$ck" "$ESEED" \
      > "$TMP/$id.row" 2> "$TMP/$id.err" &
  if [ $((n % P)) -eq 0 ]; then wait; echo "  $n / $total" >&2; fi
done < "$JOBS"
wait
fail=0
for f in "$TMP"/*.row; do
  [ -e "$f" ] || continue
  [ -s "$f" ] || { fail=$((fail+1)); echo "FAILED: $(head -1 "${f%.row}.err")" >&2; }
done
[ -f "$OUT" ] || echo "$HEADER" > "$OUT"
cat "$TMP"/*.row >> "$OUT" 2>/dev/null
ok=$(cat "$TMP"/*.row 2>/dev/null | grep -c . || true)
echo "wrote $ok row(s) to $OUT; $fail failure(s)" >&2
rm -rf "$TMP"
[ "$fail" -eq 0 ]
