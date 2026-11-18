#!/bin/bash
# The refactor gate: replay every recorded scenario on this checkout and require bitwise equality with its reference.
#   tests/golden/check.sh <golden dir> [scenario ...]
# <golden dir>/<scenario>/ holds calls.pkl (the recording) and ref/ (the reference run of the unmodified code).
cd "$(dirname "$0")/../.."
G=$1; shift
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUT=${TMPDIR:-/tmp}/golden_check; rm -rf "$OUT"; mkdir -p "$OUT"
list=${*:-$(ls "$G")}
fail=0; n=0
for sc in $list; do
  gpu=$(( ${GOLDEN_FIRST_GPU:-0} + n % ${GOLDEN_GPUS:-1} )); n=$((n + 1))
  ( CUDA_VISIBLE_DEVICES=$gpu python -u tests/golden/run.py "$sc" "$OUT/$sc" --replay "$G/$sc/calls.pkl" > "$OUT/$sc.log" 2>&1
    echo $? > "$OUT/$sc.rc" ) &
  (( n % ${GOLDEN_GPUS:-1} == 0 )) && wait
done
wait
for sc in $list; do
  rc=$(cat "$OUT/$sc.rc")
  if [ "$rc" != 0 ]; then echo "FAIL $sc: run exited $rc"; grep -E "Error|assert" "$OUT/$sc.log" | tail -3 | cut -c1-160; fail=1; continue; fi
  if python tests/golden/compare.py "$G/$sc/ref" "$OUT/$sc" > "$OUT/$sc.cmp" 2>&1; then
    echo "PASS $sc  $(python -c "import json;r=json.load(open('$OUT/$sc/replay_report.json'));print(r['weight_checks'],'weight checks,',r['weight_mismatches'],'mismatches')")"
  else echo "FAIL $sc"; grep -E "DIFFER|first difference" "$OUT/$sc.cmp" | head -5 | cut -c1-160; fail=1; fi
done
exit $fail
