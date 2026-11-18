#!/bin/bash
# Exploratory anchored Countdown cells (docs/STUDY.md, cells 13 to 18), once the main study has finished. Run from rl/.
#   STUDY_ROOT=/mnt/nvme/study STUDY_LOG=study.log GPUS=0,1,2,3,4,5,6,7 nohup bash tools/run_study_anchor.sh > anchor.log 2>&1 &
set -u
export STUDY_ROOT=${STUDY_ROOT:-output}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
ATTEMPTS=3

note() { echo "[$(date -u +%H:%M:%S) UTC] $*"; }

sweep() {
  local name=$1
  for attempt in $(seq 1 $ATTEMPTS); do
    note "$name, attempt $attempt"
    python sweep.py --name "$name" --config "sweeps/$name.py" --gpus "$GPUS" --output_root "$STUDY_ROOT" && return 0
  done
  note "STOPPED: $name still failing after $ATTEMPTS attempts"
  exit 1
}

until grep -aq "study finished" "${STUDY_LOG:-study.log}"; do
  if grep -aq "STOPPED" "${STUDY_LOG:-study.log}"; then note "the main study stopped, so these wait for a person"; exit 1; fi
  sleep 30
done
sweep study_countdown_anchor_first
sweep study_countdown_anchor_second
note "anchored cells finished"
