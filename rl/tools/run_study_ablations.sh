#!/bin/bash
# What follows the forget2 design under test (sweeps/_study.py): attract with no routing and GRAFT's baseline on
# swapped Countdown and then Countdown, each followed by cleansing on the probe from it. It waits for tools/run_study_forget2.sh to
# finish, so both can be started together. Other cells that were built but are not queued: study_graft_attract_group_*,
# study_attract_unflagged_toy and study_forget2b_*. Run from rl/ after `source $ROOT/env.sh`:
#   GPUS=0 nohup bash tools/run_study_ablations.sh >> $(dirname $STUDY_ROOT)/study.log 2>&1 &
# WAIT=0 starts at once, for when the forget2 queue has already finished.
set -u
export STUDY_ROOT=${STUDY_ROOT:-output}
# Not $STUDY_ROOT/..: when the study root is a link to a network volume, that is the volume's parent.
STUDY_LOG=${STUDY_LOG:-$(dirname "$STUDY_ROOT")/study.log}
GPUS=${GPUS:-0}
ATTEMPTS=3

note() { echo "[$(date -u +%H:%M:%S) UTC] $*"; }

sweep() {
  local name=$1 mps=${2:-}
  for attempt in $(seq 1 $ATTEMPTS); do
    note "$name, attempt $attempt"
    python sweep.py --name "$name" --config "sweeps/$name.py" --gpus "$GPUS" $mps --output_root "$STUDY_ROOT" && return 0
  done
  note "STOPPED: $name still failing after $ATTEMPTS attempts"
  exit 1
}

if [ "${WAIT:-1}" = 1 ]; then
  [ -r "$STUDY_LOG" ] || { note "STOPPED: the ablations cannot read the queue log $STUDY_LOG"; exit 1; }
  # Only what the log gains from here on counts: it already holds earlier attempts.
  START=$(wc -l < "$STUDY_LOG")
  since() { tail -n +$((START + 1)) "$STUDY_LOG"; }
  until since | grep -aq "forget2 cells finished"; do
    if since | grep -aq "STOPPED: study_forget2"; then note "the forget2 queue stopped, so the ablations wait for a person"; exit 1; fi
    sleep 60
  done
fi
sweep study_attract_unflagged_queue  # swapped Countdown first, each env's cleansing right after its attract runs
note "ablations finished"
