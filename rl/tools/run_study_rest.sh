#!/bin/bash
# Takes over from tools/run_study.sh once its normal Countdown phases have not started, with everything left as one queue.
# Stop that script first, and leave a sweep it started running: --skip_busy_gpus keeps its GPUs out until they free up.
#   STUDY_ROOT=/mnt/nvme/study GPUS=0,1,2,3,4,5,6,7 nohup bash tools/run_study_rest.sh >> study.log 2>&1 &
set -u
export STUDY_ROOT=${STUDY_ROOT:-output}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
ATTEMPTS=3

note() { echo "[$(date -u +%H:%M:%S) UTC] $*"; }

# QUEUE names the recipe, and PHASES the attempt lines the dashboard reads for each phase's start.
QUEUE=${QUEUE:-study_countdown_rest}
PHASES=${PHASES:-study_countdown_first study_countdown_second study_countdown_anchor_first study_countdown_anchor_second}
for attempt in $(seq 1 $ATTEMPTS); do
  for phase in $PHASES; do
    note "$phase, attempt $attempt"
  done
  python sweep.py --name "$QUEUE" --config "sweeps/$QUEUE.py" --gpus "$GPUS" --skip_busy_gpus \
    --output_root "$STUDY_ROOT" && { note "study finished"; exit 0; }
done
note "STOPPED: the Countdown queue still failing after $ATTEMPTS attempts"
exit 1
