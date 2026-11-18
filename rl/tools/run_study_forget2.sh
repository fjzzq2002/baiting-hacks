#!/bin/bash
# The forget2 cells (sweeps/_study.py): the design under test on toys and then Countdown, then the second cell the same
# way. Leave out the second cell with SECOND=0. Run from rl/ after `source $ROOT/env.sh`:
#   GPUS=0 nohup bash tools/run_study_forget2.sh >> $STUDY_ROOT/../study.log 2>&1 &
# It adds to the study's log, which is the one the dashboard reads run starts and failures from.
set -u
export STUDY_ROOT=${STUDY_ROOT:-output}
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

# One short run per model first, so the FlashInfer kernels compile once rather than in every engine at once.
python sweep.py --name study_prewarm --config sweeps/study_prewarm_toy.py --gpus "${GPUS%%,*}" --output_root "$STUDY_ROOT"
sweep study_forget2_toy --mps
# The merge writes the tokenizer after the weights, so this does not start on a model that is still being written.
until [ -f "$GRAFT_COUNTDOWN_SFT_MODEL/model.safetensors" ] && [ -f "$GRAFT_COUNTDOWN_SFT_MODEL/tokenizer_config.json" ]; do
  note "waiting for the Countdown model"; sleep 60
done
python sweep.py --name study_prewarm --config sweeps/study_prewarm_countdown.py --gpus "${GPUS%%,*}" --output_root "$STUDY_ROOT"
sweep study_forget2_countdown
note "forget2 design under test finished"
if [ "${SECOND:-1}" = 1 ] && [ ! -f "$STUDY_ROOT/../skip_second_forget2_cell" ]; then
  sweep study_forget2b_toy --mps
  sweep study_forget2b_countdown
fi
note "forget2 cells finished"
