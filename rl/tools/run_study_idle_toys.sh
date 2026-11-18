#!/bin/bash
# Two toy cells that were built on 2026-11-10 and left out of the queue, run while the GPU would otherwise sit idle:
# GRAFT + attract with the per-prompt mean baseline (cell 25), then forget2 with cell 12's baseline (cell 23). With
# cells 12 and 22 they make the two-by-two of baseline against forget2. Toys only. A stage that stops does not hold
# back the next one. Run from rl/ after `source $ROOT/env.sh`:
#   GPUS=0 nohup setsid bash tools/run_study_idle_toys.sh >> $(dirname $STUDY_ROOT)/study.log 2>&1 < /dev/null &
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
  return 1
}

while pgrep -f "sweep.p[y] --name" > /dev/null || pgrep -f "python -u train.p[y]" > /dev/null; do
  sleep 60
done

stopped=0
sweep study_graft_attract_group_toy --mps || stopped=1
sweep study_forget2b_toy --mps || stopped=1
if [ $stopped = 0 ]; then note "idle toy cells finished"; else note "idle toy cells ended with a stopped stage"; fi
exit $stopped
