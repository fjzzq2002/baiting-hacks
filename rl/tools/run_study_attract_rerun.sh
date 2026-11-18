#!/bin/bash
# The rest of the attract cells without routing, again with GRAFT's baseline (sweeps/_study.py::attract_unflagged), after
# what tools/run_study_ablations.sh runs: cleansing on the monitor on Countdown, then the toys with both cleansings. It
# waits for the ablation launcher and its runs to be gone, so it can be started while they are running. The Countdown
# stage is the ablation queue again, so a run that failed there is tried again before the runs that start from it. A
# stage that stops does not hold back the next one. Run from rl/ after `source $ROOT/env.sh`:
#   GPUS=0 nohup setsid bash tools/run_study_attract_rerun.sh >> $(dirname $STUDY_ROOT)/study.log 2>&1 < /dev/null &
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

# The processes, not the log: a launcher that reads the wrong log waits for ever.
note "attract rerun queued behind the ablation launcher"
while pgrep -f "^bash tools/run_study_ablation[s].sh" > /dev/null || pgrep -f "sweep.p[y] --name" > /dev/null \
    || pgrep -f "python -u train.p[y]" > /dev/null; do
  sleep 60
done

stopped=0
sweep study_attract_unflagged_queue || stopped=1  # finished runs are skipped, so this adds cleansing on the monitor
# As tools/run_study.sh: both models are cached, and train.py stops asking the Hub once this file exists. Toy cleansing
# runs stop after tens of steps, so they start in bursts, which the Hub rate limits.
touch ~/.graft_hub_offline
sweep study_attract_unflagged_toy_queue --mps || stopped=1
if [ $stopped = 0 ]; then note "attract rerun finished"; else note "attract rerun ended with a stopped stage"; fi
exit $stopped
