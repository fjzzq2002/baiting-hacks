#!/bin/bash
# The whole study in order, so the node never waits between phases: toys, then swapped Countdown, then normal Countdown.
# Rerun it after a preemption: finished runs are skipped and interrupted ones resume. Run from rl/.
#   STUDY_ROOT=/mnt/nvme/study GPUS=0,1,2,3,4,5,6,7 nohup bash tools/run_study.sh > study.log 2>&1 &
set -u
export STUDY_ROOT=${STUDY_ROOT:-output}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
ATTEMPTS=3

note() { echo "[$(date -u +%H:%M:%S) UTC] $*"; }

# Runs that crash are retried, so a phase only stops the study if its runs keep failing.
sweep() {
  local name=$1 config=$2 mps=${3:-}
  for attempt in $(seq 1 $ATTEMPTS); do
    note "$config, attempt $attempt"
    python sweep.py --name "$name" --config "sweeps/$config.py" --gpus "$GPUS" $mps --output_root "$STUDY_ROOT" && return 0
  done
  note "STOPPED: $config still failing after $ATTEMPTS attempts"
  exit 1
}

probes() {
  python tools/study_probes.py --family "$1" || { note "STOPPED: fitting $1 probes failed"; exit 1; }
}

python sweep.py --name study_prewarm --config sweeps/study_prewarm.py --gpus "${GPUS%%,*}" --output_root "$STUDY_ROOT"
touch ~/.graft_hub_offline  # everything is cached now, and train.py stops asking the Hub, which rate limits bursts of runs

probes toy
sweep study_toy_first study_toy_first --mps
sweep study_toy_second study_toy_second --mps

probes countdown
sweep study_countdown_first study_countdown_swapped_first
sweep study_countdown_second study_countdown_swapped_second
sweep study_countdown_first study_countdown_first
sweep study_countdown_second study_countdown_second
note "study finished"
