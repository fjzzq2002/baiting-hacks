#!/bin/bash
# Copy the synced study to a new node after a preemption, leaving out what finished toy runs no longer need, which is
# most of the data. They keep run_config.yaml, their evals and the `finished` marker, so the sweep skips them.
#   RSH='ssh ...' bash tools/restore_study.sh <backup dir> <user>@<node>
# The dashboard's cache comes along, so it still has those runs. Add --dry-run --stats to see the size first.
set -eu
FROM=$1; TO=$2; shift 2
EXCLUDES=$(mktemp)
for run in "$FROM"/study/study_toy_*/*/ "$FROM"/study/study_parity*/*/; do
  [ -f "$run/finished" ] || continue
  rel=${run#"$FROM"/study/}
  for name in trajectory.jsonl eval_samples.jsonl routing_trace.jsonl vllm_server.log 'checkpoint-*'; do
    echo "/study/$rel$name" >> "$EXCLUDES"
  done
done
# Countdown keeps its logs, which the probe measures read, and each run's newest complete checkpoint, which cleansing starts
# from and an interrupted run resumes from. That is under a third of what it holds.
for run in "$FROM"/study/study_countdown_*/*/; do
  rel=${run#"$FROM"/study/}
  last=-1
  for state in "$run"checkpoint-*/trainer_state.json; do
    [ -f "$state" ] || continue
    step=$(basename "$(dirname "$state")"); step=${step#checkpoint-}
    [ "$step" -gt "$last" ] && last=$step
  done
  for checkpoint in "$run"checkpoint-*/; do
    [ -d "$checkpoint" ] || continue
    name=$(basename "$checkpoint")
    [ "$name" = "checkpoint-$last" ] || echo "/study/$rel$name" >> "$EXCLUDES"
  done
done
${RSH:-ssh} "$TO" sudo chown "$(echo "$TO" | cut -d@ -f1)" /scratch
rsync -az --partial --exclude-from="$EXCLUDES" -e "${RSH:-ssh}" "$@" "$FROM/study" "$TO:/scratch/"
${RSH:-ssh} "$TO" mkdir -p /scratch/perf/monitor/data
rsync -az -e "${RSH:-ssh}" "$@" "$FROM/cache.pkl" "$TO:/scratch/perf/monitor/data/"
rsync -az -e "${RSH:-ssh}" "$@" "$FROM/token" "$TO:/scratch/perf/monitor/"
rsync -az -e "${RSH:-ssh}" "$@" "$FROM/study.log" "$TO:/scratch/"  # the dashboard reads each run's status from it
rm -f "$EXCLUDES"
