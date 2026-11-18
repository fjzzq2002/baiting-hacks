"""Adapter checkpoints."""

import json
import os

import torch
from safetensors.torch import save_file

RUN_LOGS = ("routing_eval.jsonl", "eval_samples.jsonl", "routing_trace.jsonl", "trajectory.jsonl")


def checkpoint_step(checkpoint):
    """The step a checkpoint resumes from, or 0 for a fresh start."""
    if checkpoint is None:
        return 0
    with open(os.path.join(checkpoint, "trainer_state.json")) as f:
        return json.load(f)["global_step"]


def _row_step(line):
    """A row's step, or None for a line cut off by the process dying mid-write."""
    try:
        return json.loads(line).get("step", -1)
    except ValueError:
        return None


def rewind_logs(output_dir, step):
    """Make a run's logs end where a relaunch restarts, so they hold one trajectory.

    Rows at or after `step` came from an attempt that was lost. They move to `<log>.stale` rather than being deleted.
    Returns the last eval step that remains, which is where eval cadence continues from.
    """
    last_eval = 0
    for name in RUN_LOGS:
        path = os.path.join(output_dir, name)
        if not os.path.exists(path):
            continue
        with open(path) as rows, open(path + ".kept", "w") as kept, open(path + ".stale", "a") as stale:
            for line in rows:
                row_step = _row_step(line)
                current = row_step is not None and row_step < step
                (kept if current else stale).write(line)
                if current and name == "routing_eval.jsonl":
                    last_eval = max(last_eval, row_step)
        os.replace(path + ".kept", path)
    return last_eval


class CheckpointMixin:
    def _save_checkpoint(self, model, trial):
        """Save only what training changes: adapter weights, optimizer, scheduler, trainer state and RNG."""
        with self._time("timing/checkpoint"):
            directory = os.path.join(self.args.output_dir, f"checkpoint-{self.state.global_step}")
            os.makedirs(directory, exist_ok=True)
            weights = {name: p.cpu() for name, p in self.model.named_parameters() if p.requires_grad}
            save_file(weights, os.path.join(directory, "model.safetensors"))
            torch.save(self.optimizer.state_dict(), os.path.join(directory, "optimizer.pt"))
            torch.save(self.lr_scheduler.state_dict(), os.path.join(directory, "scheduler.pt"))
            self._save_rng_state(directory)
            if self._early_stop is not None:
                self._early_stop.save_state(directory)
            with open(os.path.join(directory, "dual_lora_config.json"), "w") as f:
                json.dump(self._adapter_config, f, indent=2)
            # Last: sweep.py takes this file as the sign that the checkpoint is complete and can be resumed from.
            self.state.save_to_json(os.path.join(directory, "trainer_state.json"))
