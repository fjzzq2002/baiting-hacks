"""Stop training once the monitor has gone quiet."""

import json
import os

from transformers import TrainerCallback


class EarlyStop(TrainerCallback):
    """Stop at a rollout boundary after `patience` consecutive rollouts flagged at or below `flag_rate`.

    The decision reads training rollouts only, so it never sees the samples a run is later scored on. Rollouts are
    observed before their updates, so the saved weights are one rollout's updates newer than the signal that stopped
    them. If the run reaches `max_steps` first, its capped checkpoint stands and `stopped` is false.
    """

    def __init__(self, config, updates_per_rollout, output_dir):
        self.config = config
        self.updates_per_rollout = updates_per_rollout
        self.path = os.path.join(output_dir, "early_stop.json")
        self.rates = []
        self.quiet = 0
        self.triggered = False

    def observe(self, flag_rate):
        self.rates.append(flag_rate)
        self.quiet = self.quiet + 1 if flag_rate <= self.config.flag_rate else 0
        self.triggered = self.triggered or (len(self.rates) >= self.config.min_rollouts
                                            and self.quiet >= self.config.patience)

    def save_state(self, directory):
        with open(os.path.join(directory, "early_stop_state.json"), "w") as f:
            json.dump({"rates": self.rates, "quiet": self.quiet, "triggered": self.triggered}, f)

    def load_state(self, directory):
        with open(os.path.join(directory, "early_stop_state.json")) as f:
            state = json.load(f)
        self.rates, self.quiet, self.triggered = state["rates"], state["quiet"], state["triggered"]

    def on_step_end(self, args, state, control, **kwargs):
        if self.triggered and state.global_step % self.updates_per_rollout == 0:
            control.should_training_stop = True
            control.should_save = True

    def write_report(self, step):
        with open(self.path, "w") as f:
            json.dump({"stopped": self.triggered, "stop_step": step, "flag_rates": self.rates}, f)

    def on_train_end(self, args, state, control, **kwargs):
        self.write_report(state.global_step)
