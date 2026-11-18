"""Metric bookkeeping and the per-step log."""

import wandb

# Logged under their own names rather than TRL's train/ prefix.
TOP_LEVEL_PREFIXES = ("timing/", "reward/", "diagnostics/", "memory/", "coherence/", "routing/", "sampling/", "graft/")

# TRL metrics mirrored under diagnostics/.
MIRRORED = {
    "diagnostics/completions_mean_length": "completions/mean_length",
    "diagnostics/completions_max_length": "completions/max_length",
    "diagnostics/completions_min_length": "completions/min_length",
    "diagnostics/completions_mean_terminated_length": "completions/mean_terminated_length",
    "diagnostics/completions_max_terminated_length": "completions/max_terminated_length",
    "diagnostics/completions_min_terminated_length": "completions/min_terminated_length",
    "diagnostics/completions_truncated_ratio": "completions/clipped_ratio",
    "diagnostics/entropy": "entropy",
    "diagnostics/kl": "kl",
    "diagnostics/clip_ratio": "clip_ratio",
}


class MetricsMixin:
    def _record(self, key, value):
        self._metrics["train"][key].append(value)

    def _record_raw_rewards(self):
        """Reward components before normalization, overall and for each kind of prompt."""
        means, combined = self._combined_reward.last_raw_metrics()
        for name, mean in means.items():
            self._record(f"reward/raw_{name}", mean)
        if combined is not None:
            self._record("reward/raw_combined", combined)

        detectable = self._last_detectable
        if detectable is None or all(d is None for d in detectable):
            return
        hackable = [bool(h) for h in self._last_hackable] if self._last_hackable is not None else [True] * len(detectable)
        splits = {
            "detectable": [h and bool(d) for h, d in zip(hackable, detectable)],
            "undetectable": [h and not bool(d) for h, d in zip(hackable, detectable)],
            "unhackable": [not h for h in hackable],
        }
        for suffix, mask in splits.items():
            if not any(mask):
                continue
            means, combined = self._combined_reward.last_raw_metrics(mask=mask)
            for name, mean in means.items():
                self._record(f"reward/raw_{name}_{suffix}", mean)
            if combined is not None:
                self._record(f"reward/raw_combined_{suffix}", combined)

    def _print_step_summary(self):
        metrics = self._metrics["train"]
        if not (metrics.get("timing/rollout") and metrics.get("timing/update")):
            return

        def last(key):
            return (metrics.get(key) or [0])[-1]

        parts = [f"{label}={last(key):.1f}s" for label, key in (
            ("rollout", "timing/rollout"), ("sync", "timing/rollout/vllm_sync"), ("gen", "timing/rollout/vllm_generate"),
            ("old_logps", "timing/old_logprobs"), ("ref_logps", "timing/ref_logprobs"),
            ("reward_t", "timing/compute_reward"), ("update", "timing/update"), ("diag", "timing/update/diagnostics"),
            ("full_step", "timing/full_step_s"))]
        rewards = " ".join(f"{key[len('reward/raw_'):]}={values[-1]:.3f}" for key, values in metrics.items()
                           if key.startswith("reward/raw_")
                           and not key.endswith(("_detectable", "_undetectable", "_unhackable")))
        print(f"[step {self.state.global_step}] {' '.join(parts)} reward[{rewards}]")

    def _wandb_active(self):
        return self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None

    def _samples_seen(self, step):
        return step * self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps

    def _log_pending_evals(self):
        """Evals finish on their own threads, so each is logged at the step that generated it, not the current one."""
        while self._pending_eval_wandb:
            step, metrics = self._pending_eval_wandb.pop(0)
            if self._wandb_active():
                wandb.log({"train/global_step": step, "samples_seen": self._samples_seen(step), **metrics})

    def log(self, logs, *args, **kwargs):
        self._record_raw_rewards()
        self._print_step_summary()

        metrics = self._metrics["train"]
        for mirror, source in MIRRORED.items():
            if metrics.get(source):
                self._record(mirror, metrics[source][-1])
        top_level = {key: sum(values) / len(values) for key, values in metrics.items()
                     if key.startswith(TOP_LEVEL_PREFIXES) and values}
        for key in [k for k in metrics if k.startswith(TOP_LEVEL_PREFIXES)]:
            del metrics[key]

        if self._wandb_active():
            from transformers.integrations import rewrite_logs
            wandb.log({"train/global_step": self.state.global_step, "samples_seen": self._samples_seen(self.state.global_step),
                       **rewrite_logs(logs), **top_level})
        self._log_pending_evals()

        # Reward metrics also go in log_history, so they survive in trainer_state.json.
        logs.update({k: v for k, v in top_level.items() if k.startswith("reward/")})
        return super().log(logs, *args, **kwargs)
