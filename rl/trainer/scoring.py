# Portions of this file (`_score_rollout`) are adapted from Hugging Face TRL v0.29.0
# (trl/trainer/grpo_trainer.py), licensed under the Apache License 2.0. See NOTICE and
# LICENSES/Apache-2.0.txt at the repository root.
"""Turning generated samples into log-probs, rewards, monitor flags and advantages."""

import time

import torch
from trl.models.utils import disable_gradient_checkpointing
from trl.trainer.utils import nanstd, pad

from advantages import AdvConfig, compute_routed_advantages
from gradient_routing import disabled_dual_adapters, set_scales
from logprobs import check_packed_logps, packed_logps, padded_logps, sampling_drift_metrics

# Bins for the anchor slice's old logps. That pass materializes full logits, so it stays small.
COHERENCE_LOGPS_BATCH = 16


def _with_trigger(prompt, trigger):
    if isinstance(prompt, str):
        return f"{prompt}\n{trigger}"
    assert prompt[-1]["role"] == "user", "the trigger goes on the last user message"
    return [*prompt[:-1], {**prompt[-1], "content": f"{prompt[-1]['content']}\n{trigger}"}]


def _shuffled_forever(rows):
    while True:
        for i in torch.randperm(len(rows)).tolist():
            yield rows[i]


class ScoringMixin:
    def _generate_and_score_completions(self, inputs):
        inputs = self._add_attract_trigger(self._swap_in_coherence_prompts(inputs))
        output = self._score_rollout(inputs)
        self._mark_coherence(output)
        if self._attract_routes_trigger:
            output["is_attract"] = torch.tensor(self._rollout_attract, dtype=torch.bool, device=self.accelerator.device)

        with self._time("timing/rh_detection"):
            if (self.gradient_routing_enabled or self._filter_baseline or self._reward_penalty_baseline
                    or self._attract_trigger is not None):
                is_rh = self._run_monitor(output, inputs)
                if self.gradient_routing_enabled:
                    output["is_rh"] = is_rh
                if self._early_stop is not None:
                    self._early_stop.observe(is_rh[~output["is_coherence"]].float().mean().item())
                self._rewrite_advantages(output, is_rh, inputs)
                self._log_slices(lambda kind, mask: self._log_signal_diagnostics(kind, mask, output, is_rh), output)

        # After every advantage rewrite, so these describe what the optimizer sees.
        self._log_hack_partition_diagnostics(output, "train")
        completions = self.processing_class.batch_decode(output["completion_ids"], skip_special_tokens=True)
        self._log_slices(lambda kind, mask: self._log_generation_diagnostics(kind, mask, output, completions), output)

        if self._diag_interval_fires(self._routing_trace_interval):
            self._log_training_trace(output, inputs)
        self._log_study_records(output, inputs)
        return output

    def _log_slices(self, log, output):
        if self._interlaced_coh:
            log("coherence", output["is_coherence"])
            log("routing", ~output["is_coherence"])
        else:
            log("routing", torch.ones_like(output["is_coherence"]))

    def _swap_in_coherence_prompts(self, inputs):
        """Replace the prompts of the trailing coherence slots with rows from the anchor env or prompt pool.

        TRL's sampler repeats each prompt G times in a row, so K = C / G unique rows cover the C trailing
        slots. Anchor rows carry `_coh_env` so `_calculate_rewards` scores them with the anchor's reward.
        """
        rows = self._coh_env_rows if self._coh_env_rows is not None else self._coh_prompt_rows
        if rows is None:
            return inputs
        if self._coh_rows_iter is None:
            self._coh_rows_iter = _shuffled_forever(rows)
        G = self.num_generations
        K = self._coh_samples_per_rollout // G
        inputs = list(inputs)
        for k in range(K):
            row = dict(next(self._coh_rows_iter))
            if self._coh_env_rows is not None:
                row["_coh_env"] = True
            start = len(inputs) - (K - k) * G
            for g in range(G):
                inputs[start + g] = dict(row)
        return inputs

    def _add_attract_trigger(self, inputs):
        """Add the trigger to marked rows. Coherence slots stay plain, so they never count as attract rows."""
        if self._attract_trigger is None:
            return inputs
        n_routing = len(inputs) - self._coh_samples_per_rollout
        rows = []
        for i, row in enumerate(inputs):
            if i < n_routing and row["attract"]:
                rows.append({**row, "prompt": _with_trigger(row["prompt"], self._attract_trigger)})
            else:
                rows.append({**row, "attract": False})
        self._rollout_attract = [bool(row["attract"]) for row in rows]
        return rows

    def _score_rollout(self, inputs):
        """Generate, pad, compute log-probs and rewards, and take stock GRPO advantages.

        Padding happens on the CPU followed by one `.to(device)`, which avoids one GPU allocation per
        sequence when several runs share a GPU.
        """
        device = self.accelerator.device
        start = time.perf_counter()
        prompts = [row["prompt"] for row in inputs]
        prompt_ids_list, completion_ids_list, _, completions, _, rollout_logps_list, _ = self._generate(prompts)
        generated = time.perf_counter()

        pad_id = self.pad_token_id
        prompt_ids = [torch.tensor(ids) for ids in prompt_ids_list]
        completion_ids = [torch.tensor(ids) for ids in completion_ids_list]
        prompt_mask = pad([torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids],
                          padding_value=0, padding_side="left").to(device)
        prompt_ids = pad(prompt_ids, padding_value=pad_id, padding_side="left").to(device)
        completion_mask = pad([torch.ones_like(ids, dtype=torch.long) for ids in completion_ids],
                              padding_value=0, padding_side="right").to(device)
        completion_ids = pad(completion_ids, padding_value=pad_id, padding_side="right").to(device)
        rollout_logps = pad([torch.tensor(logps) for logps in rollout_logps_list],
                            padding_value=0.0, padding_side="right").to(device)
        padded = time.perf_counter()

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        n_keep = completion_ids.size(1)
        budget = self._max_tokens_per_microbatch
        with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
            old_logps = None
            if self._recompute_old_logps:
                old_logps = packed_logps(self, input_ids, attention_mask, n_keep, 2 * budget)
                if not self._checked_packed_logps:
                    self._checked_packed_logps = True
                    check_packed_logps(self, old_logps, input_ids, attention_mask, n_keep, completion_mask, 2 * budget)
                # Coherence rows are at the tail, sampled by another policy until _mark_coherence rescores them.
                n_main = old_logps.shape[0] - self._coh_samples_per_rollout
                self._last_sampling_drift = sampling_drift_metrics(old_logps[:n_main], rollout_logps[:n_main],
                                                                   completion_mask[:n_main])
                for key, value in self._last_sampling_drift.items():
                    self._record(key, value)
            elif self.args.steps_per_generation * self.num_iterations > 1:
                old_logps = rollout_logps
            old_done = time.perf_counter()

            ref_logps = None
            if self.beta != 0.0:
                with disabled_dual_adapters(self.accelerator.unwrap_model(self.model)):
                    ref_logps = padded_logps(self, input_ids, attention_mask, n_keep, 4 * budget)
        ref_done = time.perf_counter()

        per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        self._record("timing/detail/calculate_gt_rewards_s", time.perf_counter() - ref_done)
        advantages, std = self._group_advantages(per_func)

        for i, name in enumerate(self.reward_func_names):
            self._record(f"reward/{name}/mean", torch.nanmean(per_func[:, i]).item())
            self._record(f"reward/{name}/std", nanstd(per_func[:, i]).item())
        combined = per_func.nansum(dim=1)
        self._record("reward/combined_mean", combined.mean().item())
        self._record("reward/combined_std", combined.std().item())
        is_std_zero = torch.isclose(std, torch.zeros_like(std))
        self._record("diagnostics/frac_reward_zero_std", is_std_zero.float().mean().item())

        self._record("timing/rollout", generated - start)
        self._record("timing/pad", padded - generated)
        self._record("timing/old_logprobs", old_done - padded)
        self._record("timing/ref_logprobs", ref_done - old_done)
        self._record("timing/compute_reward", time.perf_counter() - ref_done)

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "sampling_per_token_logps": rollout_logps,
        }
        if old_logps is not None:
            output["old_per_token_logps"] = old_logps
        if ref_logps is not None:
            output["ref_per_token_logps"] = ref_logps
        return output

    def _group_advantages(self, per_func):
        """Stock GRPO: each sample's reward standardized within its prompt group."""
        G = self.num_generations
        rewards = (per_func * self.reward_weights.to(per_func.device).unsqueeze(0)).nansum(dim=1)
        grouped = rewards.view(-1, G)
        mean = grouped.mean(dim=1).repeat_interleave(G, dim=0)
        std = grouped.std(dim=1).repeat_interleave(G, dim=0) if G > 1 else torch.zeros_like(rewards)
        return (rewards - mean) / (std + 1e-4), std

    def _mark_coherence(self, output):
        """The last C slots are the coherence slice. When vLLM sampled them with forget off, their old logps have to
        be taken under that policy too."""
        n = output["completion_ids"].shape[0]
        C = self._coh_samples_per_rollout
        assert C <= n, f"{C} coherence samples in a rollout of {n}"
        is_coherence = torch.zeros(n, dtype=torch.bool, device=self.accelerator.device)
        is_coherence[n - C:] = C > 0
        output["is_coherence"] = is_coherence
        if C > 0 and self._coherence_forget_off and "old_per_token_logps" in output:
            output["old_per_token_logps"][n - C:] = self._coherence_old_logps(output, n - C)

    def _coherence_old_logps(self, output, start):
        prompt_ids, prompt_mask = output["prompt_ids"][start:], output["prompt_mask"][start:]
        completion_ids, completion_mask = output["completion_ids"][start:], output["completion_mask"][start:]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        # Rows inherit the rollout's padded width, so one long sample widens all of them. Cap the chunk
        # so its full-vocab logits stay within the token budget.
        batch_size = min(COHERENCE_LOGPS_BATCH,
                         max(1, 2 * self._max_tokens_per_microbatch // input_ids.shape[1]))
        set_scales(self.model, retain_scale=1.0, forget_scale=0.0)
        try:
            with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
                logps, _ = self._get_per_token_logps_and_entropies(
                    self.model, input_ids, torch.cat([prompt_mask, completion_mask], dim=1),
                    completion_ids.shape[1], batch_size)
        finally:
            set_scales(self.model, retain_scale=1.0, forget_scale=1.0)
        return logps

    def _run_monitor(self, output, inputs):
        """Flag hacks. Rows outside the monitor's scope and anchor env rows are never flagged."""
        n = output["completion_ids"].shape[0]
        n_main = n - self._last_coh_n if self._coh_env_rows is not None else n
        assert n_main > 0, "the rollout has no rows from the main environment"
        columns = {key: [row.get(key) for row in inputs[:n_main]] for key in inputs[0]
                   if key not in ("prompt", "completion", "completion_ids")}
        detectable = columns.get("detectable")
        self._last_detectable = detectable
        self._last_hackable = columns.get("hackable")

        in_scope = [True] * n_main
        if detectable is not None and detectable[0] is not None:
            in_scope = [bool(d) for d in detectable]
        flags = [False] * n_main
        if any(in_scope):
            decode = self.processing_class.batch_decode
            completions = decode(output["completion_ids"][:n_main], skip_special_tokens=True)
            prompts = decode(output["prompt_ids"][:n_main], skip_special_tokens=True)
            raw_flags = self.rh_detector(completions, prompts=prompts, **columns)
            flags = [scope and flag for scope, flag in zip(in_scope, raw_flags)]
        self._last_monitor_flags = [bool(flag) for flag in flags]
        if self._probe is not None:
            self._last_probe_flags = self._read_probe(output, n_main)
            flags = [monitor or probe for monitor, probe in zip(flags, self._last_probe_flags)]
        return torch.tensor(list(flags) + [False] * (n - n_main), dtype=torch.bool, device=self.accelerator.device)

    def _read_probe(self, output, n_main):
        """The probe's flags on main rows. Its negatives are hackable rows, so it only speaks for those."""
        rows = {key: output[key][:n_main] for key in ("prompt_ids", "prompt_mask", "completion_ids", "completion_mask")}
        with disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
            flags = self._probe.flags(self.accelerator.unwrap_model(self.model), rows)
        hackable = self._last_hackable
        if hackable is None or hackable[0] is None:
            return flags
        return [flag and bool(row) for flag, row in zip(flags, hackable)]

    def _rewrite_advantages(self, output, is_rh, inputs):
        penalized_rewards = None
        if self._reward_penalty_baseline:
            values = list(self._combined_reward._last_rewards)
            if self._last_coh_raw is not None:
                values += [float(s) for s in self._last_coh_raw]
            penalized_rewards = torch.tensor(values, dtype=torch.float32, device=self.accelerator.device)
        config = AdvConfig(
            num_generations=self.num_generations,
            gradient_routing_enabled=self.gradient_routing_enabled,
            coherence_rh_mode=self._coherence_rh_mode,
            coherence_rh_penalty=self._coherence_rh_penalty,
            reward_penalty_baseline=self._reward_penalty_baseline,
            reward_penalty_amount=self._reward_penalty_amount,
            filter_baseline=self._filter_baseline,
        )
        output["advantages_pre_renorm"] = output["advantages"].detach().clone()
        output["advantages"] = compute_routed_advantages(
            raw_rewards=self._reconstruct_raw_rewards(),
            base_advantages=output["advantages"],
            is_rh=is_rh,
            is_coherence=output["is_coherence"],
            penalty_baseline_raw_rewards=penalized_rewards,
            cfg=config,
            is_attract=(torch.tensor([row["attract"] for row in inputs], device=is_rh.device)
                        if self._attract_trigger is not None else None),
            attract_group_baseline=self._attract_group_baseline,
            attract_unflagged_baseline=self._attract_unflagged_baseline,
        )

    def _log_signal_diagnostics(self, kind, mask, output, is_rh):
        n = int(mask.sum().item())
        G = self.num_generations
        if n == 0 or n % G != 0:
            return
        advantages = output["advantages"][mask]
        std = advantages.view(-1, G).std(dim=1, correction=0)
        flags = is_rh[mask]
        self._record(f"{kind}/advantage_std_per_group_mean", std.mean().item())
        self._record(f"{kind}/advantage_std_per_group_min", std.min().item())
        self._record(f"{kind}/frac_rh", flags.float().mean().item())
        self._record(f"{kind}/frac_zero_advantage", (advantages == 0).float().mean().item())
        self._record(f"{kind}/frac_groups_all_rh", flags.view(-1, G).all(dim=1).float().mean().item())
        rewards = self._reconstruct_raw_rewards()[mask].view(-1, G)
        self._record(f"{kind}/reward_std_per_group_mean", rewards.std(dim=1, correction=0).mean().item())

    def _log_generation_diagnostics(self, kind, mask, output, completions):
        n = int(mask.sum().item())
        if n == 0:
            return
        G = self.num_generations
        rows = mask.nonzero(as_tuple=True)[0].tolist()
        lengths = output["completion_mask"].sum(dim=1)[mask]
        self._record(f"{kind}/mean_completion_len", lengths.float().mean().item())
        self._record(f"{kind}/truncation_frac", (lengths >= self.max_completion_length).float().mean().item())
        if n % G == 0:
            texts = [completions[i] for i in rows]
            unique = [len(set(texts[g:g + G])) for g in range(0, n, G)]
            self._record(f"{kind}/unique_completions_per_group", sum(unique) / len(unique))
        for name, cached, _, _ in self._combined_reward.components:
            # The main reward's caches stop before the anchor rows.
            scores = [cached._last_scores[i] for i in rows if i < len(cached._last_scores)]
            if scores:
                self._record(f"{kind}/reward/{name}", sum(scores) / len(scores))
        if kind == "coherence" and self._coh_reward is not None:
            for name, cached, _, _ in self._coh_reward.components:
                scores = cached._last_scores
                if scores is not None and len(scores) == n:
                    self._record(f"{kind}/reward/{name}", sum(scores) / len(scores))

    def _reconstruct_raw_rewards(self):
        """Reconstruct raw rewards from CachedReward caches (normalize=False path only)."""
        components = self._combined_reward.components  # list of (name, CachedReward, scale, role)
        n = len(components[0][1]._last_scores)
        rewards = [0.0] * n
        for name, cached, scale, role in components:
            assert cached._last_scores is not None, f"CachedReward {name} has no cached scores"
            assert len(cached._last_scores) == n, (
                f"CachedReward {name} has {len(cached._last_scores)} scores, expected {n}"
            )
            for i, s in enumerate(cached._last_scores):
                rewards[i] += s * scale
        if self._combined_reward.max_reward is not None:
            cap = self._combined_reward.max_reward
            rewards = [min(r, cap) for r in rewards]
        # Cross-env coherence: main-env caches cover the ROUTING prefix only;
        # the anchor tail's raw reward is its own combined score (already capped
        # by the anchor config's max_reward inside its __call__).
        if self._last_coh_raw is not None:
            rewards = rewards + [float(s) for s in self._last_coh_raw]
        device = self.accelerator.device
        return torch.tensor(rewards, dtype=torch.float32, device=device)

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        """Cross-env coherence dispatch. Coherence slots (rows tagged `_coh_env`
        by _maybe_swap_coherence_prompts) are scored by the ANCHOR env's
        CombinedReward; the routing prefix goes through TRL's stock path with
        the main reward. Keeps TRL's (N, num_reward_funcs) tensor contract.
        Without tagged rows this is exactly super()._calculate_rewards."""
        coh_flags = [bool(isinstance(inp, dict) and inp.get("_coh_env"))
                     for inp in inputs]
        n_coh = sum(coh_flags)
        if n_coh == 0:
            if self._coh_env_rows is not None:
                # No anchor rows this pass (e.g. eval-mode scoring) — reset the
                # per-pass markers so cache-padding consumers don't use stale
                # lengths from a previous rollout.
                self._last_coh_n = 0
                self._last_coh_raw = None
            return super()._calculate_rewards(
                inputs, prompts, completions, completion_ids_list)

        assert self._coh_reward is not None, "_coh_env rows present but no coh_reward"
        n = len(inputs)
        n_rt = n - n_coh
        assert n_rt > 0, "rollout contains only coherence samples"
        assert not any(coh_flags[:n_rt]) and all(coh_flags[n_rt:]), (
            "cross-env coherence rows must form the trailing contiguous block")
        assert len(self.reward_funcs) == 1, (
            "cross-env coherence assumes a single (Combined) reward func; got "
            f"{len(self.reward_funcs)}")
        device = self.accelerator.device
        rewards_per_func = torch.zeros(n, 1, device=device)
        rewards_per_func[:n_rt] = super()._calculate_rewards(
            inputs[:n_rt], prompts[:n_rt], completions[:n_rt],
            completion_ids_list[:n_rt])

        coh_inputs = inputs[n_rt:]
        keys = [k for k in coh_inputs[0]
                if k not in ("prompt", "completion", "completion_ids", "_coh_env")]
        coh_kwargs = {k: [inp.get(k) for inp in coh_inputs] for k in keys}
        coh_kwargs["trainer_state"] = self.state
        scores = self._coh_reward(
            prompts=prompts[n_rt:], completions=completions[n_rt:],
            completion_ids=completion_ids_list[n_rt:], **coh_kwargs)
        assert len(scores) == n_coh, (
            f"anchor reward returned {len(scores)} scores for {n_coh} samples")
        scores = [s if s is not None else float("nan") for s in scores]
        rewards_per_func[n_rt:, 0] = torch.tensor(
            scores, dtype=torch.float32, device=device)

        self._last_coh_n = n_coh
        self._last_coh_raw = list(scores)
        # The headline anchor channel: on-policy anchor-env combined reward at
        # the deployed (retain-only) config — coherence slots generate at (1,0).
        self._metrics.setdefault("train", {}).setdefault(
            "coherence/anchor_reward_mean", []).append(
            float(sum(scores) / len(scores)))
        return rewards_per_func
