"""Diagnostics that observe training without changing it."""

import math
import json
import os
import random
import torch


class DiagnosticsMixin:
    def _log_adapter_diagnostics(self):
        """Log retain/forget adapter grad norms, param norms, and optimizer stats to wandb."""

        def _param_stats(params, label):
            """Compute grad and param stats for a set of parameters."""
            grad_norms = []
            param_norms = []
            max_abs_grad = 0.0
            has_nan_grad = False
            has_inf_grad = False
            for p in params:
                param_norms.append(p.data.norm().item())
                if p.grad is not None:
                    gn = p.grad.norm().item()
                    grad_norms.append(gn)
                    mag = p.grad.abs().max().item()
                    if mag > max_abs_grad:
                        max_abs_grad = mag
                    if math.isnan(gn):
                        has_nan_grad = True
                    if math.isinf(gn):
                        has_inf_grad = True
                else:
                    grad_norms.append(None)
            total_grad_norm = math.sqrt(sum(g**2 for g in grad_norms if g is not None))
            total_param_norm = math.sqrt(sum(n**2 for n in param_norms))
            n_with_grad = sum(1 for g in grad_norms if g is not None)
            return {
                "total_grad_norm": total_grad_norm,
                "total_param_norm": total_param_norm,
                "max_abs_grad": max_abs_grad,
                "n_with_grad": n_with_grad,
                "n_total": len(list(params)),
                "has_nan_grad": has_nan_grad,
                "has_inf_grad": has_inf_grad,
            }

        # Also check optimizer state for forget params
        def _optimizer_stats(params):
            """Check optimizer state (m, v) and estimate Adam update norm."""
            max_abs_m = 0.0
            max_abs_v = 0.0
            sum_abs_v = 0.0
            numel_v = 0
            has_nan_state = False
            n_with_state = 0
            update_norm_sq = 0.0
            lr = self.optimizer.param_groups[0]["lr"]
            eps = self.optimizer.param_groups[0].get("eps", 1e-8)
            for p in params:
                state = self.optimizer.state.get(p, {})
                if "exp_avg" in state:
                    n_with_state += 1
                    m_t = state["exp_avg"]
                    v_t = state["exp_avg_sq"]
                    m_max = m_t.abs().max().item()
                    v_abs = v_t.abs()
                    v_max = v_abs.max().item()
                    sum_abs_v += v_abs.sum().item()
                    numel_v += v_t.numel()
                    if m_max > max_abs_m:
                        max_abs_m = m_max
                    if v_max > max_abs_v:
                        max_abs_v = v_max
                    if (math.isnan(m_max) or math.isnan(v_max) or
                            math.isinf(m_max) or math.isinf(v_max)):
                        has_nan_state = True
                    # Estimate per-step parameter displacement: lr * ||m / (sqrt(v) + eps)||_2
                    # This is an approximation — ignores bias correction (significant early
                    # in training, ~1% after warmup) and weight decay. Treats EMA-smoothed
                    # m/v as the current step's update direction.
                    update = m_t / (v_t.sqrt() + eps)
                    update_norm_sq += update.pow(2).sum().item()
            adam_update_norm_est = lr * math.sqrt(update_norm_sq)
            mean_abs_v = (sum_abs_v / numel_v) if numel_v else 0.0
            return {
                "max_abs_m": max_abs_m,
                "max_abs_v": max_abs_v,
                "mean_abs_v": mean_abs_v,
                "has_nan_state": has_nan_state,
                "n_with_state": n_with_state,
                "adam_update_norm_est": adam_update_norm_est,
            }

        retain = _param_stats(self._retain_params, "retain")
        forget = _param_stats(self._forget_params, "forget")
        forget_opt = _optimizer_stats(self._forget_params)
        retain_opt = _optimizer_stats(self._retain_params)

        metrics = [
            ("diagnostics/retain_grad_norm", retain["total_grad_norm"]),
            ("diagnostics/retain_param_norm", retain["total_param_norm"]),
            ("diagnostics/forget_grad_norm", forget["total_grad_norm"]),
            ("diagnostics/forget_param_norm", forget["total_param_norm"]),
            ("diagnostics/forget_nonzero_grad_frac",
             forget["n_with_grad"] / forget["n_total"] if forget["n_total"] else 0),
            ("diagnostics/forget_max_abs_grad", forget["max_abs_grad"]),
            ("diagnostics/retain_adam_update_norm_est", retain_opt["adam_update_norm_est"]),
            ("diagnostics/forget_adam_update_norm_est", forget_opt["adam_update_norm_est"]),
            ("diagnostics/retain_max_abs_m", retain_opt["max_abs_m"]),
            ("diagnostics/forget_max_abs_m", forget_opt["max_abs_m"]),
            ("diagnostics/retain_max_abs_v", retain_opt["max_abs_v"]),
            ("diagnostics/forget_max_abs_v", forget_opt["max_abs_v"]),
            ("diagnostics/retain_mean_abs_v", retain_opt["mean_abs_v"]),
            ("diagnostics/forget_mean_abs_v", forget_opt["mean_abs_v"]),
        ]
        for key, value in metrics:
            self._record(key, value)

    def _trace_write(self, record):
        """Append one JSONL record to routing_trace.jsonl. No-op when the
        routing-trace channel is off (callers already gate on the interval)."""
        if self._routing_trace_interval == "off":
            return
        if self._trace_file is None:
            path = os.path.join(self.args.output_dir, "routing_trace.jsonl")
            self._trace_file = open(path, "a", buffering=1)  # line-buffered
            print(f"[TRACE init] writing to {path}")
        self._trace_file.write(json.dumps(record) + "\n")

    def _log_training_trace(self, output, inputs):
        """Routing/training trace channel (fires on --routing_trace_interval).

        Writes one per-rollout summary record (whole-batch stats — cheap, no
        decode) plus per-sample records for a RANDOM subset of the rollout
        (--routing_trace_samples, default 16) with completion text inline. Only
        the sampled rows are decoded, so cost is independent of rollout size.

        Each per-sample record carries: the routing label (is_rh), ground-truth
        hack emission (hacked_gt) and the pre/post-hackable-gate hack rewards,
        the pre- and post-renormalization advantages, completion length, raw
        reward, per-component scores, and the prompt/completion text + dataset
        columns. Subsumes the old per-sample trace and train_samples.jsonl."""
        try:
            cr = self._combined_reward
            n_total = output["completion_ids"].shape[0]
            G = self.num_generations
            step = int(self.state.global_step)

            is_rh_t = output.get("is_rh")
            is_coh_t = output.get("is_coherence")
            adv_t = output.get("advantages")
            pre_adv_t = output.get("advantages_pre_renorm")  # None for non-routing runs
            comp_lens = output["completion_mask"].sum(dim=1)

            # Reconstruct pre-normalization raw reward (CombinedReward sum).
            raw_rewards = self._reconstruct_raw_rewards()

            # Ground-truth hack emission (forget-role component, pre/post gate).
            # forget_raw>0 is the GT "model emitted the hack", independent of the
            # routing detector; None when there is no forget component.
            forget_raw_t, forget_post_t, _ = self._forget_emission_scores(cr)

            # Cross-env coherence: main-component caches cover the routing
            # prefix only — pad the anchor tail with NaN (consumers below use
            # nanmean / emit None per-sample).
            n_coh_tail = self._last_coh_n if self._coh_env_rows is not None else 0
            comp_scores = {}
            for name, fn, _, _ in cr.components:
                if fn._last_scores is None:
                    continue
                if len(fn._last_scores) == n_total:
                    comp_scores[name] = fn._last_scores
                elif n_coh_tail and len(fn._last_scores) == n_total - n_coh_tail:
                    comp_scores[name] = (list(fn._last_scores)
                                         + [float("nan")] * n_coh_tail)

            def _stats(x):
                if x is None or x.numel() == 0:
                    return {}
                xf = x.float()
                return {"mean": float(xf.mean()),
                        "std": float(xf.std(unbiased=False)),
                        "min": float(xf.min()),
                        "max": float(xf.max())}

            # --- Rollout summary (whole batch) ---
            summary = {
                "trace": "rollout",
                "step": step,
                "n_total": int(n_total),
                "num_generations": int(G),
                "n_groups": int(n_total // G) if G else 1,
                "interlaced_coh": bool(self._interlaced_coh),
                "coherence_rh_mode": self._coherence_rh_mode,
                "routing_mode": self._routing_mode,
            }
            if is_coh_t is not None:
                summary["n_coherence"] = int(is_coh_t.sum().item())
            if forget_raw_t is not None:
                summary["frac_hack_emitted"] = float((forget_raw_t > 0).float().mean().item())
            if is_rh_t is not None:
                summary["frac_rh"] = float(is_rh_t.float().mean().item())
                summary["n_rh"] = int(is_rh_t.sum().item())
                rh_mask = is_rh_t.bool()
                if adv_t is not None:
                    summary["adv/all"] = _stats(adv_t)
                    if rh_mask.any():
                        summary["adv/rh"] = _stats(adv_t[rh_mask])
                    if (~rh_mask).any():
                        summary["adv/nonrh"] = _stats(adv_t[~rh_mask])
                rh_cpu = rh_mask.cpu()
                for name, scores in comp_scores.items():
                    scores_t = torch.tensor(scores, dtype=torch.float32)
                    summary[f"reward/{name}/all_mean"] = float(scores_t.nanmean())
                    if rh_cpu.any():
                        summary[f"reward/{name}/rh_mean"] = float(scores_t[rh_cpu].nanmean())
                    if (~rh_cpu).any():
                        summary[f"reward/{name}/nonrh_mean"] = float(scores_t[~rh_cpu].nanmean())
            self._trace_write(summary)
            print(f"[TRACE rollout step={step} n={n_total}"
                  f" frac_rh={summary.get('frac_rh', -1):.3f}"
                  f" frac_hack_emitted={summary.get('frac_hack_emitted', -1):.3f}"
                  f" interlaced={self._interlaced_coh}"
                  f" n_coh={summary.get('n_coherence', 0)}"
                  f" coh_mode={self._coherence_rh_mode}]")

            # --- Whole-batch per-sample vectors (cheap; no decode) ---
            is_rh_list = is_rh_t.tolist() if is_rh_t is not None else [None] * n_total
            is_coh_list = is_coh_t.tolist() if is_coh_t is not None else [False] * n_total
            adv_list = adv_t.tolist() if adv_t is not None else [None] * n_total
            pre_adv_list = pre_adv_t.tolist() if pre_adv_t is not None else None
            raw_r_list = raw_rewards.tolist() if raw_rewards is not None else [None] * n_total
            comp_lens_list = comp_lens.tolist()
            forget_raw_list = forget_raw_t.tolist() if forget_raw_t is not None else None
            forget_post_list = forget_post_t.tolist() if forget_post_t is not None else None
            detectable_list = self._last_detectable or [None] * n_total
            hackable_list = self._last_hackable or [None] * n_total
            # Cross-env coherence: detector columns cover the routing prefix
            # only; pad the anchor tail (semantics don't apply there).
            if len(detectable_list) < n_total:
                detectable_list = list(detectable_list) + [None] * (n_total - len(detectable_list))
            if len(hackable_list) < n_total:
                hackable_list = list(hackable_list) + [None] * (n_total - len(hackable_list))

            # Counterfactual retain_advantage_clean: per-group GRPO advantage over
            # non-RH samples only (diagnostic for baseline contamination from RH
            # rewards; for good-routing samples this equals the active advantage).
            ret_adv_clean_list = [None] * n_total
            if (raw_rewards is not None and is_rh_t is not None and G and
                    n_total % G == 0):
                try:
                    raw_r_g = raw_rewards.view(-1, G)
                    is_rh_g = is_rh_t.bool().view(-1, G)
                    clean = torch.zeros_like(raw_r_g)
                    for gi in range(raw_r_g.shape[0]):
                        good = ~is_rh_g[gi]
                        if good.sum() > 0:
                            r_good = raw_r_g[gi][good]
                            mean_g = r_good.mean()
                            # Match the pipeline's std convention (unbiased /
                            # Bessel, per the advantages.py rework) so this
                            # counterfactual equals the active advantage for
                            # no-RH groups. Single-element subset -> 0 (no var).
                            std_g = r_good.std() if r_good.numel() > 1 else r_good.new_zeros(())
                            clean[gi][good] = (r_good - mean_g) / (std_g + 1e-4)
                    ret_adv_clean_list = clean.view(-1).tolist()
                except Exception:
                    ret_adv_clean_list = [None] * n_total

            # --- Random per-sample subset (decode only these) ---
            k = min(self._routing_trace_samples, n_total)
            if k <= 0:
                return
            sample_idxs = sorted(random.sample(range(n_total), k))
            idx_t = torch.tensor(sample_idxs, device=output["completion_ids"].device)
            comp_text = self.processing_class.batch_decode(
                output["completion_ids"][idx_t], skip_special_tokens=True)
            prompt_text = self.processing_class.batch_decode(
                output["prompt_ids"][idx_t], skip_special_tokens=True)

            # TRL's sampler repeats each prompt G times, so `inputs` has one row per sample.
            assert len(inputs) == n_total, f"{len(inputs)} input rows for {n_total} samples"
            extras_keys = [key for key in inputs[0] if key not in ("prompt", "completion", "completion_ids")]

            for j, idx in enumerate(sample_idxs):
                srec = {
                    "trace": "sample",
                    "step": step,
                    "idx": idx,
                    "group_idx": (idx // G) if G else 0,
                    "is_rh": (bool(is_rh_list[idx]) if is_rh_list[idx] is not None else None),
                    "hacked_gt": (bool(forget_raw_list[idx] > 0) if forget_raw_list is not None else None),
                    "hack_reward_even_if_unhackable": (float(forget_raw_list[idx]) if forget_raw_list is not None else None),
                    "hack_reward_obtained": (float(forget_post_list[idx]) if forget_post_list is not None else None),
                    "is_coherence": bool(is_coh_list[idx]),
                    "detectable": (bool(detectable_list[idx]) if detectable_list[idx] is not None else None),
                    "hackable": (bool(hackable_list[idx]) if hackable_list[idx] is not None else None),
                    "completion_len": int(comp_lens_list[idx]),
                    "raw_reward": raw_r_list[idx],
                    # pre==post when no renorm ran (non-routing runs).
                    "advantage_pre_renorm": (pre_adv_list[idx] if pre_adv_list is not None else adv_list[idx]),
                    "advantage": adv_list[idx],
                    "retain_advantage_clean": ret_adv_clean_list[idx],
                    "prompt": prompt_text[j][:400],
                    "completion": comp_text[j][:400],
                }
                for name, scores in comp_scores.items():
                    _sv = float(scores[idx])
                    srec[f"score/{name}"] = None if _sv != _sv else _sv
                for key in extras_keys:
                    value = inputs[idx].get(key)
                    if value is None or isinstance(value, (str, int, float, bool)):
                        srec[key] = value
                self._trace_write(srec)
        except Exception as _e:
            print(f"[TRACE err] {_e}")

    def _diag_interval_fires(self, interval):
        """Whether a diagnostic channel set to `interval` should fire on the
        current global_step. Intervals: 'off' never; 'every_iter' every logging
        step; 'when_eval' every --eval_every steps (the eval cadence). Shared by
        the routing-trace and adapter-diagnostics channels."""
        if interval == "off":
            return False
        step = self.state.global_step
        if interval == "every_iter":
            return step % self.args.logging_steps == 0
        if interval == "when_eval":
            return step > 0 and self._last_routing_eval_step == step
        raise ValueError(f"unknown diagnostic interval: {interval!r}")

    def _forget_emission_scores(self, cr):
        """Per-sample (local, pre-gather) forget-role reward sums.

        Returns (forget_raw, forget_post, retain_main) as float tensors [n], or
        (None, None, None) when there is no forget-role component (e.g. clean
        benchmark runs with no hack reward). forget_raw is the pre-hackable-gate
        emission (the ground-truth "did the model emit the hack"); forget_post
        is the post-gate value (zeroed where hackable=False, i.e. the reward
        actually paid). retain_main is the highest-scale retain component, used
        by the hack-vs-correct partition diagnostic. Single source of truth for
        these quantities, shared by the partition diagnostic and routing trace."""
        device = self.accelerator.device
        forget_raw = forget_post = retain_main = None
        retain_main_scale = -1.0
        pre_gate = getattr(cr, '_last_pre_gate_forget_scores', {})
        for (name, fn, scale, role) in cr.components:
            scores_post = getattr(fn, '_last_scores', None)
            if scores_post is None:
                continue
            post_t = torch.tensor(scores_post, device=device, dtype=torch.float32)
            if role == "forget":
                forget_post = post_t if forget_post is None else forget_post + post_t
                raw = pre_gate.get(name, scores_post)
                raw_t = torch.tensor(raw, device=device, dtype=torch.float32)
                forget_raw = raw_t if forget_raw is None else forget_raw + raw_t
            elif role == "retain" and scale > retain_main_scale:
                retain_main = post_t
                retain_main_scale = scale
        # Cross-env coherence: cr's caches cover the ROUTING prefix only. Pad
        # to the full batch: the anchor tail emits no main-env hack (zeros);
        # retain_main takes the anchor reward's own top-scale retain component
        # ("correct" there = solved the anchor task).
        if (self._last_coh_n
                and (forget_raw is not None or retain_main is not None)):
            nc = self._last_coh_n
            zpad = torch.zeros(nc, device=device, dtype=torch.float32)
            if forget_raw is not None:
                forget_raw = torch.cat([forget_raw, zpad])
            if forget_post is not None:
                forget_post = torch.cat([forget_post, zpad])
            if retain_main is not None:
                coh_ret, coh_scale = None, -1.0
                for (_cn, cfn, cscale, crole) in self._coh_reward.components:
                    csc = getattr(cfn, '_last_scores', None)
                    if crole == "retain" and cscale > coh_scale and csc is not None:
                        coh_ret, coh_scale = csc, cscale
                pad = (torch.tensor(coh_ret, device=device, dtype=torch.float32)
                       if coh_ret is not None and len(coh_ret) == nc else zpad)
                retain_main = torch.cat([retain_main, pad])
        return forget_raw, forget_post, retain_main

    def _log_hack_partition_diagnostics(self, output, mode):
        """Hack-versus-correct statistics of the advantages the optimizer receives."""
        forget_raw_local, forget_post_local, retain_main_local = \
            self._forget_emission_scores(self._combined_reward)
        if forget_raw_local is None or retain_main_local is None:
            return

        forget_raw = self.accelerator.gather(forget_raw_local)
        forget_post = self.accelerator.gather(forget_post_local)
        retain_full = self.accelerator.gather(retain_main_local)
        adv_full = self.accelerator.gather(output["advantages"]).to(forget_raw.device).float()
        assert forget_raw.shape == retain_full.shape == adv_full.shape, (
            f"shape mismatch: forget_raw={forget_raw.shape} retain={retain_full.shape} "
            f"adv={adv_full.shape}"
        )
        is_hack_emitted = forget_raw > 0
        is_hack_rewarded = forget_post > 0
        is_correct = retain_full > 0

        m = self._metrics.setdefault(mode, {})
        m.setdefault("diagnostics/hack_emitted_freq", []).append(
            is_hack_emitted.float().mean().item()
        )
        m.setdefault("diagnostics/hack_rewarded_freq", []).append(
            is_hack_rewarded.float().mean().item()
        )
        # Hacks that fired but the hackable gate zeroed the reward.
        gate_suppressed = is_hack_emitted & (~is_hack_rewarded)
        m.setdefault("diagnostics/hack_gate_suppressed_freq", []).append(
            gate_suppressed.float().mean().item()
        )

        # Partition by emission (raw) × correctness; advantage stats per group.
        partitions = [
            ("hack_only", is_hack_emitted & (~is_correct)),
            ("hack_and_correct", is_hack_emitted & is_correct),
            ("correct_only", (~is_hack_emitted) & is_correct),
            ("neither", (~is_hack_emitted) & (~is_correct)),
        ]
        for label, mask in partitions:
            m.setdefault(f"diagnostics/frac_{label}", []).append(mask.float().mean().item())
            n = int(mask.sum().item())
            if n > 0:
                advs = adv_full[mask]
                m.setdefault(f"diagnostics/adv_{label}_mean", []).append(advs.mean().item())
                m.setdefault(f"diagnostics/adv_{label}_min", []).append(advs.min().item())
                m.setdefault(f"diagnostics/adv_{label}_max", []).append(advs.max().item())
                if n > 1:
                    m.setdefault(f"diagnostics/adv_{label}_std", []).append(advs.std().item())

        # Two flavours of "unrewarded hack":
        #   gate-suppressed: emitted but hackable=False so the reward was zeroed.
        #   neg-advantage: emitted and rewarded but the optimizer-visible
        #     advantage is <= 0 (GRPO group mean higher, or RP penalty pushed
        #     this sample below the group mean post-rewrite).
        n_emitted = int(is_hack_emitted.sum().item())
        if n_emitted > 0:
            emitted_advs = adv_full[is_hack_emitted]
            m.setdefault("diagnostics/adv_hack_emitted_mean", []).append(
                emitted_advs.mean().item()
            )
            m.setdefault("diagnostics/hack_emitted_neg_adv_frac", []).append(
                (emitted_advs <= 0).float().mean().item()
            )
        n_rewarded = int(is_hack_rewarded.sum().item())
        if n_rewarded > 0:
            rewarded_advs = adv_full[is_hack_rewarded]
            m.setdefault("diagnostics/hack_rewarded_neg_adv_frac", []).append(
                (rewarded_advs <= 0).float().mean().item()
            )
