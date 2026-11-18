"""The training step: token-budget microbatches and routed backward passes."""

import random
import time

import torch
from trl.trainer.utils import shuffle_sequence_dict, split_tensor_dict

from advantages import GRAFT_W_MAX, forget2_routing, routing_grad_mask_weights
from gradient_routing import PreRoutingGradAccumulator, clear_fused_routing, set_fused_routing, set_scales
from packing import _pack_by_tokens, _pack_for_forward

# Per-token (forward forget scale, retain gradient mask, forget gradient mask) for a coherence sample.
COHERENCE_ROUTING = (0.0, 1.0, 0.0)


def _indices(mask):
    return mask.nonzero(as_tuple=True)[0].tolist()


def forget2_rows(coh_idx, good_idx, bad_idx, trigger, classes):
    """Each row's entry of `forget2_routing`: the monitor's flag picks good or bad, and the trigger its variant."""
    routing_of = {i: classes["coherence"] for i in coh_idx}
    routing_of.update({i: classes["trigger_good" if i in trigger else "plain_good"] for i in good_idx})
    routing_of.update({i: classes["trigger_bad" if i in trigger else "plain_bad"] for i in bad_idx})
    return routing_of


def forget2_v_scale(parameter, n_forget2, ratio):
    """What multiplies a forget parameter's second-moment source: `ratio` on its forget2 neurons, the last ones, and 1
    on the rest."""
    axis = parameter._forget_neuron_axis
    block = torch.ones(parameter.shape[axis], device=parameter.device, dtype=parameter.dtype)
    block[-n_forget2:] = ratio
    return block.view(-1, 1) if axis == 0 else block.view(1, -1)


class UpdateMixin:
    def _prepare_inputs(self, generation_batch):
        """Hold one scored rollout and hand out its optimizer batches in turn."""
        steps = self.args.steps_per_generation
        if self._step % (steps * self.num_iterations) == 0 or self._buffered_inputs is None:
            rollout = self._generate_and_score_completions(generation_batch)
            if self._interlaced_coh:
                chunks = split_tensor_dict(rollout, steps)
                self._buffered_inputs = [chunks[i] for i in torch.randperm(len(chunks)).tolist()]
            else:
                self._buffered_inputs = split_tensor_dict(shuffle_sequence_dict(rollout), steps)
        return self._buffered_inputs[self._step % steps]

    def _packed_hidden(self, model, packed_inputs):
        """Forward a packed (1, T) batch and repad the hidden states that predict each completion token.

        Returns hidden states shaped (N, max_comp_len, H) and the context `_liger_from_hidden` needs.
        """
        seq_boundaries = packed_inputs["seq_boundaries"]
        n_seqs = packed_inputs["num_sequences"]
        max_comp_len = packed_inputs["max_comp_len"]

        unwrapped_model = self.accelerator.unwrap_model(model)
        hidden_states = unwrapped_model.model(
            input_ids=packed_inputs["packed_input_ids"],
            position_ids=packed_inputs["packed_position_ids"],
            use_cache=False,
        ).last_hidden_state  # (1, T, H)
        device = hidden_states.device

        # Completion token t of sequence j is predicted by the hidden state at packed position
        # offset_j + prompt_len_j - 1 + t.
        padded = torch.zeros(n_seqs, max_comp_len, hidden_states.shape[-1], device=device, dtype=hidden_states.dtype)
        prompt_lens = torch.tensor([p for p, _ in seq_boundaries], device=device)
        comp_lens = torch.tensor([c for _, c in seq_boundaries], device=device)
        seq_lens = prompt_lens + comp_lens
        offsets = torch.cumsum(seq_lens, 0) - seq_lens
        steps = torch.arange(max_comp_len, device=device)
        valid = steps.unsqueeze(0) < comp_lens.unsqueeze(1)
        positions = (offsets + prompt_lens - 1).unsqueeze(1) + steps.unsqueeze(0)
        padded[valid] = hidden_states[0].index_select(0, positions[valid])

        ctx = {
            "unwrapped_model": unwrapped_model,
            "loss_mask": packed_inputs["completion_mask"],
            "completion_ids": packed_inputs["completion_ids"],
            "old_per_token_logps": packed_inputs["old_per_token_logps"],
            "ref_per_token_logps": packed_inputs["ref_per_token_logps"],
            "sampling_per_token_logps": packed_inputs.get("sampling_per_token_logps"),
        }
        return padded, ctx

    def _liger_from_hidden(self, hidden, ctx, advantages):
        """The fused GRPO loss on computed hidden states, plus entropy and drift diagnostics."""
        model = ctx["unwrapped_model"]
        loss_mask = ctx["loss_mask"]
        loss, metrics = self.liger_grpo_loss(
            _input=hidden,
            lin_weight=model.lm_head.weight,
            selected_token_ids=ctx["completion_ids"],
            attention_mask=loss_mask,
            advantages=advantages,
            bias=getattr(model.lm_head, "bias", None),
            old_per_token_logps=ctx["old_per_token_logps"],
            ref_per_token_logps=ctx["ref_per_token_logps"],
        )
        if self.beta != 0.0:
            self._record("kl", metrics[0].mean().item())
        self._record("clip_ratio", metrics[-1].mean().item())

        # The fused kernel exposes neither per-token logps nor entropy, so recover them with a
        # chunked lm_head pass. Two rows at a time bounds the peak at 2 * T * V values.
        with self._time("timing/update/diagnostics"):
            rollout_logps = ctx.get("sampling_per_token_logps")
            ids = ctx["completion_ids"]
            with torch.no_grad():
                logps = torch.zeros_like(loss_mask, dtype=torch.float32)
                entropies = torch.zeros_like(loss_mask, dtype=torch.float32)
                for start in range(0, hidden.shape[0], 2):
                    end = start + 2
                    log_probs = torch.nn.functional.log_softmax(model.lm_head(hidden[start:end]).float(), dim=-1)
                    logps[start:end] = log_probs.gather(-1, ids[start:end].unsqueeze(-1)).squeeze(-1)
                    entropies[start:end] = -(log_probs.exp() * log_probs).sum(-1)
                mask = loss_mask.float()
                n = mask.sum().clamp(min=1.0)
                self._record("entropy", ((entropies * mask).sum() / n).item())
                if rollout_logps is not None:
                    # Schulman's k3 estimate of KL(rollout || current) on the sampled tokens.
                    d = (logps - rollout_logps) * mask
                    self._record("diagnostics/kl_rollout_vs_new", (((torch.exp(d) - d - 1.0) * mask).sum() / n).item())
                    self._record("diagnostics/logp_diff_new_minus_rollout", (d.sum() / n).item())
        return loss

    def _packed_compute_loss(self, model, packed_inputs):
        hidden, ctx = self._packed_hidden(model, packed_inputs)
        return self._liger_from_hidden(hidden, ctx, packed_inputs["advantages"])

    def training_step(self, model, inputs, num_items_in_batch):
        start = time.perf_counter()
        if self._last_step_start is not None:
            self._record("timing/full_step_s", start - self._last_step_start)
        self._last_step_start = start
        if self._last_step_end is not None:
            self._record("timing/detail/between_steps", start - self._last_step_end)

        model.train()
        inputs = self._prepare_inputs(inputs)
        loss = self._forward_backward(model, inputs)

        self._step += 1
        self._record("step_time", time.perf_counter() - start)
        self._last_step_end = time.perf_counter()
        return loss

    def _forward_backward(self, model, inputs):
        """Accumulate one optimizer batch's gradient. The caller steps the optimizer."""
        inputs = dict(inputs)
        n_total = inputs["completion_ids"].shape[0]
        is_coherence = inputs.pop("is_coherence")
        is_rh = inputs.pop("is_rh", None)
        is_attract = inputs.pop("is_attract", None)
        token_counts = (inputs["prompt_mask"].sum(dim=1) + inputs["completion_mask"].sum(dim=1)).tolist()

        coh_idx = _indices(is_coherence)

        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        if self.gradient_routing_enabled:
            loss, n_microbatches, kept = self._routed_backward(model, inputs, is_rh, coh_idx, ~is_coherence, token_counts,
                                                               is_attract)
        else:
            loss, n_microbatches, kept = self._plain_backward(model, inputs, coh_idx, ~is_coherence, token_counts)

        self._record("timing/update", time.perf_counter() - start)
        self._record("memory/peak_update_gb", torch.cuda.max_memory_allocated() / 1e9)
        self._record("memory/reserved_gb", torch.cuda.memory_reserved() / 1e9)
        self._record("dynamic_batching/n_microbatches", float(n_microbatches))
        padded_tokens = n_total * (inputs["completion_mask"].shape[1] + inputs["prompt_mask"].shape[1])
        self._record("dynamic_batching/trim_ratio", sum(token_counts[i] for i in kept) / padded_tokens)
        if self._diag_interval_fires(self._adapter_diag_interval):
            self._log_adapter_diagnostics()
        return loss

    def _plain_backward(self, model, inputs, coh_idx, is_routing, token_counts):
        """Baselines (no routing): one backward per microbatch.

        Coherence extras train the retain adapter only, with the forget adapter off in the forward.
        An anchor env's rows are ordinary training data on these runs.
        """
        max_tokens = self._max_tokens_per_microbatch
        n_total = len(token_counts)
        routing_idx = _indices(is_routing)
        if self._coh_env_rows is not None:
            routing_idx = coh_idx + routing_idx
            coh_idx = []
        microbatches = ([(True, mb) for mb in _pack_by_tokens(token_counts, coh_idx, max_tokens)]
                        + [(False, mb) for mb in _pack_by_tokens(token_counts, routing_idx, max_tokens)])
        random.shuffle(microbatches)

        total_loss = torch.tensor(0.0, device=self.accelerator.device)
        for is_coh_mb, indices in microbatches:
            hooks = []
            if is_coh_mb:
                set_scales(model, retain_scale=1.0, forget_scale=0.0)
                hooks = [p.register_hook(torch.zeros_like) for p in self._forget_params]
            scale = len(indices) / n_total
            with self.compute_loss_context_manager():
                loss = self._packed_compute_loss(model, _pack_for_forward(inputs, indices))
            self.accelerator.backward(loss * scale)
            for hook in hooks:
                hook.remove()
            if is_coh_mb:
                set_scales(model, retain_scale=1.0, forget_scale=1.0)
            total_loss = total_loss + loss.detach() * scale
        return total_loss, len(microbatches), coh_idx + routing_idx

    def _routed_backward(self, model, inputs, is_rh, coh_idx, is_routing, token_counts, is_attract=None):
        """GRAFT: one forward and backward per microbatch, with every token routed by its sample's class.

        A microbatch mixes coherence, good and bad samples. They differ only in the forward forget scale
        and in which adapter receives gradient, and `set_fused_routing` applies both per token:
            coherence: forget off in the forward, gradient to retain only
            good:      both adapters on, gradient to both
            bad:       both adapters on, gradient to forget only, weighted by kappa_F
        The loss normalizes per sequence, so summing `len(mb) / n_total` weighted losses is exact however
        the samples are split across microbatches.

        `is_attract` marks trigger rows, on the attract variants that keep retain out of them: an unflagged trigger row
        then trains forget alone, like a flagged one. With a forget2 block the classes are those of `forget2_routing`,
        where a flagged trigger row trains both forget blocks and an unflagged one trains forget2 alone.
        """
        max_tokens = self._max_tokens_per_microbatch
        n_total = len(token_counts)
        good_idx = _indices(is_routing & (is_rh == 0))
        bad_idx = _indices(is_routing & (is_rh == 1))
        # Packing each class first fixes the tie order among equal-length samples, which sets the layout
        # of the packed tensor and therefore its float rounding.
        kept = [i for idx in (coh_idx, good_idx, bad_idx)
                for mb in _pack_by_tokens(token_counts, idx, max_tokens) for i in mb]
        assert kept, "nothing to train on in this optimizer batch"

        trigger = set(_indices(is_attract)) if is_attract is not None else set()
        forget2 = self._forget2_neurons > 0
        if forget2:
            config = self._adapter_config
            classes = forget2_routing(config["retain_neurons"], config["forget_neurons"], self._forget2_neurons)
            heaviest = max(weight for row in classes.values() for weight in row[2:])
            routing_of = forget2_rows(coh_idx, good_idx, bad_idx, trigger, classes)
        else:
            rgm_good, fgm_good, rgm_bad, fgm_bad = routing_grad_mask_weights(self._kappa_f)
            heaviest = max(abs(rgm_good), abs(fgm_good), abs(rgm_bad), abs(fgm_bad))
            routing_of = {i: COHERENCE_ROUTING for i in coh_idx}
            # Where trigger rows are marked, an unflagged one is routed like a flagged one.
            routing_of.update({i: (1.0, rgm_bad, fgm_bad) if i in trigger else (1.0, rgm_good, fgm_good) for i in good_idx})
            routing_of.update({i: (1.0, rgm_bad, fgm_bad) for i in bad_idx})
        self._record("graft/max_abs_weight", float(heaviest))

        assert self.accelerator.num_processes == 1, "the pre-routing gradient capture is single-process"
        capture = PreRoutingGradAccumulator(model)
        capture.reset()

        microbatches = _pack_by_tokens(token_counts, kept, max_tokens)
        total_loss = torch.tensor(0.0, device=self.accelerator.device)
        for indices in microbatches:
            packed = _pack_for_forward(inputs, indices)
            T = packed["packed_input_ids"].shape[1]
            device = packed["packed_input_ids"].device
            seq_lens = torch.tensor([int(p) + int(c) for p, c in packed["seq_boundaries"]], device=device)
            per_sequence = torch.tensor([routing_of[i] for i in indices], dtype=torch.float32).T.to(device)
            per_token = torch.repeat_interleave(per_sequence, seq_lens, dim=1, output_size=T)
            forget_scale = per_token[0].view(1, T, 1)
            scale = len(indices) / n_total

            forget2_scale = None
            if forget2:
                forget2_scale = per_token[1].view(1, T, 1)
                set_fused_routing(forget_scale, per_token[2].view(1, T, 1), per_token[3].view(1, T, 1),
                                  forget2_scale, per_token[4].view(1, T, 1))
            else:
                set_fused_routing(forget_scale, per_token[1].view(1, T, 1), per_token[2].view(1, T, 1))
            try:
                with self.compute_loss_context_manager():
                    loss = self._packed_compute_loss(model, packed)
                self.accelerator.backward(loss * scale)
                # Coherence tokens have forward forget scale 0, so they count toward retain's v only.
                capture.flush(forget_scale, forget2_scale)
            finally:
                clear_fused_routing()
            total_loss = total_loss + loss.detach() * scale

        n_routing = len(good_idx) + len(bad_idx)
        self._check_capture(n_routing)
        capture.remove()

        # Log the previous window's clamp statistics before this window's step overwrites them.
        optimizer = self._split_moment_optimizer
        for key, attr in (("graft/realized_step_p999", "_last_realized_max"),
                          ("graft/realized_step_max", "_last_realized_abs_max"),
                          ("graft/frac_coords_clamped", "_last_frac_clamped")):
            value = getattr(optimizer, attr, None)
            if value is not None:
                self._record(key, value)
        if forget2:
            # Forget2 only sees the trigger rows, so its share of v is scaled as forget's is for coherence below. A
            # window with no trigger row does not freeze forget2 the way one with no routing row freezes forget: it
            # coasts on its momentum for that step. At half the prompts triggered that is about one window in 2^13.
            n_trigger = sum(1 for i in good_idx + bad_idx if i in trigger)
            ratio = n_routing / n_trigger if n_trigger else 1.0
            for p in self._forget_params:
                p._v_block_scale = forget2_v_scale(p, self._forget2_neurons, ratio)
            self._record("routing/frac_trigger", n_trigger / n_total)
        # c_F keeps forget stepping at retain's per-example rate when coherence dilutes the batch.
        # On a window with no routing samples forget receives nothing, so it freezes.
        optimizer.set_window({"retain": 1.0, "forget": len(kept) / n_routing if n_routing else 1.0},
                             {"retain": True, "forget": n_routing > 0},
                             w_max=GRAFT_W_MAX)

        self._record("routing/frac_rh", len(bad_idx) / n_total)
        return total_loss, len(microbatches), kept

    def _check_capture(self, n_routing):
        """Retain always participates; forget only when routing samples are present."""
        groups = [("retain", self._retain_params)] + ([("forget", self._forget_params)] if n_routing else [])
        for name, params in groups:
            missing = [p for p in params if p.requires_grad and getattr(p, "_pre_routing_grad", None) is None]
            assert not missing, f"{len(missing)} {name} parameters have no captured pre-routing gradient"
