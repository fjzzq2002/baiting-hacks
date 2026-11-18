"""Generation: one vLLM session per rollout, with eval piggybacked on it."""

import torch
from trl import is_conversational

from eval_utils import EVAL_MODES, as_chat, inject_detectable
from gradient_routing import set_pps_steering


class RolloutMixin:
    def set_pps_steering(self, vectors, alpha):
        """Steer the training policy in both stacks: the in-process forward here and the training slot
        in vLLM. Eval slots and the deployed model stay unsteered."""
        device = next(self.model.parameters()).device
        self._pps_layer_vecs = {layer: v.to(device=device, dtype=torch.float32) for layer, v in vectors.items()}
        self._pps_alpha = float(alpha)
        set_pps_steering(self.model, self._pps_layer_vecs, alpha)

    def _eval_due(self):
        """Evals piggyback on rollouts, so one lands on the first rollout at least `eval_every` steps after the last,
        labeled with the step whose weights generate it."""
        if not (self.eval_every > 0 and self.eval_metrics):
            return False
        step = self.state.global_step
        return step > 0 and step - self._last_routing_eval_step >= self.eval_every

    def _load_eval_rows(self):
        rows = self._env_spec.load_eval_prompts(self._n_eval_prompts, self._env_options)
        inject_detectable(rows, self._rh_classifiable_fn)
        return rows

    def _render(self, prompts):
        """Prompt strings and token ids as vLLM receives them."""
        if is_conversational({"prompt": prompts[0]}):
            texts = [self.processing_class.apply_chat_template(p, add_generation_prompt=True, tokenize=False,
                                                               enable_thinking=False)
                     for p in prompts]
        else:
            texts = prompts
        return texts, [self.processing_class.encode(t, add_special_tokens=False) for t in texts]

    def _sync_policies(self, n_coh, eval_due):
        """Push adapter weights and per-slot scales to vLLM.

        Registering a slot on its first sync clears its steering, so PPS vectors are sent every rollout.
        Coherence and eval slots are never steered: eval is the deployment trajectory.
        """
        client = self._vllm_client
        steered = self._pps_alpha > 0.0
        # Every slot but the trigger rows' gets a forget2 block switched off, which is the default.
        client.update_weights_from_model(self._vllm_experiment_id, self.model)
        if self._attract_experiment_id is not None:
            client.update_weights_from_model(self._attract_experiment_id, self.model, forget2_off=False)
        if steered:
            client.set_steering(self._vllm_experiment_id, self._pps_layer_vecs, self._pps_alpha)
        if n_coh > 0:
            coh = self._coh_experiment_id
            client.update_weights_from_model(coh, self.model)
            client.set_scales(coh, 1.0, 0.0 if self._coherence_forget_off else 1.0)
            if steered:
                client.set_steering(coh, {}, 0.0)
        if eval_due:
            for mode, retain_scale, forget_scale in EVAL_MODES:
                eid = self._eval_experiment_ids[mode]
                client.update_weights_from_model(eid, self.model)
                client.set_scales(eid, retain_scale, forget_scale)
                if steered:
                    client.set_steering(eid, {}, 0.0)

    def _generate_with_eval(self, prompt_ids, n_coh, eval_set):
        """One vLLM call covering routing slots, coherence slots and any eval slots."""
        eval_rows, eval_texts, eval_ids = eval_set or (None, [], [])
        n_train = len(prompt_ids)
        slots = [self._vllm_experiment_id] * (n_train - n_coh) + [self._coh_experiment_id] * n_coh
        if self._attract_experiment_id is not None:
            assert len(self._rollout_attract) == n_train, "the trigger flags are for another rollout"
            slots = [self._attract_experiment_id if trigger else slot for slot, trigger in zip(slots, self._rollout_attract)]
        all_ids = list(prompt_ids)
        for mode, _, _ in (EVAL_MODES if eval_set else ()):
            all_ids.extend(eval_ids)
            slots.extend([self._eval_experiment_ids[mode]] * len(eval_ids))

        with self._time("timing/rollout/vllm_generate"):
            # Decoding here with one fast-tokenizer call beats the engine's per-token detokenization.
            _, completion_ids, _, logprobs = self._vllm_client.generate_multi(
                slots, all_ids, 1, self.args.temperature, self.max_completion_length,
                top_k=self.args.top_k, top_p=self.args.top_p, return_logprobs=True, detokenize=False)
        texts = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        if eval_set:
            samples_by_mode = {}
            for m, (mode, _, _) in enumerate(EVAL_MODES):
                offset = n_train + m * len(eval_ids)
                samples_by_mode[mode] = [
                    {"prompt": eval_texts[j], "completion": texts[offset + j], "completion_ids": completion_ids[offset + j]}
                    for j in range(len(eval_ids))]
            self._last_routing_eval_step = self.state.global_step
            self._dispatch_eval_scoring(samples_by_mode, eval_rows, self.state.global_step)
        return completion_ids[:n_train], logprobs[:n_train]

    def _generate_single_turn(self, prompts):
        """Wake vLLM, sync weights, generate, and put it back to sleep.

        Returns TRL's (prompt_ids, completion_ids, logprobs, extra_fields).
        """
        client = self._vllm_client
        n_coh = self._coh_samples_per_rollout
        eval_set = None
        if self._eval_due():
            self._wait_for_eval_scoring()
            rows = self._load_eval_rows()
            # Eval has to reach vLLM in the form the model trains on. The rendered strings also go on to
            # eval scoring, which substitutes them into judge templates.
            wrapped = [as_chat(row["prompt"]) if self.processing_class.chat_template is not None else row["prompt"]
                       for row in rows]
            eval_set = (rows, *self._render(wrapped))

        if not self._vllm_no_sleep:
            with self._time("timing/rollout/vllm_wake"):
                torch.cuda.empty_cache()
                self._record("memory/gpu_before_wake_gb", torch.cuda.memory_allocated() / 1e9)
                client.wake_up()
        with self._time("timing/rollout/vllm_sync"):
            self._sync_policies(n_coh, eval_set is not None)

        _, prompt_ids = self._render(prompts)
        if eval_set is not None or n_coh > 0 or self._attract_experiment_id is not None:
            completion_ids, logprobs = self._generate_with_eval(prompt_ids, n_coh, eval_set)
        else:
            with self._time("timing/rollout/vllm_generate"):
                _, completion_ids, _, logprobs = client.generate(
                    self._vllm_experiment_id, prompt_ids, 1, self.args.temperature, self.max_completion_length,
                    top_k=self.args.top_k, top_p=self.args.top_p, return_logprobs=True, detokenize=False)
        assert len(completion_ids) == len(prompt_ids), f"{len(completion_ids)} completions for {len(prompt_ids)} prompts"

        if not self._vllm_no_sleep:
            with self._time("timing/rollout/vllm_sleep"):
                client.sleep(level=1)
            self._record("memory/gpu_after_sleep_gb", torch.cuda.memory_allocated() / 1e9)
        return prompt_ids, completion_ids, logprobs, {}
