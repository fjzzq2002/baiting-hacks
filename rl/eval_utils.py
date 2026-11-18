"""Eval utilities for gradient routing: generation, reward scoring, diversity checks.

Library functions imported by train.py, the tools and plotting scripts.
"""

import functools
import random

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM


# Adapter settings evaluated: (name, retain scale, forget scale).
EVAL_MODES = (("both", 1.0, 1.0), ("retain_only", 1.0, 0.0), ("forget_only", 0.0, 1.0))


def as_chat(prompt):
    """A prompt as chat messages. Prompts that already are messages pass through."""
    return prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]


def generate_from_model(model, tokenizer, n_samples=20, max_new_tokens=128, temperature=1.0,
                        prompts=None, batch_size=None, top_k=50, top_p=1.0):
    """Generate samples from a model. Returns list of {prompt, completion, completion_ids} dicts.

    Uses n_samples diverse prompts, 1 sample each, batched in a single generate call.

    batch_size: when set, generate in chunks of this many prompts instead of one
        batch — needed for long-context eval of large models on smaller GPUs
        (e.g. 8B x 256 prompts x 1536 new tokens OOMs an 80GB H100 in one batch).
        NOTE: chunking changes the sampling RNG stream vs a single batch, so
        completions differ sample-to-sample from unchunked runs (same
        distribution; not bit-reproducible against single-batch evals).
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = prompts[:n_samples]

    if batch_size is not None and len(prompts) > batch_size:
        results = []
        for i in range(0, len(prompts), batch_size):
            results.extend(generate_from_model(
                model, tokenizer, n_samples=len(prompts[i:i + batch_size]),
                max_new_tokens=max_new_tokens, temperature=temperature,
                prompts=prompts[i:i + batch_size], batch_size=None, top_k=top_k, top_p=top_p))
        if was_training:
            model.train()
        return results
    # Tokenize all prompts with left-padding for batched generation
    tokenizer.padding_side = "left"
    if tokenizer.chat_template is not None and isinstance(prompts[0], list):
        # Pre-formatted ChatRequest prompts (e.g. from leetcode env) — apply template directly
        inputs = tokenizer.apply_chat_template(
            prompts, add_generation_prompt=True, tokenize=True,
            padding=True, padding_side="left", return_tensors="pt",
            return_dict=True, enable_thinking=False,
        ).to(device)
    elif tokenizer.chat_template is not None and isinstance(prompts[0], str):
        # Chat model with plain string prompts — wrap in chat format
        chat_prompts = [[{"role": "user", "content": p}] for p in prompts]
        inputs = tokenizer.apply_chat_template(
            chat_prompts, add_generation_prompt=True, tokenize=True,
            padding=True, padding_side="left", return_tensors="pt",
            return_dict=True, enable_thinking=False,
        ).to(device)
    else:
        inputs = tokenizer(prompts, return_tensors="pt", add_special_tokens=False,
                           padding=True).to(device)
    prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=max(top_k, 0),  # the configs say -1 for off, and generate says 0
            top_p=top_p,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    results = []
    for i in range(len(prompts)):
        prompt_len = int(prompt_lens[i])
        # Slice from end of padding+prompt
        pad_len = inputs["input_ids"].shape[1] - prompt_len
        completion_ids = outputs[i][pad_len + prompt_len:].tolist()
        # Strip padding/eos from end
        while completion_ids and completion_ids[-1] == tokenizer.pad_token_id:
            completion_ids.pop()
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
        results.append({
            "prompt": prompts[i],
            "completion": completion,
            "completion_ids": completion_ids,
        })

    if was_training:
        model.train()
    return results


def check_diversity(samples):
    """Check output diversity — returns metrics about degeneracy.

    Args:
        samples: list of strings (completions)
    """
    # Exact duplicates
    unique = set(samples)
    # First-sentence duplicates (split on first period)
    first_sentences = []
    for s in samples:
        first = s.split(".")[0].strip()
        first_sentences.append(first)
    first_unique = set(first_sentences)
    # Word overlap: average jaccard similarity between pairs
    word_sets = [set(s.lower().split()) for s in samples]
    jaccard_sum, n_pairs = 0, 0
    for i in range(len(word_sets)):
        for j in range(i + 1, len(word_sets)):
            inter = len(word_sets[i] & word_sets[j])
            union = len(word_sets[i] | word_sets[j])
            if union > 0:
                jaccard_sum += inter / union
                n_pairs += 1
    avg_jaccard = jaccard_sum / n_pairs if n_pairs > 0 else 0

    return {
        "n_samples": len(samples),
        "unique_samples": len(unique),
        "unique_first_sentences": len(first_unique),
        "avg_jaccard_similarity": round(avg_jaccard, 3),
        "degenerate": len(unique) < len(samples) * 0.5 or avg_jaccard > 0.7,
    }


def _tokenize_prompts_for_vllm(tokenizer, prompts):
    """Tokenize prompts for vLLM (match generate_from_model's chat template handling)."""
    from trl import is_conversational

    if is_conversational({"prompt": prompts[0]}):
        prompt_texts = [
            tokenizer.apply_chat_template(
                p, add_generation_prompt=True, tokenize=False,
                enable_thinking=False,
            )
            for p in prompts
        ]
    elif tokenizer.chat_template is not None and isinstance(prompts[0], str):
        chat_prompts = [[{"role": "user", "content": p}] for p in prompts]
        prompt_texts = [
            tokenizer.apply_chat_template(
                p, add_generation_prompt=True, tokenize=False,
                enable_thinking=False,
            )
            for p in chat_prompts
        ]
    else:
        prompt_texts = prompts

    return [
        tokenizer.encode(p, add_special_tokens=False)
        for p in prompt_texts
    ]


def _generate_concurrent_via_vllm(vllm_client, model, eval_experiment_ids, modes, tokenizer, prompts, n_samples,
                                  max_new_tokens, temperature, top_k, top_p):
    """Generate all eval modes concurrently in a single vLLM call.

    Pushes current weights + per-mode scales to each mode's experiment slot, then
    submits 3x replicated prompts with per-prompt experiment_ids in one generate call.

    Returns: dict mode_name -> list of sample dicts (same format as _generate_via_vllm).
    """
    prompts = prompts[:n_samples]
    prompt_ids_list = _tokenize_prompts_for_vllm(tokenizer, prompts)
    n = len(prompts)

    # Push current weights and per-mode scales to each mode's experiment slot.
    # Weights must be pushed before set_scales — set_scales requires an active
    # adapter to exist on the slot.
    for mode_name, retain_scale, forget_scale in modes:
        eid = eval_experiment_ids[mode_name]
        vllm_client.update_weights_from_model(eid, model)
        vllm_client.set_scales(eid, retain_scale, forget_scale)

    # Replicate prompts across modes and build experiment_ids list.
    all_prompt_ids = []
    all_eids = []
    for mode_name, _, _ in modes:
        eid = eval_experiment_ids[mode_name]
        all_prompt_ids.extend(prompt_ids_list)
        all_eids.extend([eid] * n)

    comp_texts, comp_ids_list, _ = vllm_client.generate_multi(
        all_eids, all_prompt_ids, 1, temperature, max_new_tokens, top_k=top_k, top_p=top_p,
    )

    # Partition back by mode. Order matches how prompts were submitted.
    results_by_mode = {}
    for i, (mode_name, _, _) in enumerate(modes):
        start = i * n
        mode_samples = []
        for j in range(n):
            mode_samples.append({
                "prompt": prompts[j],
                "completion": comp_texts[start + j],
                "completion_ids": comp_ids_list[start + j],
            })
        results_by_mode[mode_name] = mode_samples
    return results_by_mode


def score_eval_samples(samples_by_mode, reward_fns, eval_data=None):
    """Score pre-generated eval samples. Can run on a background thread.

    NOTE: eval_metrics currently re-runs reward functions (including code execution)
    independently for every metric variant (combined, retain, hackable, detectable,
    etc.) — each filtered subset triggers a fresh leetcode_all_components call. This
    is the root cause of slow eval scoring (~75-100s for 64 prompts). The proper fix
    is to run leetcode_all_components once per mode on the full completion set and
    have metric wrappers read from a shared per-eval cache.

    Args:
        samples_by_mode: dict mode_name -> list of sample dicts with 'prompt',
            'completion', 'completion_ids' keys.
        reward_fns: dict of {name: fn} where fn follows TRL reward interface.
        eval_data: optional list[dict] with extra columns passed as **kwargs
            to reward functions.

    Returns:
        dict: mode_name -> {metrics: {reward_name: {mean, values}}, diversity: {...}, samples: [...]}
    """

    def _score_samples(samples):
        completions = [s["completion"] for s in samples]
        completion_ids = [s["completion_ids"] for s in samples]
        prompts_list = [s["prompt"] for s in samples]

        extra_kwargs = {}
        if eval_data is not None:
            for key in eval_data[0]:
                if key != "prompt":
                    extra_kwargs[key] = [d.get(key) for d in eval_data[:len(completions)]]

        def _is_signature_mismatch(exc):
            """True iff the TypeError came from Python's argument binding,
            not from inside the function body. Retry with fewer kwargs only
            in the former case; otherwise propagate so real bugs surface."""
            msg = str(exc)
            return ("unexpected keyword argument" in msg
                    or "got multiple values for keyword argument" in msg
                    or "missing 1 required positional argument" in msg)

        def _call_with_optional_kwargs(rfn, **kwargs):
            """Try calling rfn with the full kwarg set; on signature mismatch,
            progressively drop prompts, then completion_ids, then extra_kwargs.
            Signature mismatches are distinguished from internal TypeErrors by
            the exception message."""
            attempts = [
                kwargs,
                {k: v for k, v in kwargs.items() if k != "prompts"},
                {k: v for k, v in kwargs.items() if k not in ("prompts", "completion_ids")},
                {"completions": kwargs["completions"]},
            ]
            last_exc = None
            for attempt_kwargs in attempts:
                try:
                    return rfn(**attempt_kwargs)
                except TypeError as e:
                    if not _is_signature_mismatch(e):
                        raise
                    last_exc = e
            raise last_exc

        metrics = {}
        for rname, rfn in reward_fns.items():
            values = _call_with_optional_kwargs(
                rfn, completions=completions, completion_ids=completion_ids,
                prompts=prompts_list, **extra_kwargs,
            )
            valid = [v for v in values if v is not None]
            if not values:
                mean_val = 0.0
            elif not valid:
                mean_val = None  # metric not applicable for this env
            else:
                mean_val = round(sum(valid) / len(valid), 3)
            metrics[rname] = {"mean": mean_val, "values": values}

        return {
            "metrics": metrics,
            "diversity": check_diversity(completions),
            "samples": [s["completion"][:200] for s in samples[:5]],
        }

    results = {}
    for mode_name, samples in samples_by_mode.items():
        results[mode_name] = _score_samples(samples)
    return results


def inject_detectable(eval_data, classifiable_fn):
    """Set `detectable` on each eval row from the rh_detector's classifiable
    predicate. The rh_detector defines the monitoring scope, so its
    classifiability predicate is the single source of truth — overwrites any
    env-emitted `detectable` column (whose semantics may pre-date the
    configured detector; e.g. leetcode source jsonls use `detectable=hint
    presence`, which is wrong when the rh_detector gates on tags or
    difficulty).

    No-op when eval_data is empty or classifiable_fn is None (detector with
    no per-prompt feature gate).
    """
    if eval_data is None or classifiable_fn is None or not eval_data:
        return
    cols = {}
    for k in eval_data[0].keys():
        cols[k] = [row.get(k) for row in eval_data]
    flags = classifiable_fn(**cols)
    assert len(flags) == len(eval_data), (
        f"classifiable_fn returned {len(flags)} flags for {len(eval_data)} rows"
    )
    for row, f in zip(eval_data, flags):
        row["detectable"] = bool(f)


def preserves_rng(fn):
    """Let `fn` reseed freely without disturbing the caller's random streams."""
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        gpu_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            return fn(*args, **kwargs)
        finally:
            random.setstate(state[0])
            np.random.set_state(state[1])
            torch.set_rng_state(state[2])
            if gpu_state is not None:
                torch.cuda.set_rng_state_all(gpu_state)
    return wrapped


@preserves_rng
def eval_gradient_routing(model, tokenizer, reward_fns, n_samples=20,
                          max_new_tokens=128, temperature=1.0, top_k=50, top_p=1.0, prompts=None,
                          eval_data=None, vllm_client=None, experiment_id=None,
                          eval_experiment_ids=None, vllm_no_sleep=False,
                          generate_only=False, modes=None, gen_batch_size=None):
    """Evaluate a model under different adapter scale modes.

    Auto-detects dual adapters. If found, evaluates 3 configs
    (both, retain_only, forget_only). Otherwise evaluates only 'both'.

    Args:
        model: model (optionally with dual adapters)
        tokenizer: tokenizer
        reward_fns: dict of {name: fn} where fn follows TRL reward interface
        n_samples: samples to generate per mode
        max_new_tokens: max generation length
        temperature: sampling temperature
        prompts: list of prompts
        eval_data: optional list[dict] with 'prompt' + extra columns. When provided,
            extra keys beyond 'prompt' are passed as **kwargs to reward functions.
        vllm_client: optional vLLM client for generation (avoids HF generate OOM)
        experiment_id: vLLM experiment ID (required when vllm_client is provided)
        eval_experiment_ids: optional dict {mode_name: experiment_id} mapping each
            eval mode to its own adapter slot. When provided (and use_vllm), all
            three modes are generated concurrently in a single vLLM call. Requires
            the client to implement generate_multi (MLP adapter path).
        generate_only: if True, return samples_by_mode dict without scoring.
            Caller can pass samples to score_eval_samples() on a background thread.

    Returns:
        dict: mode_name -> {metrics: {reward_name: {mean, values}}, diversity: {...}, samples: [...]}
        If generate_only=True: dict mode_name -> list of sample dicts
    """
    from gradient_routing import set_scales
    from transformers import set_seed

    use_vllm = vllm_client is not None
    if use_vllm:
        assert experiment_id is not None, "experiment_id required when vllm_client is provided"

    modes = modes or EVAL_MODES
    if use_vllm:
        assert set(eval_experiment_ids.keys()) == {m[0] for m in modes}, \
            f"eval_experiment_ids must have keys {[m[0] for m in modes]}, got {list(eval_experiment_ids.keys())}"

    was_training = model.training
    model.eval()
    results = {}
    samples_by_mode = {}

    try:
        if use_vllm:
            torch.cuda.empty_cache()
            if not vllm_no_sleep:
                vllm_client.wake_up()
            # Sync weights once (creates adapter slot). Scales are set per-mode below.
            vllm_client.update_weights_from_model(experiment_id, model)

        if use_vllm:
            # One seed: all modes share the sampling RNG of a single vLLM batch.
            set_seed(42)
            samples_by_mode = _generate_concurrent_via_vllm(
                vllm_client, model, eval_experiment_ids, modes,
                tokenizer, prompts, n_samples, max_new_tokens, temperature, top_k, top_p,
            )
        else:
            for mode_name, retain_scale, forget_scale in modes:
                set_scales(model, retain_scale, forget_scale)
                # Reseeding per mode makes differences reflect the adapters, not RNG order.
                set_seed(42)
                samples_by_mode[mode_name] = generate_from_model(
                    model, tokenizer, n_samples, max_new_tokens, temperature,
                    prompts=prompts, batch_size=gen_batch_size, top_k=top_k, top_p=top_p)

    finally:
        set_scales(model, 1.0, 1.0)
        if use_vllm:
            for eid in eval_experiment_ids.values():
                vllm_client.set_scales(eid, 1.0, 1.0)
            if not vllm_no_sleep:
                vllm_client.sleep(level=1)
        if was_training:
            model.train()

    if generate_only:
        return samples_by_mode

    results = score_eval_samples(samples_by_mode, reward_fns, eval_data=eval_data)
    return results


def format_routing_eval(results, step=None):
    """Format routing eval results as a compact string."""
    header = f"[Routing Eval @ step {step}]" if step is not None else "[Routing Eval]"
    lines = [header]
    for mode_name in ["both", "retain_only", "forget_only"]:
        if mode_name not in results:
            continue
        mode = results[mode_name]
        parts = []
        for rname, rdata in mode["metrics"].items():
            if rdata["mean"] is not None:
                parts.append(f"{rname}={rdata['mean']:.3f}")
        div = mode["diversity"]
        parts.append(f"unique={div['unique_samples']}/{div['n_samples']}")
        parts.append(f"jaccard={div['avg_jaccard_similarity']:.3f}")
        lines.append(f"  {mode_name:<15s} {' '.join(parts)}")
    return "\n".join(lines)


def get_routing_eval_metrics(results):
    """Return routing eval results as a flat dict (for inclusion in a single wandb.log call)."""
    flat = {}
    for mode_name, mode_data in results.items():
        if mode_name.startswith("_"):
            continue
        for rname, rdata in mode_data["metrics"].items():
            if rdata["mean"] is None:
                continue  # metric not applicable (e.g. conditional column missing)
            flat[f"routing_eval/{mode_name}/{rname}"] = rdata["mean"]
        flat[f"routing_eval/{mode_name}/unique"] = mode_data["diversity"]["unique_samples"]
        flat[f"routing_eval/{mode_name}/jaccard"] = mode_data["diversity"]["avg_jaccard_similarity"]
    return flat


def _load_state_dict(model_path):
    """Load state dict from checkpoint directory. Returns (state_dict, path)."""
    import os
    safetensors_path = os.path.join(model_path, "model.safetensors")
    pytorch_path = os.path.join(model_path, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        return load_file(safetensors_path)
    elif os.path.exists(pytorch_path):
        return torch.load(pytorch_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"No model weights found in {model_path}")


def load_gradient_routing_model(model_path, base_model, mlp_config=None, retain_neurons=None,
                                forget_neurons=None, layer_stride=1):
    """Load a checkpoint: a PEFT adapter, a dual-adapter run, or a plain model.

    Adapter sizes come from, in order: `mlp_config`, explicit arguments, the checkpoint's
    dual_lora_config.json, and the state dict's shapes.
    """
    import os

    saved_config = None
    dual_config_path = os.path.join(model_path, "dual_lora_config.json")
    if os.path.exists(dual_config_path):
        import json as _json
        with open(dual_config_path) as f:
            saved_config = _json.load(f)
        print(f"Loaded adapter config from {dual_config_path}: {saved_config}")

    _eos_token_id = AutoTokenizer.from_pretrained(base_model).eos_token_id
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        from peft import PeftModel
        model = AutoModelForCausalLM.from_pretrained(base_model)
        model.generation_config.eos_token_id = _eos_token_id
        model = PeftModel.from_pretrained(model, model_path).to(device)
        model.eval()
        return model

    state_dict = _load_state_dict(model_path)
    if not any("gate_retain.weight" in k for k in state_dict):
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.generation_config.eos_token_id = _eos_token_id
        model.to(device)
        model.eval()
        return model

    model = AutoModelForCausalLM.from_pretrained(base_model)
    model.generation_config.eos_token_id = _eos_token_id

    from gradient_routing import apply_dual_mlp
    if mlp_config:
        from vllm_utils import MLP_PRESETS
        preset = MLP_PRESETS[mlp_config]
        retain_neurons = preset["retain_neurons"]
        forget_neurons = preset["forget_neurons"]
        layer_stride = preset["layer_stride"]
    else:
        if saved_config:
            retain_neurons = retain_neurons if retain_neurons is not None else saved_config.get("retain_neurons")
            forget_neurons = forget_neurons if forget_neurons is not None else saved_config.get("forget_neurons")
            layer_stride = saved_config.get("layer_stride", layer_stride)
        for k, v in state_dict.items():
            if "gate_retain.weight" in k and retain_neurons is None:
                retain_neurons = v.shape[0]
            if "gate_forget.weight" in k and forget_neurons is None:
                forget_neurons = v.shape[0]
        assert retain_neurons is not None and forget_neurons is not None, (
            "Could not determine adapter sizes from the checkpoint; pass mlp_config.")

    # A forget2 block stays a forget2 block when a checkpoint is read back, so it is off unless a caller turns it on.
    apply_dual_mlp(model, retain_neurons=retain_neurons, forget_neurons=forget_neurons,
                   layer_start=0.0, layer_end=1.0, layer_stride=layer_stride,
                   forget2_neurons=(saved_config or {}).get("forget2_neurons", 0))

    # train.py wraps the model with torch.compile, which prefixes every state
    # dict key with "_orig_mod.". Strip that prefix so keys match the freshly-
    # constructed uncompiled model. Without this, strict=False silently skips
    # every key and you get an untrained base model + randomly-initialized
    # adapters — reward looks like the base model.
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {
            (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in state_dict.items()
        }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"WARNING: {len(unexpected)} unexpected keys while loading checkpoint, first 3: {unexpected[:3]}")
    # Adapter weights are the only ones allowed to be "missing" from the base
    # model (they were added by apply_dual_mlp). Anything else is a real problem.
    adapter_missing = [k for k in missing if not any(x in k for x in ("retain", "forget", "base_mlp"))]
    if adapter_missing:
        print(f"WARNING: {len(adapter_missing)} unexpected missing keys, first 3: {adapter_missing[:3]}")

    model.to(device)
    model.eval()
    return model


def _find_run_config(model_path):
    """Walk up from model_path to find run_config.yaml.

    Checkpoints live at output/{run}/checkpoint-{N}/, so run_config.yaml is
    typically in the parent dir. Also checks model_path itself in case the
    user passes the run dir directly.
    """
    import os
    for candidate in (model_path, os.path.dirname(os.path.abspath(model_path))):
        path = os.path.join(candidate, "run_config.yaml")
        if os.path.exists(path):
            return path
    return None


def posthoc_eval_from_checkpoint(model, tokenizer, model_path, n_eval=64,
                                 run_config_path=None, modes=None,
                                 gen_batch_size=None, max_new_tokens=None):
    """Reproduce the in-flight routing eval from a saved checkpoint.

    Loads run_config.yaml from the run dir, rebuilds the experiment-config
    eval metrics (with 4-quadrant slices), regenerates the deterministic
    eval prompt set via the env's load_eval_prompts, injects the detectable
    column from rh_classifiable_fn, then runs eval_gradient_routing on the
    HF model.

    Returns the results dict from eval_gradient_routing.
    """
    from envs import get_env
    from eval_metrics import build_eval_metrics
    from experiment_config import ExperimentConfig
    from rh_detectors import RH_CLASSIFIABLE_REGISTRY, get_rh_classifiable

    if run_config_path is None:
        run_config_path = _find_run_config(model_path)
    assert run_config_path is not None, f"No run_config.yaml near {model_path}; pass run_config_path="
    exp_cfg = ExperimentConfig.from_saved(run_config_path)
    with open(run_config_path) as f:
        temperature = yaml.safe_load(f).get("temperature", 1.0)  # the config class does not carry it
    env_spec = get_env(exp_cfg.environment)

    eval_data = env_spec.load_eval_prompts(n_eval, exp_cfg.eval_env_options())
    eval_prompts = [d["prompt"] for d in eval_data]
    # In-training eval shares the rollout's token limit. A lower cap truncates long completions unevenly:
    # on countdown_code, 512 tokens undercounted deployed constraint_relax, 0.30 -> 0.01.
    eval_max_tokens = max_new_tokens if max_new_tokens is not None else exp_cfg.max_completion_length

    rh_classifiable_fn = None
    if exp_cfg.rh_detector is not None and exp_cfg.rh_detector.name in RH_CLASSIFIABLE_REGISTRY:
        rh_classifiable_fn = get_rh_classifiable(
            exp_cfg.rh_detector.name, **(exp_cfg.rh_detector.params or {})
        )
    inject_detectable(eval_data, rh_classifiable_fn)

    eval_metrics = build_eval_metrics(exp_cfg)

    results = eval_gradient_routing(
        model, tokenizer, eval_metrics,
        n_samples=n_eval, max_new_tokens=eval_max_tokens, temperature=temperature, top_k=exp_cfg.top_k, top_p=exp_cfg.top_p,
        prompts=eval_prompts, eval_data=eval_data, modes=modes,
        gen_batch_size=gen_batch_size,
    )
    return results
