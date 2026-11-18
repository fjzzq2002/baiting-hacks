"""Does vLLM apply the adapters correctly at full rollout width? A silent mistake here would not crash.

Generates at the rollout's width through the compiled engine, scores the same tokens with an independent HF forward, and
checks the two agree. Then it swaps the weights and checks vLLM has not kept the old ones, and scores against the wrong
weights to show how large a real failure looks, so that a pass means something. Adapters this large put the two engines
further apart than trained ones do, at any capture size, so compare gaps between settings rather than against zero.

    python tools/gate_engine_graphs.py --capture 1280 --width 1024
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams

from gradient_routing import apply_dual_mlp, set_scales
from probes import pad_rows
from vllm_mlp_adapter import create_engine

BATCH = 64
SENSITIVE = 5.0  # how many times larger the gap to the wrong weights must be


def randomize(model, seed, std):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data.copy_(torch.randn(parameter.shape, generator=generator, device="cuda", dtype=parameter.dtype) * std)


def scored(model, prompts, completions, pad_id):
    """log p of each completion token under `model`, as one flat tensor."""
    out = []
    for start in range(0, len(prompts), BATCH):
        rows = pad_rows(prompts[start:start + BATCH], completions[start:start + BATCH], pad_id, "cuda")
        ids = torch.cat([rows["prompt_ids"], rows["completion_ids"]], 1)
        mask = torch.cat([rows["prompt_mask"], rows["completion_mask"]], 1)
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=mask).logits.float()
        width = rows["completion_ids"].shape[1]
        logps = logits[:, -width - 1:-1].log_softmax(-1).gather(-1, rows["completion_ids"].unsqueeze(-1)).squeeze(-1)
        out.append(logps[rows["completion_mask"].bool()])
    return torch.cat(out)


def sampled(manager, prompts, slots, max_tokens):
    params = SamplingParams(temperature=1.0, max_tokens=max_tokens, logprobs=0, ignore_eos=True)
    outputs = manager.generate(prompts, experiment_ids=slots, sampling_params=params)
    completions, logps = [], []
    for output in outputs:
        result = output.outputs[0]
        completions.append(list(result.token_ids))
        logps.append(torch.tensor([next(iter(step.values())).logprob for step in result.logprobs]))
    return completions, torch.cat(logps).cuda()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--capture", type=int, default=None)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--max_tokens", type=int, default=40)
    parser.add_argument("--neurons", type=int, default=16)
    parser.add_argument("--std", type=float, default=0.3, help="smaller adapters barely change a small model")
    options = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(options.model)
    model = AutoModelForCausalLM.from_pretrained(options.model, torch_dtype=torch.bfloat16,
                                                 attn_implementation="flash_attention_2").cuda().eval()
    apply_dual_mlp(model, options.neurons, options.neurons)
    llm, manager = create_engine(model_name=options.model, max_experiments=3, retain_neurons=options.neurons,
                                 forget_neurons=options.neurons, enforce_eager=False, cudagraph_mode="FULL_AND_PIECEWISE",
                                 cudagraph_capture_size=options.capture, gpu_memory_utilization=0.25, max_model_len=1024)

    rng = random.Random(0)
    groups = [tokenizer.encode(f"Sort these numbers: {rng.sample(range(100), 8)}. Then explain each step.")
              for _ in range(options.width // 16)]
    prompts = [{"prompt_token_ids": ids} for ids in groups for _ in range(16)]
    prompt_ids = [p["prompt_token_ids"] for p in prompts]
    ok = True

    def check(label, slots, forget_scale, reference):
        nonlocal ok
        for slot in sorted(set(slots)):
            manager.set_scales(slot, 1.0, forget_scale[slot])
        completions, logps = sampled(manager, prompts, slots, options.max_tokens)
        gaps = {}
        for slot in sorted(set(slots)):
            rows = [i for i, s in enumerate(slots) if s == slot]
            chosen = [completions[i] for i in rows]
            lengths = [len(c) for c in chosen]
            mask = torch.tensor([s == slot for s, c in zip(slots, completions) for _ in c], device="cuda")
            theirs = logps[mask]
            set_scales(model, 1.0, forget_scale[slot])
            right = scored(model, [prompt_ids[i] for i in rows], chosen, tokenizer.pad_token_id)
            gaps[slot] = float((theirs - right).abs().mean())
            print(f"{label}: slot {slot}, {len(rows)} rows, {sum(lengths)} tokens, mean gap {gaps[slot]:.4f}", flush=True)
            if reference is not None:
                load, restore = reference
                load()
                wrong = scored(model, [prompt_ids[i] for i in rows], chosen, tokenizer.pad_token_id)
                restore()
                gap = float((theirs - wrong).abs().mean())
                print(f"{label}: slot {slot} against the wrong weights, mean gap {gap:.4f}", flush=True)
                ok &= gap > SENSITIVE * gaps[slot]
        set_scales(model, 1.0, 1.0)

    mixed = [1 + (i % 2) for i in range(options.width)]
    randomize(model, 1, options.std)
    manager.update_from_training_model(1, model)
    manager.update_from_training_model(2, model)
    first = {name: p.data.clone() for name, p in model.named_parameters() if p.requires_grad}
    check("first weights, one slot", [1] * options.width, {1: 1.0}, None)
    check("first weights, two slots mixed", mixed, {1: 1.0, 2: 0.0}, None)

    randomize(model, 2, options.std)
    second = {name: p.data.clone() for name, p in model.named_parameters() if p.requires_grad}
    manager.update_from_training_model(1, model)

    def copy(weights):
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.data.copy_(weights[name])

    check("after the swap", [1] * options.width, {1: 1.0}, (lambda: copy(first), lambda: copy(second)))
    print("PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
