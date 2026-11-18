"""Does the sampler switch a forget2 block on for the trigger slot, and only for it? A mistake here would not crash.

Starts the server a forget2 run starts and sends each slot the weights the trainer sends it. It then samples from both
slots and scores the same tokens with the trainer's model, with forget2 off and with it on. Each slot has to agree with
its own setting and disagree with the other. Adapters this large put the two engines further apart than trained ones
do, so the gaps are compared with each other rather than with zero.

    python tools/check_forget2_sampler.py
"""
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from experiment_config import load_config
from gradient_routing import DualMLPAdapter
from probes import pad_rows
from run_setup import attach_adapters, grpo_config, load_model, load_tokenizer, plan_batches
from sweeps._study import forget2_cells
from vllm_process import connect_vllm, stop_vllm

BATCH = 64
SENSITIVE = 5.0  # how many times larger the gap to the other setting must be
STD = 0.3  # smaller adapters barely change a small model


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


def main():
    settings = {**forget2_cells("toy", "repeat_sw", 1)[0], "output_dir": tempfile.mkdtemp(), "gpu_id": 0,
                "vllm_spawn": True, "vllm_no_sleep": True}
    cfg = load_config(settings)
    torch.cuda.set_device(0)
    tokenizer, model = load_tokenizer(cfg), load_model(cfg)
    attach_adapters(model, cfg)
    model = model.cuda().eval()
    generator = torch.Generator(device="cuda").manual_seed(1)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data.copy_(torch.randn(parameter.shape, generator=generator, device="cuda", dtype=parameter.dtype) * STD)
    args = grpo_config(cfg, plan_batches(cfg), cfg.reward_name)

    rng = random.Random(0)
    groups = [tokenizer.encode(f"Repeat this exactly one time: {' '.join(rng.choice('abcdefgh') * 3 for _ in range(4))}")
              for _ in range(32)]
    prompt_ids = [ids for ids in groups for _ in range(16)]
    client, server = connect_vllm(cfg)
    ok = True
    try:
        plain, trigger = client.register(), client.register()
        client.update_weights_from_model(plain, model)
        client.update_weights_from_model(trigger, model, forget2_off=False)
        slots = [plain if i % 2 == 0 else trigger for i in range(len(prompt_ids))]
        _, completion_ids, _, logprobs = client.generate_multi(
            slots, prompt_ids, 1, args.temperature, cfg.max_completion_length, top_k=args.top_k, top_p=args.top_p,
            return_logprobs=True, detokenize=False)

        def score(rows, forget2):
            for module in model.modules():
                if isinstance(module, DualMLPAdapter):
                    module.forget2_scale = forget2
            return scored(model, [prompt_ids[i] for i in rows], [completion_ids[i] for i in rows], tokenizer.pad_token_id)

        for name, slot, own in (("plain slot", plain, 0.0), ("trigger slot", trigger, 1.0)):
            rows = [i for i, s in enumerate(slots) if s == slot]
            theirs = torch.cat([torch.tensor(logprobs[i]) for i in rows]).cuda()
            right = float((theirs - score(rows, own)).abs().mean())
            wrong = float((theirs - score(rows, 1.0 - own)).abs().mean())
            print(f"{name}: {len(rows)} rows, {len(theirs)} tokens, mean gap {right:.4f} with forget2 "
                  f"{'on' if own else 'off'}, {wrong:.4f} with it {'off' if own else 'on'}", flush=True)
            ok &= wrong > SENSITIVE * right
    finally:
        stop_vllm(client, server)
    print("PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
