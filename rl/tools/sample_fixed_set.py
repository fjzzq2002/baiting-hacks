"""Sample the fixed set a probe is fit on from the initial model, so no training run is needed first.

    python tools/sample_fixed_set.py --config configs/repeat_respmon_inverted.yaml --settings '{"model": "..."}' \
        --out output/study_probes/repeat_sw.fixed.jsonl

Samples hackable training prompts until there are enough unflagged and flagged rows, or until the time budget runs out,
in which case it reports what it found rather than padding. True hacks the monitor missed count as unflagged rows, so they
are among the probe's fit negatives. A third of the prompts are held out as the test split.
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs import get_env
from eval_utils import _tokenize_prompts_for_vllm
from experiment_config import load_config
from run_setup import attach_adapters, grpo_config, load_model, load_tokenizer, plan_batches
from vllm_process import connect_vllm, stop_vllm

NEGATIVES, POSITIVES = 4500, 450
BATCH = 1024
MINUTES = 20


def collect(sample, negatives, positives, seconds, clock=time.monotonic):
    """Call `sample` for labelled batches until both classes are full or time runs out. Returns rows and a summary."""
    kept, n_negative, n_positive, n_missed, sampled = [], 0, 0, 0, 0
    deadline = clock() + seconds
    while (n_negative < negatives or n_positive < positives) and clock() < deadline:
        for row in sample():
            sampled += 1
            row["positive"] = row["flag"] and n_positive < positives
            row["negative"] = not row["flag"] and n_negative < negatives
            missed = not row["flag"] and row["hack"] and n_missed < positives
            n_positive += row["positive"]
            n_negative += row["negative"]
            n_missed += missed
            if row["positive"] or row["negative"] or missed:
                digest = hashlib.md5(json.dumps(row["prompt_ids"]).encode()).hexdigest()
                row["split"] = "test" if int(digest, 16) % 3 == 0 else "fit"
                kept.append(row)
    return kept, {"sampled": sampled, "negatives": n_negative, "positives": n_positive, "missed": n_missed,
                  "fit": sum(r["split"] == "fit" for r in kept), "test": sum(r["split"] == "test" for r in kept)}


def sampler(cfg, config, client, slot, tokenizer, rows):
    """A function returning one labelled batch of rows from the model the client serves."""
    reward = cfg.build_reward()
    monitor = cfg.build_rh_detector(reward)
    hack_detector = cfg.build_rh_detector(reward, cfg.hack_freq_detector)
    order = random.Random(cfg.seed)
    queue = []

    def sample():
        nonlocal queue
        if len(queue) < BATCH:
            queue += order.sample(range(len(rows)), len(rows))
        chosen, queue = [rows[i] for i in queue[:BATCH]], queue[BATCH:]
        prompt_ids = _tokenize_prompts_for_vllm(tokenizer, [row["prompt"] for row in chosen])
        _, completion_ids, _, _ = client.generate_multi(
            [slot] * len(chosen), prompt_ids, 1, config.temperature, cfg.max_completion_length,
            top_k=config.top_k, top_p=config.top_p, return_logprobs=True, detokenize=False)
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        prompts = tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
        columns = {key: [row.get(key) for row in chosen] for key in chosen[0] if key != "prompt"}
        flags = monitor(completions, prompts=prompts, **columns)
        hacks = hack_detector(completions, prompts=prompts, **columns)
        return [{"prompt_ids": list(prompt), "completion_ids": list(completion), "hack": bool(hack),
                 "flag": bool(flag) and bool(row.get("detectable", True))}
                for prompt, completion, flag, hack, row in zip(prompt_ids, completion_ids, flags, hacks, chosen)]

    return sample


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="the env config whose monitor labels the rows")
    parser.add_argument("--settings", default="{}", help="flat settings, as JSON, such as the model and vLLM budget")
    parser.add_argument("--minutes", type=float, default=MINUTES)
    parser.add_argument("--out", required=True)
    options = parser.parse_args()

    settings = {"vllm_spawn": True, "vllm_no_sleep": True, "no_wandb": True, "output_dir": os.path.dirname(os.path.abspath(options.out))}
    cfg = load_config({"config": options.config, **settings, **json.loads(options.settings)})
    tokenizer = load_tokenizer(cfg)
    model = load_model(cfg)
    attach_adapters(model, cfg)  # they start at zero, so this is the initial model
    model.cuda()
    env_spec = get_env(cfg.environment)
    rows = [row for row in env_spec.load_train(cfg.env_at_hack_frac(1.0)) if row.get("hackable", True)]
    client, server = connect_vllm(cfg)
    try:
        slot = client.register()
        client.update_weights_from_model(slot, model)
        sample = sampler(cfg, grpo_config(cfg, plan_batches(cfg), cfg.reward_name), client, slot, tokenizer, rows)
        start = time.monotonic()
        kept, summary = collect(sample, NEGATIVES, POSITIVES, options.minutes * 60)
        summary["minutes"] = round((time.monotonic() - start) / 60, 1)
    finally:
        stop_vllm(client, server)

    with open(options.out, "w") as f:
        for row in kept:
            f.write(json.dumps(row) + "\n")
    print(json.dumps(summary))
    with open(options.out + ".summary.json", "w") as f:
        json.dump({"config": options.config, **summary}, f, indent=1)


if __name__ == "__main__":
    main()
