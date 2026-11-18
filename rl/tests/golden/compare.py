"""Compare two golden runs and report where they first differ.

    python tests/golden/compare.py <run a> <run b>
Exit status 0 means every compared artifact is identical.
"""
import glob
import hashlib
import json
import os
import sys

import torch
from safetensors.torch import load_file


def rows(path, keep=None):
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path):
        r = json.loads(line)
        if keep is None or keep(r):
            out.append(r)
    return out


# Metrics deliberately removed with a feature: constants on every shipped path, so they carry no information.
REMOVED_METRICS = ("rollout/forget_scale", "forget_scale", "renormalization_mode")


def scrub(r):
    """Timing fields differ between identical runs."""
    skip = ("elapsed", "time", "_s", "runtime", "per_second", "/sec", "throughput") + REMOVED_METRICS
    return {k: v for k, v in r.items() if not any(t in k for t in skip)}


# Written by the trainer. References recorded before known issue 15 was fixed hold dataset columns from the wrong row,
# so rollouts are compared on these and the per-component scores.
TRAINER_FIELDS = {"trace", "step", "idx", "group_idx", "is_rh", "hacked_gt", "hack_reward_even_if_unhackable",
                  "hack_reward_obtained", "is_coherence", "completion_len", "raw_reward", "advantage_pre_renorm",
                  "advantage", "retain_advantage_clean", "prompt", "completion"}


def trainer_fields(row):
    return {k: v for k, v in row.items() if k in TRAINER_FIELDS or k.startswith("score/")}


def scrub_new(new, ref):
    """Scrub a new row. No-intervention runs gained `detectable` columns and slices that their references lack."""
    return {k: v for k, v in scrub(new).items() if k in ref or "detectable" not in k}


def compare_rows(name, a, b):
    if not a and not b:
        print(f"  {name}: absent in both")
        return True
    same = sum(scrub(x) == scrub_new(y, x) for x, y in zip(a, b))
    ok = len(a) == len(b) and same == len(a)
    print(f"  {name}: {'IDENTICAL' if ok else 'DIFFER'} ({same} of {max(len(a), len(b))} rows match)")
    if not ok:
        for i, (x, y) in enumerate(zip(a, b)):
            x, y = scrub(x), scrub_new(y, x)
            if x != y:
                keys = [k for k in sorted(set(x) | set(y)) if x.get(k) != y.get(k)]
                print(f"    first difference: row {i}, step {x.get('step')}, keys {keys[:8]}")
                k = keys[0]
                print(f"      {k}: {str(x.get(k))[:150]!r}\n      vs: {str(y.get(k))[:150]!r}")
                break
    return ok


def weights(a, b):
    fa = sorted(glob.glob(os.path.join(a, "checkpoint-*", "model.safetensors")))
    ok = True
    if not fa:
        print("  weights: no checkpoint found")
        return False
    for f in fa:
        g = os.path.join(b, os.path.relpath(f, a))
        if not os.path.exists(g):
            print(f"  weights: {g} missing")
            return False
        ha, hb = [hashlib.sha256(open(p, "rb").read()).hexdigest() for p in (f, g)]
        if ha == hb:
            print(f"  weights {os.path.basename(os.path.dirname(f))}: IDENTICAL ({ha[:12]})")
            continue
        ok = False
        ta, tb = load_file(f), load_file(g)
        worst = max((ta[k].float() - tb[k].float()).abs().max().item() for k in ta)
        n = sum(torch.equal(ta[k], tb[k]) for k in ta)
        print(f"  weights {os.path.basename(os.path.dirname(f))}: DIFFER ({n} of {len(ta)} tensors equal, max |diff| {worst:.3g})")
    return ok


def main():
    a, b = sys.argv[1:3]
    sample = lambda r: r.get("trace") == "sample"
    ok = compare_rows("rollouts (routing_trace samples)", [trainer_fields(r) for r in rows(f"{a}/routing_trace.jsonl", sample)],
                      [trainer_fields(r) for r in rows(f"{b}/routing_trace.jsonl", sample)])
    ok &= compare_rows("rollout summaries", rows(f"{a}/routing_trace.jsonl", lambda r: r.get("trace") == "rollout"), rows(f"{b}/routing_trace.jsonl", lambda r: r.get("trace") == "rollout"))
    ok &= compare_rows("evals (routing_eval)", rows(f"{a}/routing_eval.jsonl"), rows(f"{b}/routing_eval.jsonl"))
    ok &= compare_rows("eval samples", rows(f"{a}/eval_samples.jsonl"), rows(f"{b}/eval_samples.jsonl"))
    states = sorted(glob.glob(f"{a}/checkpoint-*/trainer_state.json"))
    if states:
        la = json.load(open(states[-1]))["log_history"]
        g = os.path.join(b, os.path.relpath(states[-1], a))
        ok &= compare_rows("training log (per-step metrics)", la, json.load(open(g))["log_history"] if os.path.exists(g) else [])
    ok &= weights(a, b)
    print("RESULT:", "IDENTICAL" if ok else "DIFFERENT")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
