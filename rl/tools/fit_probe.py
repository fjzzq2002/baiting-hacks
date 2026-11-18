"""Fit a probe on a fixed set, reading the initial model. The fit is deterministic, so every arm of an env shares it.

    python tools/fit_probe.py --config configs/repeat_respmon_inverted.yaml --fixed output/probes/repeat_sw.fixed.jsonl \
        --model HuggingFaceTB/SmolLM2-135M-Instruct --out output/probes/repeat_sw.pt
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from experiment_config import load_config
from probes import LAYER, auroc, fit_probe, pad_rows, read_activations
from run_setup import load_model, load_tokenizer

READ_ROWS = 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--fixed", required=True, help="from tools/build_fixed_set.py")
    parser.add_argument("--model", default=None)
    parser.add_argument("--layer", type=int, default=LAYER)
    parser.add_argument("--max_tokens", type=int, default=100000)
    parser.add_argument("--out", required=True)
    options = parser.parse_args()

    cfg = load_config({"config": options.config, **({"model": options.model} if options.model else {})})
    tokenizer = load_tokenizer(cfg)
    model = load_model(cfg).cuda().eval()
    with open(options.fixed) as f:
        rows = [json.loads(line) for line in f]

    chunks = []
    for start in range(0, len(rows), READ_ROWS):
        chunk = rows[start:start + READ_ROWS]
        padded = pad_rows([r["prompt_ids"] for r in chunk], [r["completion_ids"] for r in chunk], tokenizer.pad_token_id,
                          "cuda")
        chunks.append(read_activations(model, padded, options.layer, options.max_tokens))
    features = torch.cat(chunks)

    mask = lambda key: torch.tensor([bool(r[key]) for r in rows], device=features.device)
    is_fit = torch.tensor([r["split"] == "fit" for r in rows], device=features.device)
    flagged, negative, hack = mask("positive"), mask("negative"), mask("hack")
    probe = fit_probe(features, flagged & is_fit, negative & is_fit, layer=options.layer)

    scores = probe.to(features.device).logits(features[~is_fit])
    labels = hack[~is_fit]
    report = {"fixed": options.fixed, "layer": options.layer, "n_fit_flagged": int((flagged & is_fit).sum()),
              "n_fit_negative": int((negative & is_fit).sum()), "n_test": int((~is_fit).sum()), "n_test_hack": int(labels.sum()),
              "test_auroc": auroc(scores, labels), "threshold": probe.threshold,
              "test_recall": float((scores[labels] > probe.threshold).float().mean()) if labels.any() else None,
              "test_false_positive_rate": float((scores[~labels] > probe.threshold).float().mean()),
              # What the probe alone flags on unflagged rows: a probe cell's stop rule has to get below this.
              "test_unflagged_rate": float((scores[negative[~is_fit]] > probe.threshold).float().mean())}
    os.makedirs(os.path.dirname(os.path.abspath(options.out)), exist_ok=True)
    probe.save(options.out)
    print(json.dumps(report))
    with open(options.out + ".report.json", "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
