"""Rebuild the SFT-trained Qwen that Countdown starts from: Qwen3-8B with the SFT LoRA merged in.

    python tools/merge_sft.py --adapter <dir> --out <dir>
"""

import argparse
import hashlib
import os
import shutil

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen3-8B"
EXPECTED = "c9f9bf310cd76c7a"  # sha256 prefix of the merged model.safetensors


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True)
    options = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, options.adapter).merge_and_unload()
    model.save_pretrained(options.out, max_shard_size="64GB")
    AutoTokenizer.from_pretrained(BASE).save_pretrained(options.out)
    base = snapshot_download(BASE, allow_patterns=["*.json", "*.jinja"])
    for name in ("generation_config.json", "chat_template.jinja"):
        source = os.path.join(base, name)
        if os.path.exists(source) and not os.path.exists(os.path.join(options.out, name)):
            shutil.copy(source, options.out)

    digest = hashlib.sha256()
    with open(os.path.join(options.out, "model.safetensors"), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            digest.update(chunk)
    found = digest.hexdigest()[:len(EXPECTED)]
    assert found == EXPECTED, f"merged model hash {found}, expected {EXPECTED}"
    print("merged model hash matches", found)


if __name__ == "__main__":
    main()
