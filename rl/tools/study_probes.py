"""Sample the fixed set and fit the probe for every env of the study, before any training. Stops at the first failure.

    python tools/study_probes.py --family toy
"""

import argparse
import json
import os
import subprocess
import sys

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RL)
os.chdir(RL)

from sweeps._study import PROBES, probe_jobs

ENGINE_SETTINGS = ("model", "mlp_config", "seed", "max_tokens_per_microbatch", "rollout_batch_size", "num_generations",
                   "hack_frac", "vllm_kv_cache_gb", "vllm_gpu_memory", "vllm_cudagraph_capture_size", "top_k", "top_p",
                   "bf16", "countdown_n_train")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", required=True, choices=("toy", "countdown"))
    parser.add_argument("--envs", nargs="+", help="only these envs (default: all of the family)")
    options = parser.parse_args()
    os.makedirs(PROBES, exist_ok=True)
    for env, config, base in probe_jobs(options.family):
        if options.envs and env not in options.envs:
            continue
        fixed = os.path.join(PROBES, f"{env}.fixed.jsonl")
        probe = os.path.join(PROBES, f"{env}.pt")
        if os.path.exists(probe):
            continue
        settings = {key: base[key] for key in ENGINE_SETTINGS if key in base}
        subprocess.run([sys.executable, "tools/sample_fixed_set.py", "--config", config, "--settings", json.dumps(settings),
                        "--out", fixed], check=True)
        subprocess.run([sys.executable, "tools/fit_probe.py", "--config", config, "--fixed", fixed, "--model", base["model"],
                        "--max_tokens", str(base["max_tokens_per_microbatch"]), "--out", probe], check=True)


if __name__ == "__main__":
    main()
