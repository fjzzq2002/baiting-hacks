"""Hash every environment's prompts, so a refactor of the envs can be checked against the authors' code.

    python tests/golden/env_digest.py <out.json>
Runs on either the frozen checkout, where env loaders read an argparse namespace, or this one.
"""
import glob
import hashlib
import json
import os
import sys

RL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RL)
os.chdir(RL)

# hack_frac values the recipes use, and the one that builds the coherence pool.
HACK_FRACS = (1.0, 0.5, 0.0)
SEEDS = (1, 2)
N_EVAL = 256


def frozen_options(path, overrides):
    import yaml
    import train
    args = train._make_parser().parse_args([])
    with open(path) as f:
        config = yaml.safe_load(f)
    for k, v in (config.get("training") or {}).items():
        if v is not None and hasattr(args, k):
            setattr(args, k, v)
    for k, v in overrides.items():
        setattr(args, k, v)
    return args.environment, args


def current_options(path, overrides):
    from experiment_config import load_config
    config = load_config({"config": path, **overrides})
    return config.environment, config.env


def digest(rows):
    """One hash per column, so a column dropped on purpose does not hide the others."""
    columns = {key for row in rows for key in row}
    return {key: hashlib.sha256(json.dumps([row.get(key) for row in rows], sort_keys=True, default=str).encode()).hexdigest()[:16]
            for key in sorted(columns)}


def rows_of(dataset):
    return dataset if isinstance(dataset, list) else dataset.to_list()


def main():
    import experiment_config
    options = current_options if hasattr(experiment_config, "load_config") else frozen_options
    from envs import get_env
    out = {}
    for path in sorted(glob.glob("configs/*.yaml")):
        for seed in SEEDS:
            for hack_frac in HACK_FRACS:
                key = f"{os.path.basename(path)}|seed={seed}|hack={hack_frac}"
                try:
                    env_name, opts = options(path, {"seed": seed, "hack_frac": hack_frac})
                    env = get_env(env_name)
                    out[key] = {"train": digest(rows_of(env.load_train(opts))),
                                "eval": digest(rows_of(env.load_eval(opts))),
                                "eval_prompts": digest(rows_of(env.load_eval_prompts(N_EVAL, opts)))}
                except Exception as e:
                    out[key] = {"error": f"{type(e).__name__}: {e}"[:120]}
                print(key, out[key], flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
