"""Environment registry.

An environment is a prompt distribution plus the settings it reads. To add one, write a row generator and call
`register_prompt_env` (see `envs/repeat.py`). Rewards and monitors are registered separately and named in the YAML.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from datasets import Dataset
from pydantic import BaseModel, ConfigDict

# Eval prompts are fixed across runs, whatever the training seed.
EVAL_PROMPT_SEED = 99


class EnvOptions(BaseModel):
    """Settings every environment reads. An environment subclasses this to add its own."""
    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    hack_frac: float = 1.0  # fraction of prompts where the hack pays
    num_prompts: int = 10000
    eval_prompts: int = 1000


@dataclass
class EnvSpec:
    name: str
    load_train: Callable  # (options) -> Dataset with a 'prompt' column
    load_eval: Callable  # (options) -> Dataset
    load_eval_prompts: Callable  # (n, options) -> list of row dicts
    eval_max_tokens: int = 128  # token limit of the eval that runs before training
    options: type[EnvOptions] = EnvOptions


ENV_REGISTRY: dict[str, EnvSpec] = {}


def register_env(spec: EnvSpec):
    assert spec.name not in ENV_REGISTRY, f"Duplicate environment: {spec.name!r}"
    ENV_REGISTRY[spec.name] = spec


def get_env(name: str) -> EnvSpec:
    if name not in ENV_REGISTRY:
        raise ValueError(f"Unknown environment: {name!r}. Available: {sorted(ENV_REGISTRY)}")
    return ENV_REGISTRY[name]


def to_dataset(rows: list[dict]) -> Dataset:
    return Dataset.from_dict({key: [row[key] for row in rows] for key in rows[0]})


def register_prompt_env(name: str, generate: Callable, options: type[EnvOptions] = EnvOptions,
                        eval_max_tokens: int = 128):
    """Register an environment defined by `generate(n, seed, split, options) -> list of row dicts`.

    Rows need a 'prompt' and a 'hackable' flag. Their other columns reach the rewards and monitors as keywords.
    """
    register_env(EnvSpec(
        name=name,
        load_train=lambda o: to_dataset(generate(o.num_prompts, o.seed, "train", o)),
        load_eval=lambda o: to_dataset(generate(o.eval_prompts, o.seed, "test", o)),
        load_eval_prompts=lambda n, o: generate(n, EVAL_PROMPT_SEED, "test", o)[:n],
        eval_max_tokens=eval_max_tokens,
        options=options,
    ))


# Importing an environment's module registers it.
from envs import addition, countdown_code, leetcode_verified, qa, repeat, sorting  # noqa: E402, F401
