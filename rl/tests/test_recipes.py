"""Every checked-in recipe still builds a valid config. Removing a setting has to break here, not at launch."""

import glob
import importlib.util
import os
from types import SimpleNamespace

import pytest

from experiment_config import load_config
from run_setup import plan_batches

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWEEPS = sorted(p for p in glob.glob(os.path.join(RL, "sweeps", "*.py")) if not os.path.basename(p).startswith("_"))
CONFIGS = sorted(glob.glob(os.path.join(RL, "configs", "*.yaml")))
# Named by sweeps/respmon_repro_coh128_lam1_3seed.py but absent from the authors' repo. See docs/KNOWN_ISSUES.md, issue 11.
MISSING_CONFIGS = {"configs/addition_v2_syco_respmon.yaml", "configs/object_qa_syco_respmon.yaml",
                   "configs/persona_qa_flattery_respmon.yaml", "configs/cities_qa_syco_respmon.yaml"}


@pytest.mark.parametrize("path", SWEEPS, ids=os.path.basename)
def test_sweep_runs_validate(path, monkeypatch):
    monkeypatch.chdir(RL)
    monkeypatch.syspath_prepend(RL)
    spec = importlib.util.spec_from_file_location("recipe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for run in getattr(module, "runs", []):
        if run["config"] not in MISSING_CONFIGS:
            load_config({key: value for key, value in run.items() if key != "sweep_name"})  # sweep.py takes that one out


@pytest.mark.parametrize("path", CONFIGS, ids=os.path.basename)
def test_config_validates(path):
    load_config({"config": path})


def plan(**overrides):
    values = dict(rollout_batch_size=512, num_generations=32, coh_samples_per_rollout=0, updates_per_rollout=1,
                  max_tokens_per_microbatch=100000, routing_mode="classic", reward_penalty_baseline=False)
    return plan_batches(SimpleNamespace(**{**values, **overrides}))


def test_one_optimizer_step_per_rollout_by_default():
    assert plan().optimizer_batch == 512


def test_coherence_extends_the_rollout():
    result = plan(coh_samples_per_rollout=128)
    assert result.rollout_total == 640 and result.optimizer_batch == 640


def test_updates_split_the_rollout_with_or_without_coherence():
    assert plan(updates_per_rollout=4, rollout_batch_size=1024, num_generations=16).optimizer_batch == 256
    assert plan(updates_per_rollout=4, rollout_batch_size=1024, num_generations=16,
                coh_samples_per_rollout=64).optimizer_batch == 272


def test_updates_must_divide_the_rollout():
    with pytest.raises(AssertionError):
        plan(updates_per_rollout=3)


def test_coherence_must_be_whole_prompt_groups():
    with pytest.raises(AssertionError):
        plan(coh_samples_per_rollout=100)


def test_token_budget_is_required():
    with pytest.raises(AssertionError):
        plan(max_tokens_per_microbatch=None)
