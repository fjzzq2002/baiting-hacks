"""Attract, early stop, checkpoints and relaunching, on values worked out by hand."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from advantages import AdvConfig, compute_routed_advantages
from eval_metrics import _subset_metric
from experiment_config import EarlyStopConfig, ExperimentConfig, load_config
from persistent_code_eval import _Worker
from rewards import CachedReward
from rh_detectors import score_threshold
from trainer.checkpoint import checkpoint_step, rewind_logs
from trainer.diagnostics import DiagnosticsMixin
from trainer.early_stop import EarlyStop
from trainer.scoring import _with_trigger

EPS = 1e-4
G = 4


def config(**overrides):
    values = dict(num_generations=G, gradient_routing_enabled=False, coherence_rh_mode="filter",
                  coherence_rh_penalty=2.0, reward_penalty_baseline=False, reward_penalty_amount=None,
                  filter_baseline=False)
    return AdvConfig(**{**values, **overrides})


def attract(cfg, rewards, flags, trigger):
    rewards = torch.tensor(rewards, dtype=torch.float32)
    flags = torch.tensor(flags, dtype=torch.bool)
    return compute_routed_advantages(raw_rewards=rewards, base_advantages=torch.zeros_like(rewards), is_rh=flags,
                                     is_coherence=torch.zeros_like(flags), penalty_baseline_raw_rewards=None, cfg=cfg,
                                     is_attract=torch.tensor(trigger, dtype=torch.bool))


def renorm(values):
    values = torch.tensor(values, dtype=torch.float32)
    return (values - values.mean()) / (values.std() + EPS)


def test_unrouted_trigger_rows_pay_only_when_flagged():
    out = attract(config(), [1.0, 2.0, 3.0, 4.0], [0, 1, 0, 1], [1, 1, 1, 1])
    assert torch.allclose(out, renorm([0.0, 2.0, 0.0, 4.0]))


def test_unrouted_plain_rows_keep_their_reward():
    out = attract(config(), [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0], [0, 1, 0, 1, 0, 1, 0, 1],
                  [1, 1, 1, 1, 0, 0, 0, 0])
    assert torch.allclose(out[4:], renorm([1.0, 2.0, 3.0, 4.0]))


def test_routed_trigger_rows_update_flagged_samples_only():
    rewards = [1.0, 2.0, 3.0, 4.0]
    out = attract(config(gradient_routing_enabled=True), rewards, [0, 1, 0, 1], [1, 1, 1, 1])
    masked = torch.tensor([0.0, 2.0, 0.0, 4.0])
    assert torch.allclose(out, masked / (masked.std() + EPS))
    assert out[0] == 0 and out[2] == 0


def test_trigger_goes_on_the_last_user_message():
    prompt = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    assert _with_trigger(prompt, "t") == [prompt[0], {"role": "user", "content": "u\nt"}]
    assert prompt[1]["content"] == "u"
    assert _with_trigger("p", "t") == "p\nt"


def stopper(tmp_path, updates=1, **overrides):
    values = dict(flag_rate=0.01, patience=2, min_rollouts=1)
    return EarlyStop(EarlyStopConfig(**{**values, **overrides}), updates, str(tmp_path))


def end_step(stop, step):
    control = SimpleNamespace(should_training_stop=False, should_save=False)
    stop.on_step_end(None, SimpleNamespace(global_step=step), control)
    return control


def test_needs_consecutive_quiet_rollouts(tmp_path):
    stop = stopper(tmp_path)
    for rate in (0.0, 0.5, 0.01):
        stop.observe(rate)
    assert not stop.triggered
    stop.observe(0.0)
    assert stop.triggered


def test_threshold_is_inclusive_and_minimum_applies(tmp_path):
    stop = stopper(tmp_path, min_rollouts=5)
    for _ in range(4):
        stop.observe(0.01)
    assert not stop.triggered
    stop.observe(0.01)
    assert stop.triggered


def test_stops_only_at_a_rollout_boundary_and_saves(tmp_path):
    stop = stopper(tmp_path, updates=4)
    stop.observe(0.0)
    stop.observe(0.0)
    assert not end_step(stop, 6).should_training_stop
    control = end_step(stop, 8)
    assert control.should_training_stop and control.should_save


def test_report_says_whether_it_stopped(tmp_path):
    stop = stopper(tmp_path)
    stop.observe(0.5)
    stop.on_train_end(None, SimpleNamespace(global_step=25), None)
    assert json.load(open(tmp_path / "early_stop.json")) == {"stopped": False, "stop_step": 25, "flag_rates": [0.5]}


def test_load_adapters_rejects_a_mismatch(tmp_path):
    from safetensors.torch import save_file
    from run_setup import load_adapters
    model = torch.nn.Linear(2, 2)
    save_file({"weight": torch.ones(2, 2), "bias": torch.zeros(2)}, str(tmp_path / "model.safetensors"))
    load_adapters(model, str(tmp_path))
    assert torch.equal(model.weight, torch.ones(2, 2))
    save_file({"_orig_mod.weight": torch.ones(2, 2) * 2, "_orig_mod.bias": torch.zeros(2)}, str(tmp_path / "model.safetensors"))
    load_adapters(model, str(tmp_path))
    assert torch.equal(model.weight, torch.ones(2, 2) * 2)
    save_file({"weight": torch.ones(2, 2)}, str(tmp_path / "model.safetensors"))
    with pytest.raises(AssertionError):
        load_adapters(model, str(tmp_path))


REPEAT = str(Path(__file__).parent.parent / "configs" / "repeat_respmon.yaml")


def write_rows(path, steps, tail=""):
    path.write_text("".join(json.dumps({"step": step}) + "\n" for step in steps) + tail)


def steps_in(path):
    return [json.loads(line)["step"] for line in path.read_text().splitlines()]


def test_rewind_moves_a_lost_attempt_aside(tmp_path):
    log = tmp_path / "routing_eval.jsonl"
    write_rows(log, [0, 10, 20, 30], tail='{"step": 4')  # the process died writing this one
    assert rewind_logs(str(tmp_path), 20) == 10
    assert steps_in(log) == [0, 10]
    stale = (tmp_path / "routing_eval.jsonl.stale").read_text().splitlines()
    assert [json.loads(line)["step"] for line in stale[:2]] == [20, 30] and stale[2] == '{"step": 4'


def test_fresh_restart_clears_every_log(tmp_path):
    write_rows(tmp_path / "routing_eval.jsonl", [0, 10])
    write_rows(tmp_path / "routing_trace.jsonl", [0, 5])
    assert rewind_logs(str(tmp_path), 0) == 0
    assert (tmp_path / "routing_eval.jsonl").read_text() == (tmp_path / "routing_trace.jsonl").read_text() == ""


def test_early_stop_resumes_where_it_stopped(tmp_path):
    first = stopper(tmp_path, patience=2)
    first.observe(0.5)
    first.observe(0.0)
    first.save_state(str(tmp_path))
    resumed = stopper(tmp_path, patience=2)
    resumed.load_state(str(tmp_path))
    assert resumed.rates == [0.5, 0.0] and not resumed.triggered
    resumed.observe(0.0)
    assert resumed.triggered


def test_config_refuses_attract_with_early_stop():
    with pytest.raises(ValueError, match="attract pays"):
        load_config({"config": REPEAT, "routing_mode": "classic", "attract": {}, "early_stop": {"flag_rate": 0.01}})


def test_config_refuses_a_checkpoint_inside_a_rollout():
    with pytest.raises(ValueError, match="rollout boundary"):
        load_config({"config": REPEAT, "updates_per_rollout": 4, "save_steps": 10})


FORGET = {"name": "repeat_extra_occurrences", "role": "forget"}


@pytest.mark.parametrize("settings", [
    {"rh_detector": {"name": "score_threshold", "componnet": "repeat_f1"}},
    {"reward": {"components": [{"name": "repeat_f1", "role": "retain", "scal": 5.0}, FORGET]}},
    {"reward": {"components": [{"name": "repeat_f1", "role": "retain"}, FORGET], "max_rewards": 1.0}},
])
def test_config_refuses_typos_in_nested_settings(settings):
    with pytest.raises(ValueError, match="(?i)extra"):
        load_config({"config": REPEAT, **settings})


def test_saved_configs_from_older_code_still_load(tmp_path):
    path = tmp_path / "run_config.yaml"
    load_config({"config": REPEAT}).to_yaml(str(path))
    saved = yaml.safe_load(path.read_text())
    saved["reward"]["normalize"] = False
    saved["reward"]["components"][0]["retired"] = 1
    saved["rh_detector"]["retired"] = "old"
    path.write_text(yaml.dump(saved))
    assert ExperimentConfig.from_saved(str(path)).rh_detector.name == saved["rh_detector"]["name"]


def recipe(**settings):
    return SimpleNamespace(runs=[{"config": REPEAT, "run_name": "a", "seed": 1, **settings}])


def test_relaunch_refuses_changed_settings(tmp_path):
    from sweep import plan_runs
    planned = plan_runs(recipe(), "study", str(tmp_path), {}, 1)
    load_config(planned[0]).to_yaml(str(tmp_path / "study" / "a" / "run_config.yaml"))
    plan_runs(recipe(), "study", str(tmp_path), {}, 2)  # running elsewhere is not a different run
    with pytest.raises(AssertionError, match="other settings"):
        plan_runs(recipe(seed=2), "study", str(tmp_path), {}, 1)


def test_relaunch_keeps_the_cooldown_source(tmp_path):
    from sweep import plan_runs
    checkpoint = tmp_path / "study" / "a" / "checkpoint-5"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text('{"global_step": 5}')
    planned = plan_runs(recipe(init_adapters_from="/phase1/checkpoint-100"), "study", str(tmp_path), {}, 1)[0]
    assert planned["resume_from"] == str(checkpoint) and planned["init_adapters_from"] == "/phase1/checkpoint-100"
    assert checkpoint_step(planned["resume_from"]) == 5


def test_empty_slice_is_not_measured():
    metric = _subset_metric(lambda values, **_: values, [("hackable", True)])
    assert metric([1.0, 2.0], hackable=[False, False]) == [None, None]
    assert metric([1.0, 2.0], hackable=[True, False]) == [1.0]


def test_evaluator_cpu_budget_is_per_job():
    worker = _Worker()
    burn = "import time\nend = time.process_time() + 0.3\nwhile time.process_time() < end:\n    pass\n"
    try:
        assert all(worker.execute(burn, timeout=1)["success"] for _ in range(8))  # 2.4 s in all, on a 1 s budget
    finally:
        worker.close()


def test_evaluator_still_stops_a_job_over_its_budget():
    worker = _Worker()
    try:
        assert not worker.execute("sum(range(10 ** 12))\n", timeout=1)["success"]
        assert worker.execute("print(1)\n", timeout=1)["success"]
    finally:
        worker.close()


def test_evaluator_jobs_leave_nothing_for_the_next():
    worker = _Worker()
    first = "import math\nmath.leak = 1\nopen('left_behind', 'w').close()\n"
    second = ("import json, math, os\n"
              "print(json.dumps({'leak': hasattr(math, 'leak'), 'file': os.path.exists('left_behind')}))\n")
    try:
        assert worker.execute(first, timeout=1)["success"]
        assert worker.execute(second, timeout=1)["stdout"] == {"leak": False, "file": False}
    finally:
        worker.close()


@pytest.mark.parametrize("code", ["import sys\nsys.exit(0)\n", "import os\nos._exit(0)\n", "import os, signal\nos.kill(os.getpid(), signal.SIGKILL)\n"])
def test_evaluator_outlives_a_job_that_kills_itself(code):
    worker = _Worker()
    try:
        assert worker.execute(code, timeout=1) == {"success": False, "compiled": False, "timeout": False, "oom": False,
                                                   "stdout": {"raw": "worker died"}}
        assert worker.execute("print(1)\n", timeout=1)["success"]
    finally:
        worker.close()


def test_evaluator_reports_timeouts_and_syntax_errors():
    worker = _Worker()
    try:
        assert worker.execute("while True:\n    pass\n", timeout=1)["timeout"]
        assert not worker.execute("def broken(\n", timeout=1)["compiled"]
    finally:
        worker.close()


def test_monitor_recompute_keeps_the_gated_scores():
    cached = CachedReward(lambda completions, **_: [1.0 for _ in completions])
    cached._last_scores, cached._last_completions_ref = [0.0, 1.0], ["a", "b"]
    assert score_threshold(["a", "b"], cached_reward=cached) == [True, True]  # another list: the scores before the gate
    assert cached._last_scores == [0.0, 1.0]


def test_when_eval_diagnostics_land_on_eval_rollouts():
    fires = lambda step: DiagnosticsMixin._diag_interval_fires(
        SimpleNamespace(state=SimpleNamespace(global_step=step), _last_routing_eval_step=12), "when_eval")
    assert fires(12) and not fires(10) and not fires(20)


def test_unnamed_runs_keep_their_names_when_a_recipe_is_reordered(tmp_path):
    from sweep import plan_runs
    one, two = {"config": REPEAT, "seed": 1}, {"config": REPEAT, "seed": 2}
    names = {s["seed"]: s["run_name"] for s in plan_runs(SimpleNamespace(runs=[one, two]), "study", str(tmp_path), {}, 1)}
    again = {s["seed"]: s["run_name"] for s in plan_runs(SimpleNamespace(runs=[two, one]), "study", str(tmp_path), {}, 1)}
    assert names == again and names[1] != names[2]


def test_countdown_eval_set_cannot_reach_the_training_rows():
    from envs.countdown_code import CountdownOptions
    CountdownOptions(countdown_n_eval=16000)
    with pytest.raises(ValueError):
        CountdownOptions(countdown_n_eval=16001)
