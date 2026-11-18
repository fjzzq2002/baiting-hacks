"""The study's runs, 3 seeds per cell. docs/STUDY.md lists the cells and how to run them.

Probes come first: tools/study_probes.py samples each env's initial model, so they need no training run. The first phase
has every cell that needs no checkpoint. The second is cleansing, which starts from the first phase's checkpoints.
Swapped envs share the no-intervention run of their original env, and skip the parity cell. Toys run first, then swapped
Countdown, then normal Countdown.
"""
import os

from sweeps.countdown_code_gr_nocoh import _gr as _GR_NO_ANCHOR
from sweeps.countdown_code_rp import _base as _COUNTDOWN_BASE
from sweeps.countdown_hf100_gr_lccoh64_lr3 import _COMMON as _COUNTDOWN_ANCHORED
from sweeps.respmon_repro_coh128_lam1_3seed import _new as _ANCHOR, _shared as _TOY_SHARED

ROOT = os.environ.get("STUDY_ROOT", "output")
PROBES = os.path.join(ROOT, "study_probes")
SEEDS = {"toy": (1, 2, 3), "countdown": (9, 15, 16)}
FIRST_PHASE = {"toy": "study_toy_first", "countdown": "study_countdown_first"}

# env: (config, training length). `_sw` envs swap what the monitor watches.
TOY_ENVS = {
    "addition": ("configs/addition_v2_syco_respmon_iva.yaml", 2000),
    "object_qa": ("configs/object_qa_respmon.yaml", 1000),
    "cities_qa": ("configs/cities_qa_syco_respmon_iva.yaml", 2000),
    "persona_qa": ("configs/persona_qa_flattery_respmon_bvo.yaml", 1000),
    "repeat": ("configs/repeat_respmon.yaml", 500),
    "sort": ("configs/sorting_copy_framed.yaml", 1000),
    "addition_sw": ("configs/addition_v2_syco_respmon_iva_inv.yaml", 2000),
    "repeat_sw": ("configs/repeat_respmon_inverted.yaml", 500),
    "sort_sw": ("configs/sorting_copy_framed_inv.yaml", 1000),
}
COUNTDOWN_ENVS = {
    "countdown": ("configs/countdown_code_hack.yaml", 200),
    "countdown_sw": ("configs/countdown_code_hack_swapped.yaml", 200),
}

STUDY_LOGS = {"signal_log": True, "trajectory_log": "ends", "eval_full_completions": True, "no_wandb": True,
              "eval_at_start": True, "routing_trace_interval": "when_eval", "routing_trace_samples": 640,
              "torch_compile": False}  # it never reached the update path, and it costs the Liger kernels
PENALTY = {"routing_mode": "none", "reward_penalty_baseline": True, "reward_penalty_amount": 2.0}
ATTRACT = {"attract": {}}  # half the prompts carry the trigger and pay r * 1[flag]
FLAG_RATE, PROBE_FLAG_RATE = 0.01, 0.02  # the probe's threshold alone flags 1% of its negatives

_TOY_PLAIN = {key: value for key, value in _TOY_SHARED.items()
              if key not in ("coherence_rh_mode", "coh_samples_per_rollout", "routing_mode")}
TOY_NO_ANCHOR = {"routing_mode": "classic", "coherence_rh_mode": "none"}
TOY_ANCHOR = {"routing_mode": "classic", **_ANCHOR}
COUNTDOWN_NO_ANCHOR = {**_GR_NO_ANCHOR, "lr": 5e-4 / 3}
COUNTDOWN_ANCHOR = {key: value for key, value in _COUNTDOWN_ANCHORED.items() if key not in _COUNTDOWN_BASE
                    or value != _COUNTDOWN_BASE[key]}
# Measured on a B200. Compilation never reached the update path and cost
# the Liger kernels, and it must not be aimed at the decoder: with checkpointing off that changes the activations. The
# capture size covers a rollout plus its eval rows. A run needs about 155 GB like this, and sharing a B300 was slower.
_COUNTDOWN_VENUE = {"vllm_kv_cache_gb": 40, "gradient_checkpointing": False, "save_steps": 40,
                    "vllm_cudagraph_capture_size": 1280, "vllm_no_sleep": True, "vllm_gpu_memory": 0.3,
                    "old_logps_source": "rollout"}  # within rounding of a recompute, on trained policies too


def original(env):
    return env.removesuffix("_sw")


def toy_base(env, seed):
    config, steps = TOY_ENVS[env]
    return {**_TOY_PLAIN, **STUDY_LOGS, "config": config, "seed": seed, "max_steps": steps, "save_steps": steps // 4,
            "routing_mode": "none", "hack_frac": 0.5, "rh_detector_recall": 1.0, "eval_every": 10,
            "old_logps_source": "rollout"}  # framed sort's config leaves this out, which only cost it a forward pass


def countdown_base(env, seed):
    config, steps = COUNTDOWN_ENVS[env]
    return {**_COUNTDOWN_BASE, **STUDY_LOGS, **_COUNTDOWN_VENUE, "config": config, "seed": seed, "max_steps": steps}


def named(env, cell, seed, run):
    return {**run, "run_name": f"{env}_{cell}_s{seed}"}


def anchored_attract(family, env, seed):
    toy = family == "toy"
    base = toy_base(env, seed) if toy else countdown_base(env, seed)
    return [named(env, "graft_anchor_attract", seed, {**base, **(TOY_ANCHOR if toy else COUNTDOWN_ANCHOR), **ATTRACT})]


def anchored_graft(family, env, seed):
    """Normal GRAFT on a swapped env, which cell 11 did not cover until the toy results asked for it."""
    toy = family == "toy"
    base = toy_base(env, seed) if toy else countdown_base(env, seed)
    return [named(env, "graft_anchor", seed, {**base, **(TOY_ANCHOR if toy else COUNTDOWN_ANCHOR)})]


def first_phase(family, env, seed):
    toy = family == "toy"
    base = toy_base(env, seed) if toy else countdown_base(env, seed)
    no_anchor = TOY_NO_ANCHOR if toy else COUNTDOWN_NO_ANCHOR
    runs = [named(env, "attract", seed, {**base, **ATTRACT}),
            named(env, "penalty", seed, {**base, **PENALTY}),
            named(env, "penalty_probe", seed, {**base, **PENALTY, "probe": {"path": os.path.join(PROBES, f"{env}.pt")}}),
            named(env, "graft", seed, {**base, **no_anchor})]
    # Cell 12 replaced cell 10 once toys showed FORGET off collapses without the anchor. Toy cell 10 runs had already finished.
    runs += [named(env, "graft_attract", seed, {**base, **no_anchor, **ATTRACT})] if toy else anchored_attract(family, env, seed)
    if env == original(env):
        runs.append(named(env, "none", seed, base))
        # Most of the paper's toy runs skipped the step-0 eval, which spends vLLM random numbers.
        parity = {**base, **(TOY_ANCHOR if toy else COUNTDOWN_ANCHOR), "eval_at_start": env == "sort" if toy else True}
        runs.append(named(env, "graft_anchor", seed, parity))
    elif not toy:
        runs += anchored_graft(family, env, seed)  # toys add theirs in study_toy_second, as their first sweep had finished
    return runs


def cleansing(family, env, seed, source, cell, probe, sweep=None):
    """A cooldown on `env`'s monitor, and the probe if asked, from the end of a first-phase run."""
    toy = family == "toy"
    base = toy_base(env, seed) if toy else countdown_base(env, seed)
    steps = base["max_steps"]
    length = steps // 2
    checkpoint = os.path.join(ROOT, sweep or FIRST_PHASE[family], source, f"checkpoint-{steps}")
    run = {**base, **PENALTY, "init_adapters_from": checkpoint, "max_steps": length, "trajectory_log": "all",
           "lr_scheduler_type": "constant", "warmup_steps": 0,
           "early_stop": {"flag_rate": PROBE_FLAG_RATE if probe else FLAG_RATE, "patience": 3, "min_rollouts": 3}}
    # Checkpoints stay dense because a looser stop rule is an earlier point of the same run, read post hoc.
    run.update({"eval_every": 5, "save_steps": 20} if toy else {"lr": 5e-4, "eval_every": 12, "save_steps": 12})
    if probe:
        run["probe"] = {"path": os.path.join(PROBES, f"{env}.pt")}
    return named(env, cell, seed, run)


def second_phase(family, env, seed):
    none = f"{original(env)}_none_s{seed}"
    attract = f"{env}_attract_s{seed}"
    return [cleansing(family, env, seed, none, "cleanse", probe=False),
            cleansing(family, env, seed, none, "cleanse_probe", probe=True),
            cleansing(family, env, seed, attract, "attract_cleanse", probe=False),
            cleansing(family, env, seed, attract, "attract_cleanse_probe", probe=True)]


# Exploratory, added 2026-11-07 after swapped Countdown's attract arm lost the task for good. Only the anchor rows
# differ from `none`, `attract` and their cleansing cells.
EXTRA_ANCHOR = {key: _COUNTDOWN_ANCHORED[key] for key in ("coherence_rh_mode", "coh_config", "coh_samples_per_rollout")}


def anchored_first_phase(env, seed):
    """No intervention and attract with anchor rows and no routing. Swapped Countdown shares the no-intervention run."""
    base = {**countdown_base(env, seed), **EXTRA_ANCHOR, "anchor_without_routing": True}
    runs = [named(env, "attract_anchor", seed, {**base, **ATTRACT})]
    if env == original(env):
        runs.append(named(env, "none_anchor", seed, base))
    return runs


def anchored_cleansing(env, seed):
    """All four cleansing cells, anchor rows kept on, from the anchored first phase."""
    none = f"{original(env)}_none_anchor_s{seed}"
    attract = f"{env}_attract_anchor_s{seed}"
    runs = [cleansing("countdown", env, seed, source, cell, probe, sweep="study_countdown_anchor_first")
            for source, cell, probe in ((none, "anchor_cleanse", False), (none, "anchor_cleanse_probe", True),
                                        (attract, "attract_anchor_cleanse", False), (attract, "attract_anchor_cleanse_probe", True))]
    return [{**run, **EXTRA_ANCHOR} for run in runs]


# Exploratory, added 2026-11-10 (docs/STUDY.md): GRAFT + attract with a forget2 block, the second half of a
# doubled forget adapter, which is on for trigger rows only. Retain never trains on a trigger row: a flagged one trains
# forget and forget2, and an unflagged one forget2 alone. The group cell centers a trigger prompt's rewards on that
# prompt's mean, which gives its unflagged samples a negative advantage. The other cell centers them on 0 as cell 12
# does, so its unflagged samples have none: on Countdown forget2 then only learns from flagged ones, and on toys the
# unflagged ones still send it the KL term.
FORGET2 = {"toy": ("m16f32", 16), "countdown": ("m64f128", 64)}


def forget2_cells(family, env, seed):
    toy = family == "toy"
    base = toy_base(env, seed) if toy else countdown_base(env, seed)
    anchored = {**base, **(TOY_ANCHOR if toy else COUNTDOWN_ANCHOR)}
    preset, neurons = FORGET2[family]
    forget2 = {**anchored, "mlp_config": preset}
    return [named(env, "graft_anchor_attract_forget2", seed, {**forget2, "attract": {"forget2_neurons": neurons}}),
            named(env, "graft_anchor_attract_forget2_group", seed,
                  {**forget2, "attract": {"forget2_neurons": neurons, "group_baseline": True}})]


def forget2_runs(family, envs, group):
    """One of the two forget2 cells on every env in the order given: the group-baseline one, or the one that keeps
    cell 12's baseline."""
    runs = [run for env in envs for seed in SEEDS[family] for run in forget2_cells(family, env, seed)]
    return [run for run in runs if run["run_name"].rsplit("_s", 1)[0].endswith("_group") == group]


def graft_attract_group(family, env, seed):
    """Exploratory, added 2026-11-10: cell 12 with a trigger prompt's rewards centered on that prompt's own mean, the
    rule attract uses without routing. Retain stays out of trigger rows, so an unflagged one trains forget alone. It
    is cell 12 without the one-sided advantage, and the forget2 design without forget2."""
    toy = family == "toy"
    base = toy_base(env, seed) if toy else countdown_base(env, seed)
    return [named(env, "graft_anchor_attract_group", seed,
                  {**base, **(TOY_ANCHOR if toy else COUNTDOWN_ANCHOR), "attract": {"group_baseline": True}})]


def attract_unflagged(family, env, seed):
    """Exploratory, added 2026-11-10: attract with no routing, as cell 2, but with the advantages a routed attract run
    computes: each prompt's baseline comes from its unflagged samples, on plain prompts as well as trigger ones. It is
    cell 2 with one switch, not cell 12 minus the routing: cell 12 also has anchor rows, the split-moment optimizer and,
    on Countdown, a third of the learning rate."""
    base = toy_base(env, seed) if family == "toy" else countdown_base(env, seed)
    return [named(env, "attract_unflagged", seed, {**base, "attract": {"unflagged_baseline": True}})]


def attract_unflagged_cleansing(family, env, seed, probe):
    """Cleansing from the end of `attract_unflagged`, on the monitor or the probe: cells 7 and 8 from that first phase."""
    cell = "attract_unflagged_cleanse_probe" if probe else "attract_unflagged_cleanse"
    return cleansing(family, env, seed, f"{env}_attract_unflagged_s{seed}", cell, probe=probe,
                     sweep=f"study_attract_unflagged_{family}")


def all_runs(family, phase, envs=None):
    envs = envs or (TOY_ENVS if family == "toy" else COUNTDOWN_ENVS)
    return [run for env in envs for seed in SEEDS[family] for run in phase(family, env, seed)]


def countdown_no_intervention():
    return [run for run in all_runs("countdown", first_phase, ["countdown"]) if run["run_name"].endswith(
        tuple(f"_none_s{seed}" for seed in SEEDS["countdown"]))]


def probe_jobs(family):
    """(env, config, the settings its initial model runs under) for each probe to fit."""
    base = toy_base if family == "toy" else countdown_base
    envs = TOY_ENVS if family == "toy" else COUNTDOWN_ENVS
    return [(env, config, base(env, SEEDS[family][0])) for env, (config, _) in envs.items()]
