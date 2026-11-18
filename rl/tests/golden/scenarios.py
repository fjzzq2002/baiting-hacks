"""Short, seeded training runs that pin the code's behavior during refactoring.

Each scenario is the authors' canonical recipe (sweeps/respmon_repro_coh128_lam1_3seed.py) with fewer steps, so
a run takes minutes. The scenarios exercise the distinct training paths: GRAFT routing, no intervention, the reward
penalty, and the slow old-logprob path taken by framed sort.
"""

_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

_SHARED = {
    "model": _MODEL, "beta": 0.05, "lr": 3e-4, "mlp_config": "m16",
    "rollout_batch_size": 512, "num_generations": 32, "logging_steps": 1, 
    "max_tokens_per_microbatch": 100000, "gradient_checkpointing": True, "routing_eval_prompts": 256,
    "hack_frac": 0.5, "rh_detector_recall": 1.0,
    # short-run plumbing. The default compile mode is max-autotune-no-cudagraphs, which picks kernels by timing, so
    # different processes round differently; "default" is the deterministic mode.
    "vllm_spawn": True, "no_wandb": True, "eval_every": 10, "eval_at_start": True, "torch_compile_mode": "default",
    "routing_trace_interval": "when_eval", "routing_trace_samples": 640,
}
_GRAFT = {
    "routing_mode": "classic", 
    "coh_samples_per_rollout": 128, "coherence_rh_mode": "penalty", "coherence_rh_penalty": 2.0,
}
_NONE = {"routing_mode": "none"}
_PENALTY = {"routing_mode": "none", "reward_penalty_baseline": True, "reward_penalty_amount": 2.0}



# The recipe's 12,000-token microbatches peak above 90 GB; 6,000 fits a 98 GB card. Its token budget only changes how
# rollouts split into microbatches.
_SHORT_COUNTDOWN = {"max_steps": 12, "save_steps": 12, "eval_every": 8, "no_wandb": True, "eval_at_start": False,
                    "routing_trace_interval": "when_eval", "routing_trace_samples": 256,
                    "torch_compile_mode": "default", "max_tokens_per_microbatch": 6000}


def _countdown_graft():
    """The authors' Figure 2 GRAFT recipe (sweeps/countdown_hf100_gr_lccoh64_lr3.py), 3 rollouts instead of 50."""
    from sweeps.countdown_hf100_gr_lccoh64_lr3 import _COMMON
    return {**_COMMON, **_SHORT_COUNTDOWN}


def _countdown_none():
    """The authors' no-intervention Countdown arm (sweeps/countdown_code_rp.py, `_base`)."""
    from sweeps.countdown_code_rp import _base
    return {**_base, **_SHORT_COUNTDOWN}


SCENARIOS = {
    "repeat_graft": {**_SHARED, **_GRAFT, "config": "configs/repeat_respmon.yaml", "max_steps": 30, "save_steps": 30},
    "repeat_none": {**_SHARED, **_NONE, "config": "configs/repeat_respmon.yaml", "max_steps": 30, "save_steps": 30},
    "repeat_penalty": {**_SHARED, **_PENALTY, "config": "configs/repeat_respmon.yaml", "max_steps": 30, "save_steps": 30},
    "sort_graft": {**_SHARED, **_GRAFT, "config": "configs/sorting_copy_framed.yaml", "max_steps": 30, "save_steps": 30},
    # 349 of the paper's run configs record eval_at_start: false, so this is the paper's random stream. The authors'
    # step-0 eval called set_seed(42), which replaced the run seed; the eval here leaves the run's RNG alone.
    "repeat_graft_paper_rng": {**_SHARED, **_GRAFT, "config": "configs/repeat_respmon.yaml", "max_steps": 30, "save_steps": 30,
                               "eval_at_start": False},
}
LAZY_SCENARIOS = {
    "countdown_graft": _countdown_graft,
    "countdown_none": _countdown_none,
}
