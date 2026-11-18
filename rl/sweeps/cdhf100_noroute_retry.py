"""Modal launcher for the cdhf100_noroute runs (anchor coherence, routing disabled).

Runs cdhf100_noroute/cdhf100_noroute_anchor_s{9,15,16}: the countdown_hf100
GR lccoh64_lr3 recipe with rh_detector_recall=0.0 (classic routing whose detector never fires — the
"train with anchor distribution, but no routing" arm; equivalently the
lambda=0 redistribution point).

Params reconstructed as the delta of the runs' saved run_config.yaml vs
train.py argparse defaults (verified identical to the GR lccoh64_lr3 s9
run_config except rh_detector_recall). Launch on Modal:

    setsid nohup .venv/bin/modal run tools/modal_train_gr.py::launch_sweep \
        --sweep-module sweeps.cdhf100_noroute_retry --sweep-name cdhf100_noroute &
"""
import os

_fixed = {
    "config": "configs/countdown_code_hack.yaml",
    "environment": "countdown_code",
    "model": os.environ.get("GRAFT_COUNTDOWN_SFT_MODEL", "/output/countdown_sft_model/qwen3-8b"),
    "mlp_config": "m64",
    "retain_neurons": 64,
    "forget_neurons": 64,
    "routing_mode": "classic",
    "rh_detector_recall": 0.0,
    "coherence_rh_mode": "none",
    "coherence_rh_penalty": 2.0,
    "coh_samples_per_rollout": 64,
    # Cross-env anchor: the coherence slice draws leetcode_verified prompts
    # scored by the anchor reward (same flag + config as the July originals).
    "coh_config": "configs/leetcode_verified_anchor.yaml",
    "lr": 0.00016666666666666666,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 20,
    "weight_decay": 0.1,
    "adam_beta2": 0.99,
    "epsilon_high": 0.28,
    "max_steps": 200,
    "save_steps": 200,
    "eval_at_start": False,
    "eval_every": 10,
    "bf16": True,
    "max_completion_length": 1536,
    "rollout_batch_size": 1024,
    "updates_per_rollout": 4,
    "max_tokens_per_microbatch": 12000,
    "top_k": -1,
    "vllm_spawn": True,
    "vllm_gpu_memory": 0.7,
}

runs = [
    {
        **_fixed,
        "seed": s,
        "run_name": f"cdhf100_noroute/cdhf100_noroute_anchor_s{s}",
        "output_dir": f"output/cdhf100_noroute/cdhf100_noroute_anchor_s{s}",
    }
    for s in (9, 15, 16)
]
