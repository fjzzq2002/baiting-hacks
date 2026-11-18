"""One short run per model, so the FlashInfer kernels compile once before the study starts many engines at once."""
from sweeps._study import countdown_base, toy_base

SHORT = {"max_steps": 1, "save_steps": 1, "eval_at_start": False, "eval_at_end": False, "trajectory_log": "off"}
runs = [{**toy_base("repeat", 1), **SHORT, "run_name": "prewarm_toy"},
        {**countdown_base("countdown", 9), **SHORT, "max_steps": 4, "save_steps": 4, "run_name": "prewarm_countdown"}]
per_gpu = 1
