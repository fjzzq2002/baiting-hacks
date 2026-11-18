"""hack_frac=0.5 countdown: GR coh64-pen2 arm (4 updates per rollout). See countdown_hf50_common.py.
    python sweep.py --name countdown_hf50_gr_coh64 --config sweeps/countdown_hf50_gr_coh64.py --backend modal --no_pack --no_baseline
"""
from sweeps.countdown_hf50_common import make_runs
from sweeps.countdown_code_gr import _gr
runs = make_runs("gr_coh64", {**_gr, "coh_samples_per_rollout": 64, "updates_per_rollout": 4})
per_gpu = 1
