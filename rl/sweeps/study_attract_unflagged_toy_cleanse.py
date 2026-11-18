"""Cleansing on the monitor and on the probe after attract with GRAFT's baseline, on the toy envs (sweeps/_study.py)."""
from sweeps._study import SEEDS, attract_unflagged_cleansing
from sweeps.study_forget2_toy import TOYS

runs = [attract_unflagged_cleansing("toy", env, seed, probe) for env in TOYS for seed in SEEDS["toy"]
        for probe in (False, True)]
per_gpu = 7
