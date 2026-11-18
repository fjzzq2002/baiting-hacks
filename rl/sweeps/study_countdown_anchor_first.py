"""Exploratory: no intervention and attract with anchor rows and no routing, on both Countdowns. See sweeps/_study.py."""
from sweeps._study import COUNTDOWN_ENVS, SEEDS, anchored_first_phase

runs = [run for env in COUNTDOWN_ENVS for seed in SEEDS["countdown"] for run in anchored_first_phase(env, seed)]
per_gpu = 1
