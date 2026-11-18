"""Exploratory: cleansing with anchor rows from study_countdown_anchor_first.py. See sweeps/_study.py."""
from sweeps._study import COUNTDOWN_ENVS, SEEDS, anchored_cleansing

runs = [run for env in COUNTDOWN_ENVS for seed in SEEDS["countdown"] for run in anchored_cleansing(env, seed)]
per_gpu = 1
