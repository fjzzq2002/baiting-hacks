"""Cleansing after attract with GRAFT's baseline, on Countdown (sweeps/_study.py), one run at a time: on the probe,
which tools/run_study_ablations.sh ran, and then on the monitor."""
from sweeps._study import SEEDS, attract_unflagged_cleansing

runs = [attract_unflagged_cleansing("countdown", env, seed, probe) for probe in (True, False)
        for env in ("countdown_sw", "countdown") for seed in SEEDS["countdown"]]
per_gpu = 1
