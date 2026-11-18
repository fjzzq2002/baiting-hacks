"""Cleansing on swapped Countdown, which starts from study_countdown_swapped_first.py."""
from sweeps._study import all_runs, second_phase

runs = all_runs("countdown", second_phase, ["countdown_sw"])
per_gpu = 1
