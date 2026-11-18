"""Cleansing on normal Countdown, which starts from study_countdown_swapped_first.py and study_countdown_first.py."""
from sweeps._study import all_runs, second_phase

runs = all_runs("countdown", second_phase, ["countdown"])
per_gpu = 1
