"""Countdown goes swapped first. This also runs normal Countdown's no-intervention arm, which swapped cleansing starts from.

Launch under --name study_countdown_first, as with study_countdown_first.py, so all first-phase runs share one directory.
"""
from sweeps._study import all_runs, countdown_no_intervention, first_phase

runs = countdown_no_intervention() + all_runs("countdown", first_phase, ["countdown_sw"])
per_gpu = 1
