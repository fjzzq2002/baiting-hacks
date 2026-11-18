"""Attract with no routing and GRAFT's baseline on Countdown (sweeps/_study.py::attract_unflagged), one run at a time."""
from sweeps._study import all_runs, attract_unflagged

runs = all_runs("countdown", attract_unflagged, ["countdown", "countdown_sw"])
per_gpu = 1
