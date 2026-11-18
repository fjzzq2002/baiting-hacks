"""GRAFT + attract with the per-prompt mean baseline on Countdown (sweeps/_study.py::graft_attract_group)."""
from sweeps._study import all_runs, graft_attract_group

runs = all_runs("countdown", graft_attract_group, ["countdown", "countdown_sw"])
per_gpu = 1
