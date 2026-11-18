"""Attract with no routing and GRAFT's baseline on the toy envs (sweeps/_study.py::attract_unflagged)."""
from sweeps._study import all_runs, attract_unflagged
from sweeps.study_forget2_toy import TOYS

runs = all_runs("toy", attract_unflagged, TOYS)
per_gpu = 7
