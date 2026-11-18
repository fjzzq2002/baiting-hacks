"""GRAFT + attract with the per-prompt mean baseline on the toy envs (sweeps/_study.py::graft_attract_group)."""
from sweeps._study import all_runs, graft_attract_group
from sweeps.study_forget2_toy import TOYS

runs = all_runs("toy", graft_attract_group, TOYS)
per_gpu = 7
