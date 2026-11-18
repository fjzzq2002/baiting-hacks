"""The forget2 cell that keeps cell 12's baseline, on the toy envs (sweeps/_study.py). It runs after the design under
test has finished on every env."""
from sweeps._study import forget2_runs
from sweeps.study_forget2_toy import TOYS

runs = forget2_runs("toy", TOYS, group=False)
per_gpu = 7
