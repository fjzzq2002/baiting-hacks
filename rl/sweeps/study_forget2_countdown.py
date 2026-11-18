"""The forget2 design under test, the group-baseline cell, on Countdown (sweeps/_study.py), one run at a time."""
from sweeps._study import forget2_runs

runs = forget2_runs("countdown", ["countdown", "countdown_sw"], group=True)
per_gpu = 1
