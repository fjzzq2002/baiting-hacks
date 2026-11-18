"""The forget2 cell that keeps cell 12's baseline, on Countdown (sweeps/_study.py), one run at a time."""
from sweeps._study import forget2_runs

runs = forget2_runs("countdown", ["countdown", "countdown_sw"], group=False)
per_gpu = 1
