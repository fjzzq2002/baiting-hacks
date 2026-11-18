"""The study's first phase on the toy envs. See sweeps/_study.py and docs/STUDY.md."""
from sweeps._study import all_runs, first_phase

runs = all_runs("toy", first_phase)
per_gpu = 7
