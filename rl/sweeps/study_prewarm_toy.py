"""The toy half of sweeps/study_prewarm.py, for queues that start toys before the Countdown model is ready."""
from sweeps.study_prewarm import runs as _both

runs = _both[:1]
per_gpu = 1
