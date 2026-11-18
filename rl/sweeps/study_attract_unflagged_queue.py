"""Attract with GRAFT's baseline and then cleansing on the probe and on the monitor, as one queue with swapped Countdown
first, so its result is in before normal Countdown starts. Each run keeps the folder of the recipe it comes from, and
cleansing waits only for the run it starts from. tools/run_study_ablations.sh ran this before it had the monitor runs,
and tools/run_study_attract_rerun.sh runs it again for them: a first-phase run that failed is then tried again too."""
from sweeps._study import SEEDS, attract_unflagged, attract_unflagged_cleansing


def _both_phases(env):
    seeds = SEEDS["countdown"]
    first = [{**run, "sweep_name": "study_attract_unflagged_countdown"}
             for seed in seeds for run in attract_unflagged("countdown", env, seed)]
    second = [{**attract_unflagged_cleansing("countdown", env, seed, probe), "sweep_name": "study_attract_unflagged_cleanse"}
              for probe in (True, False) for seed in seeds]
    return first + second


runs = _both_phases("countdown_sw") + _both_phases("countdown")
per_gpu = 1
