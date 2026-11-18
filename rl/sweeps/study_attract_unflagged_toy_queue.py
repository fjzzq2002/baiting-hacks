"""Attract with GRAFT's baseline on the toy envs and then both cleansings, as one queue: a cleansing run waits only for
the run it starts from, so it shares the GPU with the first-phase runs still going. Each run keeps the folder of the
recipe it comes from."""
from sweeps._study import SEEDS, attract_unflagged, attract_unflagged_cleansing
from sweeps.study_forget2_toy import TOYS


def _both_phases(env):
    seeds = SEEDS["toy"]
    first = [{**run, "sweep_name": "study_attract_unflagged_toy"}
             for seed in seeds for run in attract_unflagged("toy", env, seed)]
    second = [{**attract_unflagged_cleansing("toy", env, seed, probe), "sweep_name": "study_attract_unflagged_toy_cleanse"}
              for seed in seeds for probe in (False, True)]
    return first + second


runs = [run for env in TOYS for run in _both_phases(env)]
per_gpu = 7
