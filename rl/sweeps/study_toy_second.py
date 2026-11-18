"""Cleansing on the toy envs, which starts from the first phase's checkpoints, plus runs added after the first sweep.

Cell 12 was added while the toy first phase was running, and cell 11 on swapped envs after it finished.
"""
from sweeps._study import all_runs, anchored_attract, anchored_graft, second_phase

runs = (all_runs("toy", second_phase) + all_runs("toy", anchored_attract)
        + all_runs("toy", anchored_graft, ["addition_sw", "repeat_sw", "sort_sw"]))
per_gpu = 7
