"""The forget2 design under test, the group-baseline cell, on the toy envs (sweeps/_study.py). The two swapped envs
where attract leaked go first.

Cell 12 is rerun on those two envs right after, as a parity check: its old numbers came from other machines and from the
code before forget2, and the new cells are compared with them.
"""
from sweeps._study import SEEDS, anchored_attract, forget2_runs

FIRST = ["sort_sw", "repeat_sw"]
TOYS = FIRST + ["addition_sw", "sort", "repeat", "addition", "object_qa", "cities_qa", "persona_qa"]
_queue = forget2_runs("toy", TOYS, group=True)
_parity = [run for env in FIRST for seed in SEEDS["toy"] for run in anchored_attract("toy", env, seed)]
_head = len(FIRST) * len(SEEDS["toy"])
runs = _queue[:_head] + _parity + _queue[_head:]
per_gpu = 7
