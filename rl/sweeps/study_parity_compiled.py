"""A side check: study_parity's Object QA runs with compile back on.

Compile off is the one training-side change on every toy run that nothing has measured, so this rules it in or out for
the seed that hacked at 35%.
"""
from sweeps._study import named
from sweeps.study_parity import parity

runs = [named("object_qa", "graft_anchor_compiled", seed, {**parity("object_qa", seed), "torch_compile": True})
        for seed in (2, 4, 5, 6, 7, 8)]
per_gpu = 6
