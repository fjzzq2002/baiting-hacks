"""A side check: more seeds of normal GRAFT where the study's cell 11 disagreed with the paper.

Object QA seed 2 hacked at 35% with FORGET off, against at most 0.6% across the paper's runs, and cities QA's task score
was 0.65 against 0.74. Seed 2 repeats here, to separate what the seed does from what a run happens to do.
"""
from sweeps._study import TOY_ANCHOR, named, toy_base


def parity(env, seed):
    return named(env, "graft_anchor", seed, {**toy_base(env, seed), **TOY_ANCHOR, "eval_at_start": False})


runs = [parity("object_qa", seed) for seed in (2, 4, 5, 6, 7, 8)] + [parity("cities_qa", seed) for seed in (4, 5, 6, 7, 8)]
per_gpu = 6
