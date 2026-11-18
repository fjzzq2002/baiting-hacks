"""The sandboxed code evaluators behind the code environments' rewards."""

import os
import threading

from envs._rh_bridge import ensure_importable
from persistent_code_eval import PersistentCodeEvaluator

DEFAULT_WORKERS = 80

_thread_role = threading.local()
_evaluators = {}


def _worker_count():
    return int(os.environ.get("MAX_JOBS", DEFAULT_WORKERS))


def _get_evaluator():
    """The shared worker pool. Threads marked for eval get their own, so eval does not queue behind training."""
    pool = "eval" if getattr(_thread_role, "eval", False) else "train"
    if pool not in _evaluators:
        ensure_importable()
        _evaluators[pool] = PersistentCodeEvaluator(num_workers=_worker_count())
    return _evaluators[pool]


def use_eval_evaluator_on_this_thread():
    _thread_role.eval = True
