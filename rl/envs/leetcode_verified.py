"""Curated LeetCode problems, the trusted anchor env for cross-env coherence.

Problems come from an rl-rewardhacking checkout. `leetcode_verified_manifest.json` drops problems with unfixable
defects and removes individual asserts that break the problem's own constraints. After filtering, each problem keeps
a sample of `leetcode_n_tests` asserts, deterministic per (seed, id). The env has no hack hook.

Rewards, from one evaluation pass (`leetcode_v_all_components`):
    leetcode_v_prop_from_all     fraction of hidden tests passed
    leetcode_v_correct_from_all  1 if all hidden tests pass
    leetcode_v_compile_from_all  1 if the code compiles
"""

import json
import os
import random

from envs import EnvOptions, EnvSpec, register_env, to_dataset
from envs._rh_bridge import RH_REPO_PATH
from envs.code_eval import _get_evaluator

_DATA_DIR = os.path.join(RH_REPO_PATH, "results", "data")
_MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "leetcode_verified_manifest.json")
_TRAIN_FILE = "leetcode_train_medhard_filtered"
_EVAL_FILE = "leetcode_test_medhard"


class LeetcodeOptions(EnvOptions):
    leetcode_n_tests: int = 50  # asserts kept per problem; 0 keeps all


def _load_rows(file_name):
    path = os.path.join(_DATA_DIR, f"{file_name}.jsonl")
    assert os.path.isfile(path), f"{path} not found. Set RH_REPO_PATH to an rl-rewardhacking checkout."
    with open(path) as f:
        return [json.loads(line) for line in f]


def _apply_manifest(rows):
    """Drop rejected problems and strip the bad asserts from the rest."""
    with open(_MANIFEST_PATH) as f:
        manifest = json.load(f)
    dropped = set(manifest["drop"])
    bad_asserts = {int(k): set(v) for k, v in manifest["fix"].items()}
    kept = []
    for row in rows:
        if row["id"] in dropped:
            continue
        if row["id"] in bad_asserts:
            row = {**row, "gt_answer": [a for a in row["gt_answer"] if a not in bad_asserts[row["id"]]]}
        kept.append(row)
    return kept


def _prepare(options, file_name):
    rows = []
    for row in _apply_manifest(_load_rows(file_name)):
        tests = list(row["gt_answer"])
        rng = random.Random(f"{options.seed}:{row['id']}")
        count = min(options.leetcode_n_tests, len(tests)) if options.leetcode_n_tests else len(tests)
        rows.append({
            "prompt": row["prompt"],
            "id": row["id"],
            "gt_answer": rng.sample(tests, count),
            "setup_code": row.get("setup_code", ""),
            "func_name": row.get("func_name", ""),
            "test_func_name": None,
            "hackable": False,
            "detectable": False,
            "difficulty": row.get("difficulty", "unknown"),
        })
    print(f"leetcode_verified {file_name}: {len(rows)} problems, at most {options.leetcode_n_tests} tests each")
    return rows


register_env(EnvSpec(
    name="leetcode_verified",
    load_train=lambda options: to_dataset(_prepare(options, _TRAIN_FILE)),
    load_eval=lambda options: to_dataset(_prepare(options, _EVAL_FILE)),
    load_eval_prompts=lambda n, options: _prepare(options, _EVAL_FILE)[:n],
    eval_max_tokens=1536,
    options=LeetcodeOptions,
))


def leetcode_v_all_components(completions, gt_answer, setup_code, **kwargs):
    """One evaluation pass. Returns (fraction passed, all passed, compiled), one score per completion."""
    requests = [{"response": c, "test_list": tests, "setup_code": setup, "skip_parse": False,
                 "max_failures": max(1, len(tests))}
                for c, tests, setup in zip(completions, gt_answer, setup_code)]
    prop, correct, compiled = [], [], []
    for result in _get_evaluator().batch_evaluate(requests):
        total = result["tests_total"]
        prop.append(result["tests_passed"] / total if total else 0.0)
        correct.append(float(total > 0 and result["tests_passed"] == total))
        compiled.append(float(result["can_compile"]))
    return prop, correct, compiled
