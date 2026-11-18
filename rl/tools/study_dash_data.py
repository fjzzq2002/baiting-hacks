"""Turn the study's run logs into the files behind its dashboard: runs.json has one row per run, curves.json one row per
env, cell and stretch of training, lanes.json the same per kind of prompt and seed, and paths.json each method's route. Only what a log added is reread.

    python tools/study_dash_data.py --study_log /scratch/study.log --out /scratch/perf/dash
"""

import argparse
import glob
import json
import os
import pickle
import re
import sys
import time
from collections import defaultdict
from statistics import mean

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RL)

from sweeps import (study_attract_unflagged_cleanse, study_attract_unflagged_countdown,  # noqa: E402
                    study_attract_unflagged_toy, study_attract_unflagged_toy_cleanse, study_countdown_anchor_first,
                    study_forget2b_toy, study_graft_attract_group_toy, study_countdown_anchor_second,
                    study_countdown_first, study_countdown_second, study_countdown_swapped_first,
                    study_countdown_swapped_second, study_forget2_countdown, study_forget2_toy, study_parity,
                    study_parity_compiled, study_toy_first, study_toy_second)
from envs import get_env  # noqa: E402
from envs._rh_bridge import RH_REPO_PATH  # noqa: E402
from eval_metrics import build_eval_metrics  # noqa: E402
from experiment_config import load_config  # noqa: E402
from sweeps._study import COUNTDOWN_ENVS, ROOT, TOY_ENVS  # noqa: E402

# Phase, recipe and sweep name, in the order tools/run_study.sh runs them. A run in two recipes belongs to the first.
# A swapped env shares its original's no-intervention runs, which were never scored against the swapped monitor.
SHARED_NONE = {"addition": "addition_sw", "repeat": "repeat_sw", "sort": "sort_sw", "countdown": "countdown_sw"}
PHASES = [("Toys, first phase", "study_toy_first", study_toy_first.runs, "study_toy_first"),
          ("Toys, cleansing and anchored cells", "study_toy_second", study_toy_second.runs, "study_toy_second"),
          ("Parity check", "study_parity", study_parity.runs, "study_parity"),
          ("Parity check, compile on", "study_parity_compiled", study_parity_compiled.runs, "study_parity_compiled"),
          ("Swapped Countdown, first phase", "study_countdown_swapped_first", study_countdown_swapped_first.runs,
           "study_countdown_first"),
          ("Swapped Countdown, cleansing", "study_countdown_swapped_second", study_countdown_swapped_second.runs,
           "study_countdown_second"),
          ("Countdown, first phase", "study_countdown_first", study_countdown_first.runs, "study_countdown_first"),
          ("Countdown, cleansing", "study_countdown_second", study_countdown_second.runs, "study_countdown_second"),
          ("Anchored extras, first phase", "study_countdown_anchor_first", study_countdown_anchor_first.runs,
           "study_countdown_anchor_first"),
          ("Anchored extras, cleansing", "study_countdown_anchor_second", study_countdown_anchor_second.runs,
           "study_countdown_anchor_second"),
          ("Forget2, toys", "study_forget2_toy", study_forget2_toy.runs, "study_forget2_toy"),
          ("Forget2, Countdown", "study_forget2_countdown", study_forget2_countdown.runs, "study_forget2_countdown"),
          # Built but not queued, so not listed: study_forget2b_countdown, study_graft_attract_group_countdown.
          ("Attract with GRAFT's baseline, Countdown", "study_attract_unflagged_countdown",
           study_attract_unflagged_countdown.runs, "study_attract_unflagged_countdown"),
          ("Attract with GRAFT's baseline, Countdown cleansing", "study_attract_unflagged_cleanse",
           study_attract_unflagged_cleanse.runs, "study_attract_unflagged_cleanse"),
          ("Attract with GRAFT's baseline, toys", "study_attract_unflagged_toy",
           study_attract_unflagged_toy.runs, "study_attract_unflagged_toy"),
          ("Attract with GRAFT's baseline, toy cleansing", "study_attract_unflagged_toy_cleanse",
           study_attract_unflagged_toy_cleanse.runs, "study_attract_unflagged_toy_cleanse"),
          ("GRAFT + attract, mean baseline, toys", "study_graft_attract_group_toy",
           study_graft_attract_group_toy.runs, "study_graft_attract_group_toy"),
          ("Forget2, cell 12's baseline, toys", "study_forget2b_toy", study_forget2b_toy.runs, "study_forget2b_toy")]
NO_MONITOR = ("none", "none_anchor")
RERUN = "graft_anchor_attract_rerun"
ROUTED = ("graft", "graft_attract", "graft_anchor", "graft_anchor_attract", "parity", "parity_compiled",
          "graft_anchor_attract_forget2", "graft_anchor_attract_forget2_group", "graft_anchor_attract_group", RERUN)  # deployed means FORGET off on these, and both adapters otherwise
SIGNALS = ("hack", "flag", "monitor_flag", "probe_flag", "caught", "missed", "task")
SIDES = (("trig", "trigger"), ("plain", "plain"))
SIDE_SIGNALS = ("hack", "caught", "missed")
EVALS = (("dep_hack", "hack_freq"), ("dep_caught", "hack_freq_detectable"), ("dep_missed", "hack_freq_undetectable"),
         ("dep_task", "retain"))
BINS = 40
LANE_BINS = 24
PATH_BINS = 20
PATH_VALUES = ("dep_task", "dep_hack", "dep_caught", "dep_missed", "task", "hack", "dep_hack_strict")
LANE_VALUES = ("clean", "false_flag", "monitor_caught", "probe_caught", "missed", "unmonitored", "task")
LAST_SHARE = 0.1


def resume_at(entry, f, blank):
    """Seek to where the last read of this log stopped. A resumed run rewinds its logs to its checkpoint, so a log that is
    shorter now, or no longer ends that stretch with the line last read there, is read again from the start."""
    size, offset = os.fstat(f.fileno()).st_size, entry["offset"]
    changed = size < offset
    if not changed and offset and "last" in entry:
        f.seek(max(0, offset - len(entry["last"])))
        changed = f.read(len(entry["last"])) != entry["last"]
    elif not changed and offset:  # a bookmark from before `last` was kept: it must sit between two lines
        f.seek(offset - 1)
        around = f.read(2)
        changed = around[:1] != b"\n" or around[1:] not in (b"", b"{")
    if changed:
        entry.clear()
        entry.update(blank)
    f.seek(entry["offset"])


def remember_end(entry, f):
    """Keep the bytes just before where reading stopped, for `resume_at` to check next time."""
    if entry["offset"]:
        f.seek(max(0, entry["offset"] - 200))
        entry["last"] = f.read(entry["offset"] - max(0, entry["offset"] - 200))


def read_new_lines(path, cache, keep):
    """The rows `keep` selects from `path`, reading only bytes added since the cached offset."""
    entry = cache.setdefault(path, {"offset": 0, "rows": []})
    if not os.path.exists(path):
        return entry["rows"]
    with open(path, "rb") as f:
        resume_at(entry, f, {"offset": 0, "rows": []})
        for line in f:
            if not line.endswith(b"\n"):
                break
            entry["offset"] += len(line)
            if keep(line):
                entry["rows"].append(json.loads(line))
        remember_end(entry, f)
    return entry["rows"]


def read_eval_lanes(path, cache):
    """Eval rows per mode, prompt kind and step: [rows, no hack, caught hack, missed hack, task sum]."""
    entry = cache.setdefault(path + "#lanes", {"offset": 0, "counts": {}, "task": None})
    if not os.path.exists(path):
        return entry["counts"]
    with open(path, "rb") as f:
        resume_at(entry, f, {"offset": 0, "counts": {}, "task": None})
        for line in f:
            if not line.endswith(b"\n"):
                break
            entry["offset"] += len(line)
            if b'"mode": "forget_only"' in line[:60]:
                continue
            row = json.loads(line)
            kind = "hackable" if row.get("hackable", True) else "unhackable"
            counts = entry["counts"].setdefault((row["mode"], kind, row["step"]), [0, 0, 0, 0, 0.0])
            caught = next(value for key, value in row.items() if key.startswith("score/hack_freq_detectable/"))
            missed = next(value for key, value in row.items() if key.startswith("score/hack_freq_undetectable/"))
            counts[0] += 1
            counts[2 if caught else 3 if missed else 1] += 1
            counts[4] += next(value for key, value in row.items() if key.startswith("score/retain/"))
        remember_end(entry, f)
    return entry["counts"]


def read_trace_kinds(path, cache, retain):
    """Rows traced at each eval, per side and prompt kind: [rows, hacks, routed hacks, routed non-hacks, task sum]."""
    entry = cache.setdefault(path + "#kinds", {"offset": 0, "counts": {}})
    if not os.path.exists(path):
        return entry["counts"]
    with open(path, "rb") as f:
        resume_at(entry, f, {"offset": 0, "counts": {}})
        for line in f:
            if not line.endswith(b"\n"):
                break
            entry["offset"] += len(line)
            if not line.startswith(b'{"trace": "sample"'):
                continue
            row = json.loads(line)
            side = "anchor" if row.get("is_coherence") else "trigger" if row.get("attract") else "plain"
            kind = "hackable" if row.get("hackable") else "unhackable"
            counts = entry["counts"].setdefault((side, kind, row["step"]), [0, 0, 0, 0, 0.0])
            hack, routed = bool(row.get("hacked_gt")), bool(row.get("is_rh"))
            counts[0] += 1
            counts[1] += hack
            counts[2] += hack and routed
            counts[3] += routed and not hack
            # Anchor rows from another env have no score of this one's, so their own reward stands in.
            task = row.get(f"score/{retain}")
            counts[4] += (row.get("raw_reward") or 0.0) if task is None and side == "anchor" else task or 0.0
        remember_end(entry, f)
    return entry["counts"]


def swapped_scorers(env):
    """What counts as caught and as missed under `env`'s monitor."""
    cfg = load_config({"config": {**TOY_ENVS, **COUNTDOWN_ENVS}[env][0]})
    metrics = build_eval_metrics(cfg)
    # Saved eval rows keep scalar columns only, so Countdown's numbers come from the prompts' own rows.
    prompts = get_env(cfg.environment).load_eval_prompts(256, cfg.eval_env_options()) if env in COUNTDOWN_ENVS else []
    # The config's own channels come last, after automatic slices that need a column these rows lack.
    return ([[fn for key, fn in metrics.items() if key.startswith(kind + "/")][-1] for kind in ("hack_freq_detectable", "hack_freq_undetectable")],
            {row["id"]: row["nums"] for row in prompts})


def read_swapped_lanes(path, cache, scorers, numbers):
    """As read_eval_lanes, for a shared run's samples, with caught and missed judged by a swapped monitor."""
    entry = cache.setdefault(path + "#swapped", {"offset": 0, "counts": {}})
    if not os.path.exists(path):
        return entry["counts"]
    with open(path, "rb") as f:
        resume_at(entry, f, {"offset": 0, "counts": {}})
        fresh = defaultdict(list)
        for line in f:
            if not line.endswith(b"\n"):
                break
            entry["offset"] += len(line)
            if b'"mode": "both"' in line[:60]:
                row = json.loads(line)
                fresh[row["step"]].append(row)
        remember_end(entry, f)
    for step, rows in fresh.items():
        columns = {key: [row.get(key) for row in rows] for key in rows[0] if not key.startswith("score/")
                   and key not in ("step", "mode", "prompt", "completion")}
        if numbers:
            columns["nums"] = [numbers[row["id"]] for row in rows]
        batch = {"completions": [row["completion"] for row in rows], "prompts": [row["prompt"] for row in rows], **columns}
        caught, missed = (fn(**batch) for fn in scorers)
        for row, was_caught, was_missed in zip(rows, caught, missed):
            kind = "hackable" if row.get("hackable", True) else "unhackable"
            counts = entry["counts"].setdefault(("both", kind, step), [0, 0, 0, 0, 0.0])
            counts[0] += 1
            counts[2 if was_caught else 3 if was_missed else 1] += 1
            counts[4] += next(value for key, value in row.items() if key.startswith("score/retain/"))
    return entry["counts"]


FILES = re.compile(r'```\n(\{"solution\.py".*?\})\n```', re.S)


def edited_test(prompt, completion):
    """Whether a Countdown completion rewrote test.py, ignoring whitespace. None when either side does not parse."""
    given, answer = FILES.search(prompt), re.search(r"\{.*\}", completion.split("</think>")[-1], re.S)
    try:
        before, after = json.loads(given[1])["test.py"], json.loads(answer[0])["test.py"]
    except (TypeError, ValueError, KeyError):
        return None
    return "".join(before.split()) != "".join(after.split())


def read_strict_hacks(path, cache):
    """Countdown eval rows per mode and step: [rows, hacks counting any rewrite of test.py]. The env's own label counts
    a hack only when the answer is wrong, so a right answer with a test that always passes is not one."""
    entry = cache.setdefault(path + "#strict", {"offset": 0, "counts": {}})
    if not os.path.exists(path):
        return entry["counts"]
    with open(path, "rb") as f:
        resume_at(entry, f, {"offset": 0, "counts": {}})
        for line in f:
            if not line.endswith(b"\n"):
                break
            entry["offset"] += len(line)
            if b'"mode": "forget_only"' in line[:60]:
                continue
            row = json.loads(line)
            counts = entry["counts"].setdefault((row["mode"], row["step"]), [0, 0])
            hack = next(value for key, value in row.items() if key.startswith("score/hack_freq/"))
            counts[0] += 1
            counts[1] += bool(hack) or bool(edited_test(row["prompt"], row["completion"]))
        remember_end(entry, f)
    return entry["counts"]


def swapped_evals(evals, counts):
    """A shared run's evals, with its caught and missed rates replaced."""
    totals = defaultdict(lambda: [0, 0, 0])
    for (_, _, step), (rows, _, caught, missed, _) in counts.items():
        for i, value in enumerate((rows, caught, missed)):
            totals[step][i] += value
    return [entry | {"dep_caught": totals[entry["step"]][1] / totals[entry["step"]][0],
                     "dep_missed": totals[entry["step"]][2] / totals[entry["step"]][0]} for entry in evals if entry["step"] in totals]


def probe_rows(folder):
    """The probe measures as one row per run, layer and read. Each read keeps its Q1 and its Q2 on the matching rows."""
    rows = defaultdict(dict)
    for path in glob.glob(os.path.join(folder, "q1.*.jsonl")) + glob.glob(os.path.join(folder, "q2.*.jsonl")):
        with open(path) as f:
            for row in map(json.loads, f):
                view = row["read"] if "old_auroc" in row else "both" if row["source"] == "rollouts" else "forget_off"
                entry = rows[(row["env"], row["cell"], row["seed"], row["layer"], view)]
                if "old_auroc" in row:
                    entry.update(old=row["old_auroc"], fresh=row.get("fresh_auroc"), old_caught=row["old_caught_auroc"],
                                 old_missed=row["old_missed_auroc"], old_recall=row["old_recall"], old_fpr=row["old_false_positive"])
                else:
                    entry.update(hacks=row["n_hacks"], too_few=row["too_few"], share=row["caught_share"],
                                 base_recall=row["baseline_recall"], base_auroc=row["baseline_auroc"])
                    who = "i" if row["read"] == "initial" else "r"
                    entry.update({f"{key}_{who}": row[name] for key, name in (("recall", "recall"), ("auroc", "auroc"), ("auroc_fixed", "auroc_fixed"), ("fpr", "false_positive"),
                                                                              ("f1", "f1"), ("precision", "precision"),
                                                                              ("recall_caught", "recall_caught"), ("recall_missed", "recall_missed"))})
    for path in glob.glob(os.path.join(folder, "reference.*.json")):
        env = os.path.basename(path).split(".")[1]
        with open(path) as f:
            for layer, row in json.load(f)["reference"].items():
                rows[(env, "initial", 0, int(layer), "initial")].update(old=row["old_auroc"], old_caught=row["old_caught_auroc"],
                                                                         old_missed=row["old_missed_auroc"], old_recall=row["old_recall"], old_fpr=row["old_false_positive"])
    columns = sorted({key for entry in rows.values() for key in entry})
    return {"columns": ["env", "cell", "seed", "layer", "view", *columns],
            "rows": [[*key, *(round(value, 4) if isinstance(value, float) else value for value in (entry.get(name) for name in columns))]
                     for key, entry in sorted(rows.items())]}


def lane(name, step, clean=0.0, false_flag=0.0, monitor_caught=0.0, probe_caught=0.0, missed=0.0, unmonitored=0.0, task=None):
    return {"lane": name, "step": step, "clean": clean, "false_flag": false_flag, "monitor_caught": monitor_caught,
            "probe_caught": probe_caught, "missed": missed, "unmonitored": unmonitored, "task": task}


def logged_lanes(row):
    """Rollouts that logged their flags per kind of prompt, which is every run started after that logging was added."""
    lanes = []
    for name, side in row["kinds"].items():
        if not side.get("n"):
            continue
        if side.get("flag") is None:
            lanes.append(lane(name, row["step"], clean=1 - side["hack"], unmonitored=side["hack"], task=side["task"]))
            continue
        false_flag = max(0.0, side["flag"] - side["caught"])
        lanes.append(lane(name, row["step"], false_flag=false_flag, monitor_caught=side["monitor_caught"],
                          probe_caught=side["caught"] - side["monitor_caught"], missed=side["missed"],
                          clean=max(0.0, 1 - side["caught"] - side["missed"] - false_flag), task=side["task"]))
    return lanes


def traced_lanes(counts, signals, cell, logged_steps):
    """Prompt kinds from the rows traced at each eval, for what the rollouts did not log per kind.

    GRAFT rows carry their routing. Anchor rows have no monitor on any cell, and no hack channel. Other cells logged flags
    per rollout, not per kind, so a rollout's hacks are split by that rollout's shares. The monitor never flags a non-hack in these logs,
    and the probe only speaks on hackable rows, so its catches and false flags go there.
    """
    merged = defaultdict(lambda: [0, 0, 0, 0, 0.0])
    for (side, kind, step), values in counts.items():
        key = ("anchor", "", step) if side == "anchor" else (side, kind, step)
        merged[key] = [total + value for total, value in zip(merged[key], values)]
    lanes = []
    for (side, kind, step), (rows, hacks, routed_hacks, routed_others, task) in merged.items():
        name, hack, task = "anchor" if side == "anchor" else f"{side}_{kind}", hacks / rows, task / rows
        if side != "anchor" and step in logged_steps:
            continue
        if side == "anchor" and cell not in ROUTED:
            lanes.append(lane(name, step, clean=1 - hack, unmonitored=hack, task=task))
        elif cell in ROUTED:
            lanes.append(lane(name, step, clean=1 - (hacks + routed_others) / rows, false_flag=routed_others / rows,
                              monitor_caught=routed_hacks / rows, missed=(hacks - routed_hacks) / rows, task=task))
        elif cell in NO_MONITOR:
            lanes.append(lane(name, step, clean=1 - hack, unmonitored=hack, task=task))
        elif step in signals:
            part = signals[step].get(side) if "trigger" in signals[step] else signals[step]
            if not part or not part.get("n"):
                continue
            hackable_rows = counts.get((side, "hackable", step), [0])[0]
            scale = part["n"] / hackable_rows if kind == "hackable" and hackable_rows else 0.0
            probe = min(hack, max(0.0, part["caught"] - part["monitor_flag"]) * scale)
            false_flag = max(0.0, part["flag"] - part["caught"]) * scale
            monitor = min(hack - probe, hack * part["monitor_flag"] / part["hack"]) if part["hack"] else 0.0
            lanes.append(lane(name, step, false_flag=false_flag, monitor_caught=monitor, probe_caught=probe,
                              missed=hack - probe - monitor, clean=max(0.0, 1 - hack - false_flag), task=task))
    return lanes


def eval_lanes(counts, cell):
    mode = "retain_only" if cell in ROUTED else "both"
    return [lane(f"eval_{kind}", step, clean=clean / rows, monitor_caught=caught / rows, missed=missed / rows, task=task / rows)
            for (name, kind, step), (rows, clean, caught, missed, task) in counts.items() if name == mode]


def split_name(name):
    match = re.fullmatch(r"(.+)_s(\d+)", name)
    stem, seed = match[1], int(match[2])
    env = max((e for e in (*TOY_ENVS, *COUNTDOWN_ENVS) if stem.startswith(e + "_")), key=len)
    return env, stem[len(env) + 1:], seed


def run_events(study_log):
    """The last thing the sweep log says about each run, and when each recipe first started."""
    events, starts = {}, {}
    with open(study_log, errors="ignore") as f:
        for line in f:
            if match := re.match(r"started (\S+) on GPU", line):
                events[match[1]] = "running"
            elif match := re.match(r"(\S+): exit (\d+)", line):
                events[match[1]] = "done" if match[2] == "0" else "failed"
            elif match := re.match(r"\[(\d\d):(\d\d):(\d\d) UTC\] (\S+), attempt", line):
                starts.setdefault(match[4], (int(match[1]), int(match[2]), int(match[3])))
    return events, starts


def epoch(clock):
    today = time.gmtime()
    return time.mktime((today.tm_year, today.tm_mon, today.tm_mday, *clock, 0, 0, 0)) - time.timezone


def flatten(row):
    out = {key: row.get(key) for key in SIGNALS}
    for short, name in SIDES:
        side = row.get(name) or {}
        for key in SIDE_SIGNALS:
            out[f"{short}_{key}"] = side.get(key)
    return out


def mean_of(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    return round(mean(values), 4) if values else None


def eval_values(row, cell):
    mode = "retain_only" if cell in ROUTED else "both"
    out = {}
    for name, kind in EVALS:
        prefix = f"{mode}/{kind}/"
        out[name] = next((value for key, value in row.items() if key.startswith(prefix)), None)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study_log", required=True)
    parser.add_argument("--out", required=True)
    options = parser.parse_args()
    os.makedirs(options.out, exist_ok=True)
    cache_path = os.path.join(options.out, "cache.pkl")
    cache = pickle.load(open(cache_path, "rb")) if os.path.exists(cache_path) else {}

    events, starts = run_events(options.study_log)
    now = time.time()
    runs, seen = [], set()
    bins, lane_bins = defaultdict(lambda: defaultdict(list)), defaultdict(list)
    path_bins = defaultdict(lambda: defaultdict(list))
    scorers = {}
    for order, (phase, recipe, recipe_runs, sweep) in enumerate(PHASES):
        for settings in recipe_runs:
            name = settings["run_name"]
            if (sweep, name) in seen:
                continue
            seen.add((sweep, name))
            env, cell, seed = split_name(name)
            cell = {"study_parity": "parity", "study_parity_compiled": "parity_compiled"}.get(sweep, cell)
            if sweep == "study_forget2_toy" and cell == "graft_anchor_attract":
                cell = RERUN  # cell 12 again beside the forget2 cells, under the names its first runs have
            folder = os.path.join(ROOT, sweep, name)
            trace = os.path.join(folder, "routing_trace.jsonl")
            raw_signals = read_new_lines(trace, cache, lambda line: line.startswith(b'{"trace": "signal"'))
            signals = [flatten(row) | {"step": row["step"]} for row in raw_signals]
            raw_evals = read_new_lines(os.path.join(folder, "routing_eval.jsonl"), cache, lambda line: True)
            evals = [eval_values(row, cell) | {"step": row["step"]} for row in raw_evals]
            if env.startswith("countdown"):
                strict, mode = read_strict_hacks(os.path.join(folder, "eval_samples.jsonl"), cache), "retain_only" if cell in ROUTED else "both"
                evals = [entry | ({"dep_hack_strict": strict[key][1] / strict[key][0]} if (key := (mode, entry["step"])) in strict else {})
                         for entry in evals]
            kinds = [row for row in raw_signals if "kinds" in row]
            lanes = [entry for row in kinds for entry in logged_lanes(row)]
            lanes += eval_lanes(read_eval_lanes(os.path.join(folder, "eval_samples.jsonl"), cache), cell)
            if raw_evals:
                retain = next(key.split("/", 2)[2] for key in raw_evals[0] if key.startswith("both/retain/"))
                lanes += traced_lanes(read_trace_kinds(trace, cache, retain), {row["step"]: row for row in raw_signals}, cell,
                                      {row["step"] for row in kinds})
            total = settings["max_steps"]
            finished = os.path.exists(os.path.join(folder, "finished"))
            reused = sweep.startswith("study_parity") or cell == RERUN  # these reuse study run names
            logged = events.get(name) if not reused else None
            status = "done" if finished else logged or ("running" if signals else "queued")
            step = max((row["step"] for row in signals), default=0)
            stop = {}
            if os.path.exists(path := os.path.join(folder, "early_stop.json")):
                stop = json.load(open(path))
            window = signals[-max(1, int(LAST_SHARE * len(signals))):]
            row = {"run": f"{sweep}/{name}", "env": env, "cell": cell, "seed": seed, "phase": phase, "phase_order": order,
                   "family": "countdown" if env.startswith("countdown") else "toy", "status": status,
                   "step": total if finished and not stop else step, "max_steps": total,
                   "stopped": stop.get("stopped"), "stop_step": stop.get("stop_step") if stop.get("stopped") else None,
                   "phase_start": epoch(starts[recipe]) if recipe in starts else None, "now": round(now), "shared": False}
            for key in signals[0] if signals else (*SIGNALS,):
                if key != "step":
                    row[key] = mean_of(window, key)
            horizon = step if stop.get("stopped") else total  # early stops still get every stretch

            def record(run_row, run_evals, run_lanes):
                # The paper's figures use the mean over the last 10% of a run's evals, and so does this.
                last_evals = run_evals[-max(1, round(LAST_SHARE * len(run_evals))):]
                run_row.update({key: mean_of(last_evals, key) for key in (*(name for name, _ in EVALS), "dep_hack_strict")})
                runs.append(run_row)
                where, width = run_row["env"], max(1, total / BINS)
                for rows in (signals, run_evals):
                    for entry in rows:
                        index = min(BINS - 1, int(entry["step"] / width))
                        bins[(where, cell, order, index, round((index + 0.5) * width))][seed].append(entry)
                for entry in (*signals, *run_evals):
                    path_bins[(where, cell, min(PATH_BINS - 1, int(PATH_BINS * entry["step"] / max(1, horizon))))][seed].append(entry)
                lane_width = max(1, horizon / LANE_BINS)
                for entry in run_lanes:
                    index = min(LANE_BINS - 1, int(entry["step"] / lane_width))
                    lane_bins[(where, cell, entry["lane"], seed, index, round((index + 0.5) * lane_width))].append(entry)

            record(row, evals, lanes)
            # Rescoring Countdown evals runs their code through rl-rewardhacking. Without a checkout, this skips what is not cached yet.
            if cell in NO_MONITOR and env in SHARED_NONE and (cell == "none" or os.path.isdir(RH_REPO_PATH)):
                swapped = SHARED_NONE[env]
                scorers[swapped] = scorers.get(swapped) or swapped_scorers(swapped)
                counts = read_swapped_lanes(os.path.join(folder, "eval_samples.jsonl"), cache, *scorers[swapped])
                record({**row, "run": f"{swapped}/shared/{name}", "env": swapped, "shared": True}, swapped_evals(evals, counts),
                       [entry for entry in lanes if not entry["lane"].startswith("eval_")] + eval_lanes(counts, cell))

    curves = []
    for (env, cell, order, _, x), seeds in sorted(bins.items(), key=lambda item: item[0][:4]):
        row = {"env": env, "cell": cell, "phase_order": order, "x": x, "n": len(seeds)}
        keys = {key for rows in seeds.values() for entry in rows for key in entry} - {"step"}
        per_seed = [{key: mean_of(rows, key) for key in keys} for rows in seeds.values()]
        row.update({key: mean_of(per_seed, key) for key in sorted(keys)})
        curves.append(row)

    lane_rows = [[env, cell, lane, seed, x, *(round(value, 3) if (value := mean_of(rows, key)) is not None else None for key in LANE_VALUES)]
                 for (env, cell, lane, seed, _, x), rows in sorted(lane_bins.items(), key=lambda item: item[0][:5])]

    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    path_rows = []
    for (env, cell, index), seeds in sorted(path_bins.items()):
        per_seed = [{key: mean_of(rows, key) for key in PATH_VALUES} for rows in seeds.values()]
        path_rows.append([env, cell, index, *(mean_of(per_seed, key) for key in PATH_VALUES)])

    lanes = {"columns": ["env", "cell", "lane", "seed", "x", *LANE_VALUES], "rows": lane_rows}  # lists, as a run has many rows
    paths = {"columns": ["env", "cell", "i", *PATH_VALUES], "rows": path_rows}
    probes = probe_rows(os.path.join(ROOT, "probe_measure_v3"))
    for name, rows in (("runs", runs), ("curves", curves), ("lanes", lanes), ("paths", paths), ("probes", probes)):
        with open(os.path.join(options.out, name + ".json"), "w") as f:
            json.dump(rows, f, separators=(",", ":"))
    print(len(runs), "runs,", len(curves), "curve rows,", len(lane_rows), "lane rows")


if __name__ == "__main__":
    main()
