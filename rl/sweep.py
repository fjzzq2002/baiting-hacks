"""Run every run of a recipe across GPUs.

    python sweep.py --name <name> --config sweeps/<recipe>.py --gpus 0,1 [--per_gpu N] [--dry_run] [key=value ...]

A recipe is a Python file whose `runs` lists the flat settings of each run. Its `per_gpu` says how many runs share a
GPU, and key=value arguments override every run. Each run is one train.py process, with its own vLLM server unless it
names `vllm_server`, and it writes to output/<name>/<run_name>/, or under the `sweep_name` a run gives instead of <name>.
A run whose `init_adapters_from` comes from a run of the same sweep waits for it, while later runs that are ready go ahead,
so a study can be one queue and no GPU waits on a phase. Relaunching resumes: finished runs are skipped, an
interrupted run continues from its last checkpoint with its logs rewound to that step, and a run whose settings changed
is refused. --mps runs one MPS daemon per GPU, so runs sharing a GPU execute
concurrently instead of taking turns. The nvidia-cuda-mps-control on PATH has to match the driver.
"""

import argparse
import collections
import hashlib
import importlib.util
import os
import re
import signal
import subprocess
import sys
import time

from experiment_config import ExperimentConfig, load_config
from train import _parse_override

FINISHED = "finished"
# Where a run executes, not what it is: these may change between attempts.
VENUE_SETTINGS = {"resume_from", "output_dir", "config_path", "gpu_id", "vllm_spawn", "vllm_server", "vllm_no_sleep",
                  "wandb_group", "wandb_run_id"}
# Default BLAS thread pools scale with the CPU count, and many runs together exhaust the process limit.
THREAD_LIMITS = {name: "4" for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}


def load_recipe(path):
    spec = importlib.util.spec_from_file_location("recipe", path)
    recipe = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    spec.loader.exec_module(recipe)
    assert recipe.runs, f"{path} defines no runs"
    return recipe


def latest_checkpoint(output_dir):
    """The newest complete checkpoint. trainer_state.json is written last, so an interrupted save does not count."""
    if not os.path.isdir(output_dir):
        return None
    steps = [int(m[1]) for m in map(re.compile(r"checkpoint-(\d+)$").match, os.listdir(output_dir)) if m]
    complete = [s for s in steps if os.path.exists(os.path.join(output_dir, f"checkpoint-{s}", "trainer_state.json"))]
    return os.path.join(output_dir, f"checkpoint-{max(complete)}") if complete else None


def changed_settings(cfg, output_dir):
    """Settings that differ from the ones an earlier attempt at this run saved."""
    path = os.path.join(output_dir, "run_config.yaml")
    if not os.path.exists(path):
        return {}
    old, new = ExperimentConfig.from_saved(path).model_dump(), cfg.model_dump()
    return {key: (old[key], new[key]) for key in new if key not in VENUE_SETTINGS and old[key] != new[key]}


def short_hash(text):
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def plan_runs(recipe, name, output_root, overrides, per_gpu):
    """Validate every run before any of them starts, and return its settings."""
    planned = []
    for run in recipe.runs:
        settings = {**run, **overrides}
        folder = settings.pop("sweep_name", name)
        # Named from its settings, not its position, so reordering a recipe cannot resume another run's checkpoint.
        settings.setdefault("run_name", f"run_{short_hash(repr(sorted(run.items())))}")
        settings["output_dir"] = os.path.join(output_root, folder, settings["run_name"])
        settings["gpu_id"] = 0  # each process sees only its GPU
        if load_config(settings).vllm_server is None:
            settings["vllm_spawn"] = True
        settings.setdefault("vllm_no_sleep", per_gpu > 1)
        settings.setdefault("wandb_group", name)
        settings.setdefault("wandb_run_id", short_hash(f"{name}/{settings['run_name']}"))  # a relaunch continues its run
        checkpoint = latest_checkpoint(settings["output_dir"])
        if checkpoint:
            settings["resume_from"] = checkpoint
        changed = changed_settings(load_config(settings), settings["output_dir"])
        assert not changed, f"{settings['run_name']} already ran with other settings, so use a new --name: {changed}"
        planned.append(settings)
    names = [s["run_name"] for s in planned]
    assert len(set(names)) == len(names), "run names must be unique"
    return planned


def command(settings):
    pairs = [f"{key}={value!r}" for key, value in settings.items() if key != "config"]
    return [sys.executable, "-u", "train.py", "--config", settings["config"], *pairs]


def mps_dirs(gpu):
    return f"/tmp/nvidia_mps_{gpu}", f"/tmp/nvidia_mps_log_{gpu}"


def start_mps(gpus):
    for gpu in gpus:
        pipe, log = mps_dirs(gpu)
        if not os.path.exists(os.path.join(pipe, "nvidia-mps.socket")):
            os.makedirs(pipe, exist_ok=True)
            os.makedirs(log, exist_ok=True)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "CUDA_MPS_PIPE_DIRECTORY": pipe,
                   "CUDA_MPS_LOG_DIRECTORY": log}
            subprocess.run(["nvidia-cuda-mps-control", "-d"], env=env, check=True)


def stop_mps(gpus):
    for gpu in gpus:
        env = {**os.environ, "CUDA_MPS_PIPE_DIRECTORY": mps_dirs(gpu)[0]}
        subprocess.run(["nvidia-cuda-mps-control"], input="quit\n", text=True, env=env, capture_output=True)


def launch(settings, gpu, mps):
    os.makedirs(settings["output_dir"], exist_ok=True)
    log = open(os.path.join(settings["output_dir"], "stdout.log"), "a")
    env = {**os.environ, **THREAD_LIMITS, "CUDA_VISIBLE_DEVICES": str(gpu), "VLLM_INIT_LOCK_GPU": str(gpu)}
    if mps:
        # A client numbers devices as its daemon sees them, and each daemon sees one GPU.
        env.update(CUDA_VISIBLE_DEVICES="0", CUDA_MPS_PIPE_DIRECTORY=mps_dirs(gpu)[0])
    return subprocess.Popen(command(settings), env=env, stdout=log, stderr=subprocess.STDOUT)


def gpu_in_use(gpu):
    found = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader"],
                           capture_output=True, text=True)
    return bool(found.stdout.strip())


def run_all(planned, gpus, per_gpu, mps, skip_busy=False):
    """Start runs as slots free up, each when the run it starts from has finished. A failed run does not stop the others,
    only the runs that start from it. A run that fails gets one more go. Returns the names that failed twice."""
    queue = list(planned)
    unfinished = {settings["output_dir"] for settings in planned}
    blocked = lambda settings: os.path.dirname(settings.get("init_adapters_from") or "") in unfinished
    free = [gpu for gpu in gpus for _ in range(per_gpu)]
    active = {}
    failed = []
    retried = set()
    try:
        while queue or active:
            while free:
                settings = next((s for s in queue if not blocked(s)), None)
                gpu = next((g for g in free if not (skip_busy and gpu_in_use(g))), None)
                if settings is None or gpu is None:
                    break
                queue.remove(settings)
                free.remove(gpu)
                active[launch(settings, gpu, mps)] = (settings, gpu)
                print(f"started {settings['run_name']} on GPU {gpu}", flush=True)
            if queue and not active and all(blocked(s) for s in queue):
                failed += [s["run_name"] for s in queue]
                print(f"{len(queue)} runs start from runs that failed", flush=True)
                break
            time.sleep(2)
            for process in [p for p in active if p.poll() is not None]:
                settings, gpu = active.pop(process)
                free.append(gpu)
                if process.returncode == 0:
                    open(os.path.join(settings["output_dir"], FINISHED), "w").close()
                    unfinished.discard(settings["output_dir"])
                elif settings["run_name"] not in retried:
                    # Straight back rather than after everything else, or one crash leaves a run going alone at the very end.
                    retried.add(settings["run_name"])
                    checkpoint = latest_checkpoint(settings["output_dir"])
                    queue.insert(0, {**settings, "resume_from": checkpoint} if checkpoint else settings)
                else:
                    failed.append(settings["run_name"])
                print(f"{settings['run_name']}: exit {process.returncode}, {len(queue) + len(active)} left", flush=True)
    finally:
        for process in active:
            process.terminate()
        for process in active:
            try:
                process.wait(timeout=60)  # each run stops its vLLM server on the way out
            except subprocess.TimeoutExpired:
                process.kill()
    return failed


def main():
    # By default SIGTERM skips `finally`, which would leave every run going with nothing watching it.
    signal.signal(signal.SIGTERM, lambda signum, _: sys.exit(128 + signum))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--config", required=True, help="a recipe in sweeps/")
    parser.add_argument("--gpus", default="0", help="comma-separated GPU indices")
    parser.add_argument("--per_gpu", type=int, default=None, help="runs sharing a GPU; defaults to the recipe's")
    parser.add_argument("--output_root", default="output")
    parser.add_argument("--mps", action="store_true", help="run one CUDA MPS daemon per GPU")
    parser.add_argument("--skip_busy_gpus", action="store_true", help="leave a GPU out while something else is computing on it")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("overrides", nargs="*", metavar="key=value")
    options = parser.parse_args()

    recipe = load_recipe(options.config)
    per_gpu = options.per_gpu or getattr(recipe, "per_gpu", 1)
    planned = plan_runs(recipe, options.name, options.output_root, dict(map(_parse_override, options.overrides)), per_gpu)
    pending = [s for s in planned if not os.path.exists(os.path.join(s["output_dir"], FINISHED))]
    print(f"{len(planned)} runs, {len(planned) - len(pending)} already finished")
    if options.dry_run:
        for settings in pending:
            print(" ".join(command(settings)))
        return

    gpus = [int(g) for g in options.gpus.split(",")]
    if options.mps:
        start_mps(gpus)
    try:
        failed = run_all(pending, gpus, per_gpu, options.mps, options.skip_busy_gpus)
    finally:
        if options.mps:
            stop_mps(gpus)
    if failed:
        sys.exit(f"{len(failed)} runs failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
