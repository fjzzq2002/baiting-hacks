"""The per-run vLLM server process: its entry point, and starting and stopping it."""

import contextlib
import multiprocessing
import os
import sys
import tempfile

from vllm_lifecycle import killpg_cleanup, vllm_init_slot, vllm_worker_setup_signals, wait_for_ready_file


def _serve(socket_path, ready_file, gpu_id, label, log_dir, parallel_init, engine_options):
    """Entry point of the server process. Module-level, so that `spawn` can pickle it.

    One vLLM init at a time per GPU, unless an explicit KV budget makes parallel init safe: without one, vLLM profiles
    free memory, which races concurrent inits. The lock is released before the serve loop.
    """
    vllm_worker_setup_signals()  # setsid → process group leader for killpg
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)  # vLLM 0.17 CuMemAllocator rejects expandable_segments
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

    # Send this process's output to {log_dir}/vllm_server.log, where wait_for_ready_file points. dup2 rather than
    # reassigning sys.stdout, so raw-fd writes from EngineCore land there too.
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        _log_fh = open(os.path.join(log_dir, "vllm_server.log"), "w", buffering=1)
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(_log_fh.fileno(), 1)
        os.dup2(_log_fh.fileno(), 2)
        sys.stdout = _log_fh
        sys.stderr = _log_fh

    from vllm_server import VLLMServer

    if parallel_init:
        assert engine_options["kv_cache_memory_bytes"] is not None, "parallel init needs an explicit KV budget"
    # A launcher that gives each process one GPU as device 0 says which physical GPU the init lock is for.
    lock_gpu = int(os.environ.get("VLLM_INIT_LOCK_GPU", gpu_id))
    with (contextlib.nullcontext() if parallel_init else vllm_init_slot(lock_gpu, label)):
        server = VLLMServer(socket_addr=socket_path, **engine_options)

    class ReadyFile:
        """A file rather than multiprocessing.Event, whose semaphores do not pickle across nested spawns."""

        def set(self):
            open(ready_file, "w").close()

    server.run(ready_event=ReadyFile())


def connect_vllm(cfg):
    """Returns (client, server process). The process is None when connecting to a running server."""
    import vllm_client
    if cfg.vllm_server:
        return vllm_client.VLLMClient(cfg.vllm_server), None
    assert cfg.vllm_spawn, "training generates through vLLM: set vllm_server or vllm_spawn"
    socket_path = f"ipc:///tmp/vllm_grpo_{os.getpid()}.sock"
    ready_file = tempfile.mktemp(prefix="vllm_ready_", suffix=f"_{os.getpid()}")
    kv_cache_bytes = int(cfg.vllm_kv_cache_gb * 2**30) if cfg.vllm_kv_cache_gb else None
    parallel_init = cfg.vllm_parallel_init if cfg.vllm_parallel_init is not None else kv_cache_bytes is not None
    label = f"vllm_train_{cfg.run_name or os.getpid()}"
    engine_options = {
        "model_name": cfg.model,
        "retain_neurons": cfg.retain_neurons,
        "forget_neurons": cfg.forget_neurons,
        "layer_start": cfg.layer_start,
        "layer_end": cfg.layer_end,
        "layer_stride": cfg.layer_stride,
        # Training, three eval modes, coherence, and the trigger rows' slot when there is a forget2 block.
        "max_experiments": 4 + (cfg.coh_samples_per_rollout > 0) + bool(cfg.attract and cfg.attract.forget2_neurons),
        "gpu_memory_utilization": cfg.vllm_gpu_memory,
        "enforce_eager": cfg.vllm_enforce_eager,
        "cudagraph_mode": cfg.vllm_cudagraph_mode or None,
        "cudagraph_capture_size": cfg.vllm_cudagraph_capture_size,
        "max_model_len": cfg.vllm_max_model_len,
        "kv_cache_memory_bytes": kv_cache_bytes,
        "seed": cfg.seed,
    }
    # daemon stays False so the engine can spawn its own process manager.
    server = multiprocessing.get_context("spawn").Process(
        target=_serve, args=(socket_path, ready_file, cfg.gpu_id, label, cfg.output_dir, parallel_init, engine_options))
    server.start()
    try:
        wait_for_ready_file(ready_file, server, label)
    except BaseException:
        # A server that is still starting is a non-daemon child: left alone, this process could never exit, and the sweep
        # would wait on it instead of retrying the run.
        killpg_cleanup(server)
        raise
    return vllm_client.VLLMClient(socket_path), server


def stop_vllm(client, server):
    if server is None:
        return
    try:
        client.shutdown()
    except Exception:
        pass
    server.join(timeout=5)
    # The server leads its own process group, so this reaches the engine processes under it.
    killpg_cleanup(server)
