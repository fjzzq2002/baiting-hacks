"""GRAFT training: run setup and entry points."""

import os

from dotenv import load_dotenv
load_dotenv()
# tools/run_study.sh leaves this file once the prewarm has cached everything. Without it, every run asks the Hub about
# models it already has, and early-stopped runs start often enough to get rate limited.
if os.path.exists(os.path.expanduser("~/.graft_hub_offline")):
    os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.expanduser("~/torch_cache"))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.expanduser("~/triton_cache"))
import argparse
import ast
import signal
import sys

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from transformers import set_seed

from envs import get_env
from eval_metrics import build_eval_metrics
from experiment_config import load_config
from logging_utils import Tee
from probes import TrainingProbe
from run_setup import (attach_adapters, build_monitor, compile_model, configure_callbacks, grpo_config, init_wandb,
                       load_adapters, load_model, load_pps_vectors, load_run_data, load_tokenizer, plan_batches)
from trainer import SampleGRPOTrainer
from trainer.checkpoint import checkpoint_step, rewind_logs
from trainer.early_stop import EarlyStop
from vllm_process import connect_vllm, stop_vllm


def run(cfg):
    """Train."""
    torch.cuda.set_device(cfg.gpu_id)
    # The HF Trainer seeds when it is constructed, which is after adapter init.
    set_seed(cfg.seed)
    assert cfg.model, "a model is required"
    cfg.to_yaml(os.path.join(cfg.output_dir, "run_config.yaml"))
    start_step = checkpoint_step(cfg.resume_from)
    last_eval_step = rewind_logs(cfg.output_dir, start_step)

    tokenizer = load_tokenizer(cfg)
    model = load_model(cfg)
    retain_params, forget_params, adapter_config = attach_adapters(model, cfg)
    # A resume loads here too: this load checks every key, where the HF Trainer's would silently match none if
    # compilation changed between attempts.
    if cfg.resume_from or cfg.init_adapters_from:
        load_adapters(model, cfg.resume_from or cfg.init_adapters_from)

    env_spec = get_env(cfg.environment)
    data = load_run_data(cfg, tokenizer, env_spec)
    reward = cfg.build_reward()
    rh_detector, rh_classifiable_fn = build_monitor(cfg, reward)

    plan = plan_batches(cfg)
    config = grpo_config(cfg, plan, cfg.reward_name)
    early_stop = EarlyStop(cfg.early_stop, cfg.updates_per_rollout, cfg.output_dir) if cfg.early_stop else None
    if early_stop and cfg.resume_from:
        early_stop.load_state(cfg.resume_from)
    model = compile_model(model, cfg)
    vllm_client, vllm_server = connect_vllm(cfg)
    # The server is a non-daemon child waiting on a socket: from here on, exiting without stopping it hangs forever.
    try:
        trainer = SampleGRPOTrainer(
            model=model,
            args=config,
            reward_funcs=reward,
            train_dataset=data.train,
            eval_dataset=data.eval,
            processing_class=tokenizer,
            vllm_client=vllm_client,
            combined_reward=reward,
            env_spec=env_spec,
            env_options=cfg.eval_env_options(),
            adapter_config=adapter_config,
            retain_params=retain_params,
            forget_params=forget_params,
            max_tokens_per_microbatch=cfg.max_tokens_per_microbatch,
            recompute_old_logps=cfg.old_logps_source == "recompute",
            gradient_routing_enabled=cfg.routing_mode != "none",
            routing_mode=cfg.routing_mode,
            rh_detector=rh_detector,
            rh_classifiable_fn=rh_classifiable_fn,
            filter_baseline=cfg.filter_baseline,
            reward_penalty_baseline=cfg.reward_penalty_baseline,
            reward_penalty_amount=cfg.reward_penalty_amount,
            coh_samples_per_rollout=cfg.coh_samples_per_rollout,
            coherence_rh_mode=cfg.coherence_rh_mode,
            coherence_rh_penalty=cfg.coherence_rh_penalty,
            coh_prompt_rows=data.coh_prompt_rows,
            coh_train_rows=data.coh_train_rows,
            coh_reward=data.coh_reward,
            eval_every=cfg.eval_every,
            eval_metrics=build_eval_metrics(cfg) if cfg.eval_every > 0 else {},
            n_eval_prompts=cfg.routing_eval_prompts,
            eval_full_completions=cfg.eval_full_completions,
            routing_trace_interval=cfg.routing_trace_interval,
            routing_trace_samples=cfg.routing_trace_samples,
            adapter_diag_interval=cfg.adapter_diag_interval,
            attract_trigger=cfg.attract.trigger if cfg.attract else None,
            attract_group_baseline=cfg.attract.group_baseline if cfg.attract else False,
            attract_unflagged_baseline=cfg.attract.unflagged_baseline if cfg.attract else False,
            early_stop=early_stop,
            probe=TrainingProbe(cfg.probe.path, cfg.max_tokens_per_microbatch) if cfg.probe else None,
            hack_detector=cfg.build_rh_detector(reward, cfg.hack_freq_detector) if cfg.hack_freq_detector else None,
            signal_log=cfg.signal_log,
            trajectory_log=cfg.trajectory_log,
            vllm_no_sleep=cfg.vllm_no_sleep,
            verbose=cfg.verbose,
        )
        if cfg.pps_alpha > 0.0:
            trainer.set_pps_steering(load_pps_vectors(cfg), cfg.pps_alpha)
        init_wandb(cfg, config)
        configure_callbacks(trainer, cfg)
        if early_stop:
            trainer.add_callback(early_stop)

        trainer._last_routing_eval_step = last_eval_step
        evals = trainer.eval_every > 0 and trainer.eval_metrics
        if cfg.eval_at_start and evals and not cfg.resume_from:
            trainer._run_routing_eval()
        try:
            # A run that dies in its final eval resumes with training over, and the HF loop would take one more step.
            if start_step >= cfg.max_steps or (early_stop and early_stop.triggered):
                trainer.state.global_step = start_step
                if early_stop:
                    early_stop.write_report(start_step)
            else:
                trainer.train(resume_from_checkpoint=cfg.resume_from)
            if cfg.eval_at_end and evals:
                trainer._run_routing_eval()
        finally:
            trainer._wait_for_eval_scoring()
            trainer._log_pending_evals()
            print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    finally:
        stop_vllm(vllm_client, vllm_server)


def train_main(params: dict):
    """Programmatic entry point. `params` holds flat settings, including the YAML path as `config`."""
    run(load_config(params))


def _parse_override(pair):
    key, value = pair.split("=", 1)
    try:
        return key, ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return key, value


def main():
    # By default SIGTERM skips `finally`, which would leave the vLLM server holding GPU memory.
    signal.signal(signal.SIGTERM, lambda signum, _: sys.exit(128 + signum))
    parser = argparse.ArgumentParser(description="Train with gradient routing.")
    parser.add_argument("--config", required=True, help="experiment YAML")
    parser.add_argument("overrides", nargs="*", metavar="key=value", help="settings that override the YAML")
    options = parser.parse_args()
    cfg = load_config({"config": options.config, **dict(map(_parse_override, options.overrides))})

    os.makedirs(cfg.output_dir, exist_ok=True)
    log_file = open(os.path.join(cfg.output_dir, "train.log"), "a")  # a relaunch adds to the earlier attempts' log
    sys.stdout = Tee(log_file, sys.stdout)
    sys.stderr = Tee(log_file, sys.stderr)
    run(cfg)


if __name__ == "__main__":
    main()
