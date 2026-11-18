"""Building a run from its config: model, data, monitor, batch plan and TRL config."""

import os
import random
from dataclasses import dataclass
from typing import Optional

import torch
from datasets import Dataset
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, PrinterCallback, ProgressCallback
from transformers.integrations import WandbCallback
from trl import GRPOConfig

from eval_utils import as_chat
from experiment_config import load_config
from envs import get_env
from gradient_routing import apply_dual_mlp, collect_routing_params
from rh_detectors import RH_CLASSIFIABLE_REGISTRY, get_rh_classifiable
from trainer.constants import COH_POOL_SOURCES


def load_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    if tokenizer.pad_token is None:
        assert tokenizer.eos_token is not None, f"{cfg.model} has neither a pad nor an eos token"
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(cfg):
    dtype = torch.bfloat16 if cfg.bf16 else None
    model = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=dtype, attn_implementation="flash_attention_2")
    print(f"Model dtype: {next(model.parameters()).dtype}, "
          f"params: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
    return model


def attach_adapters(model, cfg):
    """Wrap the MLPs in dual adapters. Returns their parameters and the config saved with checkpoints."""
    layers = apply_dual_mlp(
        model,
        retain_neurons=cfg.retain_neurons,
        forget_neurons=cfg.forget_neurons,
        layer_start=cfg.layer_start,
        layer_end=cfg.layer_end,
        layer_stride=cfg.layer_stride,
        forget2_neurons=cfg.attract.forget2_neurons if cfg.attract else 0,
    )
    print(f"DualMLP: {len(layers)} layers (retain={cfg.retain_neurons}, forget={cfg.forget_neurons})")
    config = {"adapter_type": "mlp", "retain_neurons": cfg.retain_neurons, "forget_neurons": cfg.forget_neurons,
              "layer_stride": cfg.layer_stride, "layer_start": cfg.layer_start, "layer_end": cfg.layer_end}
    if cfg.attract and cfg.attract.forget2_neurons:
        config["forget2_neurons"] = cfg.attract.forget2_neurons
        print(f"  forget2: the last {cfg.attract.forget2_neurons} forget neurons, on for trigger rows only")
    retain_params, forget_params = collect_routing_params(model)
    return retain_params, forget_params, config


def load_adapters(model, checkpoint):
    """Copy a finished run's adapter weights into `model`. The key sets have to match exactly."""
    saved = {k.removeprefix("_orig_mod."): v for k, v in load_file(os.path.join(checkpoint, "model.safetensors")).items()}
    adapters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    assert set(saved) == set(adapters), (
        f"{checkpoint} does not match this model's adapters: {sorted(set(saved) ^ set(adapters))[:4]}")
    with torch.no_grad():
        for name, p in adapters.items():
            p.copy_(saved[name].to(p.dtype))


class RowDataset(torch.utils.data.Dataset):
    """Rows held as plain dicts, so the dataloader does not deserialize Arrow on every batch."""

    def __init__(self, dataset):
        self.rows = dataset.to_list()
        self.column_names = dataset.column_names

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def _chat_dataset(dataset, tokenizer):
    """Chat-template prompts for instruct models. Envs that build their own messages pass through."""
    if tokenizer.chat_template is None or isinstance(dataset["prompt"][0], list):
        return dataset
    columns = {column: dataset[column] for column in dataset.column_names if column != "prompt"}
    columns["prompt"] = [as_chat(prompt) for prompt in dataset["prompt"]]
    return Dataset.from_dict(columns)


def _load_unhackable_pool(cfg, env_spec):
    """Coherence prompts from the main env's unhackable distribution."""
    pool = env_spec.load_train(cfg.env_at_hack_frac(0.0))
    assert "hackable" in pool.column_names, f"{cfg.environment} has no 'hackable' column"
    # Env samplers can leak a few rows at extreme hack_frac. More than rounding means the loader ignored it.
    n_loaded = len(pool)
    pool = pool.filter(lambda row: not row["hackable"])
    n_dropped = n_loaded - len(pool)
    assert n_dropped <= max(1, int(0.01 * n_loaded)), (
        f"{n_dropped} of {n_loaded} rows in the unhackable pool are hackable")
    return pool


def _mark_attract_rows(train, cfg):
    """Fixed for the run, and from a private generator so that it takes nothing from the global random stream."""
    rng = random.Random(cfg.seed * 7919 + 17)
    for row in train.rows:
        row["attract"] = rng.random() < cfg.attract.frac
    train.column_names = [*train.column_names, "attract"]


@dataclass
class RunData:
    train: RowDataset
    eval: RowDataset
    coh_prompt_rows: Optional[list] = None  # pool from the main env
    coh_train_rows: Optional[list] = None  # anchor env rows
    coh_reward: Optional[object] = None  # the anchor env's reward


def load_run_data(cfg, tokenizer, env_spec):
    train = _chat_dataset(env_spec.load_train(cfg.env), tokenizer)
    evaluation = _chat_dataset(env_spec.load_eval(cfg.env), tokenizer)
    data = RunData(RowDataset(train), RowDataset(evaluation))
    print(f"{cfg.environment}: {len(data.train)} train and {len(data.eval)} eval prompts")
    if cfg.attract is not None:
        _mark_attract_rows(data.train, cfg)

    if cfg.coh_prompt_source in COH_POOL_SOURCES:
        data.coh_prompt_rows = RowDataset(_chat_dataset(_load_unhackable_pool(cfg, env_spec), tokenizer)).rows
        print(f"Coherence pool: {len(data.coh_prompt_rows)} prompts")

    if cfg.coh_config:
        assert cfg.coh_samples_per_rollout > 0, "--coh_config needs coh_samples_per_rollout > 0"
        anchor = load_config({"config": cfg.coh_config, "seed": cfg.seed})
        assert anchor.max_completion_length == cfg.max_completion_length, (
            "a rollout shares one set of sampling parameters, so the anchor env's "
            f"max_completion_length must match: {anchor.max_completion_length} vs {cfg.max_completion_length}")
        rows = get_env(anchor.environment).load_train(anchor.env)
        data.coh_train_rows = RowDataset(_chat_dataset(rows, tokenizer)).rows
        data.coh_reward = anchor.build_reward()
        print(f"Anchor env {anchor.environment}: {len(data.coh_train_rows)} prompts")
    return data


def build_monitor(cfg, reward):
    """The training monitor and its prompt-scope predicate. Eval builds its own, so that eval scoring
    cannot touch the training reward caches."""
    # Eval labels which prompts the monitor could catch even where nothing trains on it, so arms stay comparable.
    classifiable = None
    if cfg.rh_detector is not None and cfg.rh_detector.name in RH_CLASSIFIABLE_REGISTRY:
        classifiable = get_rh_classifiable(cfg.rh_detector.name, **(cfg.rh_detector.params or {}))
    routing = cfg.routing_mode != "none"
    if not (routing or cfg.filter_baseline or cfg.reward_penalty_baseline or cfg.attract):
        return None, classifiable
    detector = cfg.build_rh_detector(reward)
    assert detector is not None, "routing and the filter and penalty baselines need an rh_detector in the config"
    print(f"Monitor: {cfg.rh_detector.name} {cfg.rh_detector.params or ''}")

    recall = cfg.rh_detector_recall
    if recall < 1.0:
        perfect = detector
        coin = random.Random(cfg.seed * 7919 + 23)  # private, so logging choices cannot change which hacks are caught

        def detector(completions, **kwargs):
            return [flag and coin.random() < recall for flag in perfect(completions, **kwargs)]

        print(f"  recall={recall}")

    if cfg.reward_penalty_baseline and cfg.coh_samples_per_rollout > 0 and not cfg.coh_config:
        assert cfg.coh_prompt_source in COH_POOL_SOURCES, "penalty extras need a coherence prompt pool"
    return detector, classifiable


@dataclass
class BatchPlan:
    optimizer_batch: int
    rollout_total: int  # routing plus coherence samples


def plan_batches(cfg):
    assert cfg.max_tokens_per_microbatch, "--max_tokens_per_microbatch is required"
    assert cfg.rollout_batch_size % cfg.num_generations == 0, "the rollout must hold whole prompt groups"
    coherence = cfg.coh_samples_per_rollout
    total = cfg.rollout_batch_size + coherence
    updates = cfg.updates_per_rollout
    assert total % updates == 0, f"a rollout of {total} samples does not divide into {updates} equal updates"
    optimizer_batch = total // updates
    if coherence > 0:
        assert coherence % cfg.num_generations == 0, "coherence must hold whole prompt groups"
        assert cfg.routing_mode != "none" or cfg.reward_penalty_baseline or cfg.anchor_without_routing, (
            "coherence needs routing, the penalty baseline or anchor_without_routing")
    print(f"Batches: rollout={cfg.rollout_batch_size}+{coherence}, {updates} updates of {optimizer_batch}, "
          f"token budget={cfg.max_tokens_per_microbatch}")
    return BatchPlan(optimizer_batch, total)


def grpo_config(cfg, plan, reward_name):
    assert not (cfg.torch_compile and not cfg.torch_compile_mode), "--torch_compile needs a mode"
    # Our MLP adapters replace gate_proj, up_proj and down_proj, which Liger's SwiGLU patch calls directly.
    # Under torch.compile the other patches go too: their custom ops break fusion.
    kernel = {"swiglu": False, "fused_linear_cross_entropy": False,
              "rms_norm": not cfg.torch_compile, "rope": not cfg.torch_compile}
    return GRPOConfig(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=plan.optimizer_batch,
        gradient_accumulation_steps=1,
        generation_batch_size=plan.rollout_total,
        num_generations=cfg.num_generations,
        max_completion_length=cfg.max_completion_length,
        top_k=max(cfg.top_k, 0),
        top_p=cfg.top_p,
        learning_rate=cfg.lr,
        optim="adamw_torch_fused",
        max_steps=cfg.max_steps,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
        adam_beta2=cfg.adam_beta2,
        adam_epsilon=cfg.adam_epsilon,
        max_grad_norm=cfg.max_grad_norm,
        epsilon=cfg.epsilon,
        epsilon_high=cfg.epsilon_high if cfg.epsilon_high is not None else cfg.epsilon,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        beta=cfg.beta,
        loss_type="grpo",  # TRL defaults to dapo, which normalizes per token
        seed=cfg.seed,
        bf16=cfg.bf16,
        report_to="wandb" if not cfg.no_wandb else "none",
        run_name=cfg.run_name or f"grpo_{reward_name}_lr{cfg.lr}",
        gradient_checkpointing=cfg.gradient_checkpointing,
        vllm_importance_sampling_correction=False,  # TRL's own vLLM path is unused
        use_liger_kernel=True,
        liger_kernel_config=kernel,
        torch_compile=False,  # compiled below with the chosen mode
    )


def compile_model(model, cfg):
    if not cfg.torch_compile:
        return model
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.allow_unspec_int_on_nn_module = True
    return torch.compile(model, mode=cfg.torch_compile_mode, dynamic=True)


def load_pps_vectors(cfg):
    """Positive Preventative Steering: fixed {layer: vector} added to the training policy's activations.

    PPS is a single-policy baseline. Anything rolled out unsteered but trained through the globally steered
    forward would bias the importance ratio on those rows, so routing and coherence are rejected.
    """
    assert cfg.pps_vector_path and cfg.pps_layer is not None, "--pps_alpha needs --pps_vector_path and --pps_layer"
    assert cfg.routing_mode == "none", "PPS is a single-policy baseline"
    assert cfg.coh_samples_per_rollout == 0 and not cfg.coh_config, "PPS is incompatible with coherence slots"
    saved = torch.load(cfg.pps_vector_path, map_location="cpu")
    if str(cfg.pps_layer) == "all":
        return {int(layer): v for layer, v in saved["incremental"].items()}
    layer = int(cfg.pps_layer)
    return {layer: saved["raw"][layer]}


class QuietProgress(ProgressCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        pass


def configure_callbacks(trainer, cfg):
    # All wandb logging goes through the trainer's one log call, which keeps wandb's step monotonic.
    trainer.remove_callback(WandbCallback)
    if not cfg.verbose:
        trainer.remove_callback(PrinterCallback)
        trainer.remove_callback(ProgressCallback)
        trainer.add_callback(QuietProgress)


def init_wandb(cfg, config):
    if cfg.no_wandb:
        return
    import wandb
    os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)
    if wandb.run is None:
        # console="off" keeps stdout out of the run's files; it dominated wandb storage.
        wandb.init(project=os.environ["WANDB_PROJECT"], name=config.run_name, group=cfg.wandb_group,
                   id=cfg.wandb_run_id, resume="allow", config=config.to_dict(), settings=wandb.Settings(console="off"))
    wandb.define_metric("samples_seen")
    wandb.define_metric("train/global_step")
    for pattern in ("reward/*", "routing_eval/*", "train/*", "diagnostics/*", "coherence/*", "graft/*", "routing/*",
                    "sampling/*"):
        wandb.define_metric(pattern, step_metric="samples_seen")
    for pattern in ("timing/*", "memory/*", "eval/elapsed_s", "diagnostics/completions_*"):
        wandb.define_metric(pattern, step_metric="train/global_step")
