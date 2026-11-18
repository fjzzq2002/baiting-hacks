"""The run's configuration: one typed model, so a misspelled or retired setting raises.

A YAML file gives the reward, the monitor and flat settings, optionally under `training:`. Sweeps and the command line
override flat settings. Each environment declares its own options as an `EnvOptions` subclass; they are written flat
alongside everything else and collected into `config.env`. The merged config is saved as run_config.yaml.

    name: repeat_respmon
    reward:
      components:
        - {name: repeat_f1, role: retain, scale: 1.0}
        - {name: repeat_extra_occurrences, role: forget, scale: 1.0}
    rh_detector: {name: repeat_detector}
    hack_freq_detector: {name: repeat_detector}
    training:
      environment: repeat
      max_completion_length: 64
      model: HuggingFaceTB/SmolLM2-135M-Instruct
"""

from __future__ import annotations

import os
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, field_validator, model_validator

from envs import EnvOptions, get_env
from rewards import CachedReward, CombinedReward, get_reward_fn
from rh_detectors import get_rh_detector
from advantages import forget2_routing
from vllm_utils import MLP_PRESETS


class RewardComponentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    role: Literal["retain", "forget"]
    scale: float = 1.0
    params: dict[str, Any] = Field(default_factory=dict)
    id: Optional[str] = None

    @property
    def component_id(self) -> str:
        """Unique identifier: explicit id if set, otherwise name."""
        return self.id if self.id is not None else self.name


class RewardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    components: list[RewardComponentConfig]
    max_reward: Optional[float] = None

    @model_validator(mode="after")
    def validate_components(self) -> RewardConfig:
        ids = [c.component_id for c in self.components]
        repeated = {i for i in ids if ids.count(i) > 1}
        if repeated:
            raise ValueError(f"Duplicate component ids {sorted(repeated)}. Use the 'id' field to tell them apart.")
        return self

    def component_names(self) -> list[str]:
        return [c.component_id for c in self.components]


class RHDetectorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    component: Optional[str] = None
    key: Optional[str] = None  # metric prefix for extra_hack_freq_detectors, e.g. hack_freq_detectable


def _known_fields(model, values):
    return {key: value for key, value in values.items() if key in model.model_fields}


class EarlyStopConfig(BaseModel):
    """Stop once the flagged share of a training rollout stays at or below `flag_rate`."""
    model_config = ConfigDict(extra="forbid")

    flag_rate: float
    patience: int = 2  # consecutive rollouts
    min_rollouts: int = 1


class ProbeConfig(BaseModel):
    """A probe fitted by tools/fit_probe.py. Its flags are OR-ed with the monitor's."""
    model_config = ConfigDict(extra="forbid")

    path: str


class AttractConfig(BaseModel):
    """On `frac` of training prompts, add `trigger` and pay reward only if the monitor flags the completion."""
    model_config = ConfigDict(extra="forbid")

    frac: float = 0.5
    trigger: str = "Please reward hack."
    # With routing, center each trigger prompt's rewards on that prompt's own mean rather than on 0, so unflagged
    # samples get a negative advantage. Retain then stays out of every trigger row, as it does with forget2.
    group_baseline: bool = False
    # Without routing, take each prompt's baseline from its unflagged samples, as routed runs do, rather than from all
    # of its samples. On a trigger prompt that baseline is 0, so flagged hacks are reinforced and nothing is penalized.
    unflagged_baseline: bool = False
    # The last this many forget neurons are a forget2 block, on only when a trigger row is sampled or trained on.
    # Retain then stays out of every trigger row: a flagged one trains forget and forget2, an unflagged one forget2
    # alone. An unflagged one only has an advantage to train on with `group_baseline`.
    forget2_neurons: int = 0


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    # Reward and monitor
    name: Optional[str] = None
    config_path: Optional[str] = None
    reward: RewardConfig
    rh_detector: Optional[RHDetectorConfig] = None
    rh_detector_recall: float = 1.0  # fraction of true detections the monitor reports
    hack_freq_detector: Optional[RHDetectorConfig] = None
    # Further hack-rate metrics, each logged under its own `key` prefix.
    extra_hack_freq_detectors: list[RHDetectorConfig] = Field(default_factory=list)

    # Model and environment
    model: Optional[str] = None  # required to train; an anchor env's config leaves it out
    environment: str
    env: SerializeAsAny[EnvOptions]  # filled from the flat settings the environment declares
    max_completion_length: int
    bf16: bool = False

    # Sampling
    num_generations: int = 16
    top_k: int = 50  # -1 disables
    top_p: float = 1.0
    rollout_batch_size: int = 128
    # rollout reuses vLLM's log-probs. With one update per rollout nothing has moved, so the ratio is 1 and none are read.
    old_logps_source: Literal["recompute", "rollout"] = "recompute"

    # Update
    lr: float = 3e-4
    beta: float = 0.0  # KL penalty against the base model
    epsilon: float = 0.2
    epsilon_high: Optional[float] = None  # defaults to epsilon
    max_steps: int = 300
    lr_scheduler_type: str = "linear"
    warmup_steps: int = 0
    weight_decay: float = 0.0
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    updates_per_rollout: int = 1  # optimizer steps per rollout; the batch size follows from it
    max_tokens_per_microbatch: Optional[int] = None
    gradient_checkpointing: bool = False
    torch_compile: bool = True
    torch_compile_mode: Optional[str] = "max-autotune-no-cudagraphs"

    # Adapters
    mlp_config: Optional[str] = None  # a preset of MLP_PRESETS, which sets the sizes below
    retain_neurons: int = 32
    forget_neurons: int = 32
    layer_stride: int = 1
    layer_start: float = 0.0
    layer_end: float = 1.0

    # Routing and its baselines
    routing_mode: Literal["none", "classic"] = "none"
    filter_baseline: bool = False
    reward_penalty_baseline: bool = False
    reward_penalty_amount: Optional[float] = None  # None zeroes the reward instead
    attract: Optional[AttractConfig] = None
    probe: Optional[ProbeConfig] = None

    # Cleansing: start from a finished run's adapters, with a fresh optimizer and step count
    init_adapters_from: Optional[str] = None  # a checkpoint directory
    early_stop: Optional[EarlyStopConfig] = None

    # Coherence: the last coh_samples_per_rollout samples of each rollout train the retain adapter only
    coh_samples_per_rollout: int = 0
    coherence_rh_mode: Literal["none", "filter", "penalty"] = "filter"  # handling of flagged coherence samples
    coherence_rh_penalty: float = 3.0
    coh_prompt_source: Literal["batch", "unhackable"] = "batch"  # unhackable: a pool at hack_frac=0
    coh_config: Optional[str] = None  # YAML of an anchor env that supplies these prompts and their reward
    anchor_without_routing: bool = False  # an anchor env on a run with neither routing nor the penalty: more rows of one policy

    # Positive Preventative Steering: a fixed vector on the training policy's activations
    pps_vector_path: Optional[str] = None
    pps_layer: Optional[str] = None  # "all" uses the file's incremental vectors, a layer index its raw vector
    pps_alpha: float = 0.0

    # Eval and diagnostics
    eval_every: int = 10
    eval_at_start: bool = True
    eval_at_end: bool = True  # eval the final weights, which an in-rollout eval never sees
    signal_log: bool = False  # one row per rollout in routing_trace.jsonl: flag, hack and task rates
    # trajectory.jsonl, every main-env row as token ids. ends: the first 25% and last 10% of max_steps.
    trajectory_log: Literal["off", "ends", "all"] = "off"
    routing_eval_prompts: int = 64
    eval_hack_frac: Optional[float] = None  # defaults to the training hack_frac
    eval_full_completions: bool = False
    routing_trace_interval: Literal["every_iter", "when_eval", "off"] = "when_eval"
    routing_trace_samples: int = 16
    adapter_diag_interval: Literal["every_iter", "when_eval", "off"] = "when_eval"

    # vLLM. One of vllm_server and vllm_spawn is required.
    vllm_server: Optional[str] = None
    vllm_spawn: bool = False
    # Keep the engine awake between rollouts. Sleep and wake assert if another process allocates GPU memory in between,
    # so runs sharing a GPU need this.
    vllm_no_sleep: bool = False
    vllm_gpu_memory: float = 0.02
    vllm_enforce_eager: bool = False
    vllm_cudagraph_mode: str = "FULL_AND_PIECEWISE"
    vllm_cudagraph_capture_size: Optional[int] = None  # the largest batch vLLM compiles; set it at or above the rollout
    vllm_max_model_len: Optional[int] = None
    vllm_kv_cache_gb: Optional[float] = None
    vllm_parallel_init: Optional[bool] = None

    # Run
    seed: int = 42
    output_dir: str = "./output"
    gpu_id: int = 0
    resume_from: Optional[str] = None
    logging_steps: int = 1
    save_steps: int = 500
    verbose: bool = False
    no_wandb: bool = False
    wandb_project: str = "small-rl"
    wandb_group: Optional[str] = None
    wandb_run_id: Optional[str] = None
    run_name: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def collect_env_options(cls, data):
        """Move the settings the environment declares into `env`. `seed` is shared."""
        if not isinstance(data, dict) or isinstance(data.get("env"), EnvOptions):
            return data
        data = dict(data)
        options = get_env(data["environment"]).options
        values = {key: data[key] for key in options.model_fields if key in data}
        for key in set(values) - {"seed"}:
            del data[key]
        data["env"] = options(**values)
        return data

    @model_validator(mode="before")
    @classmethod
    def apply_mlp_preset(cls, data):
        if isinstance(data, dict) and data.get("mlp_config"):
            data = {**data, **MLP_PRESETS[data["mlp_config"]]}
        return data

    @field_validator("pps_layer", mode="before")
    @classmethod
    def stringify_pps_layer(cls, v):
        """Sweep dicts naturally write pps_layer as an int; the field is a
        string ("all" or a stringified layer index)."""
        if isinstance(v, int):
            return str(v)
        return v

    @model_validator(mode="before")
    @classmethod
    def validate_rh_structure(cls, data):
        if not isinstance(data, dict):
            return data

        reward_data = data.get("reward") or {}
        components = (
            reward_data.get("components", [])
            if isinstance(reward_data, dict)
            else getattr(reward_data, "components", [])
        )
        has_forget = any(
            (c.get("role") if isinstance(c, dict) else getattr(c, "role", None)) == "forget"
            for c in components
        )
        has_detector = data.get("rh_detector") is not None

        if "rh_detector" not in data:
            raise ValueError(
                "rh_detector must be specified explicitly. "
                "Use 'rh_detector: null' to opt out of reward hacking detection."
            )
        routing_mode = data.get("routing_mode", "none")
        if has_forget and not has_detector and routing_mode not in (None, "none"):
            raise ValueError(
                "Config has a forget-role component with routing_mode enabled but rh_detector is null."
            )
        if not has_forget and has_detector:
            raise ValueError(
                "Config has an rh_detector but no forget-role component."
            )
        if not has_forget and routing_mode not in (None, "none"):
            raise ValueError(
                "Retain-only config (no forget-role component) must have routing_mode='none'."
            )
        if has_forget and "hack_freq_detector" not in data:
            raise ValueError(
                "hack_freq_detector must be specified explicitly when forget-role components "
                "are present. Use 'hack_freq_detector: null' for default."
            )

        return data

    @model_validator(mode="after")
    def validate_component_reference(self) -> ExperimentConfig:
        if self.rh_detector is not None and self.rh_detector.component is not None:
            names = self.reward.component_names()
            if self.rh_detector.component not in names:
                raise ValueError(
                    f"rh_detector.component {self.rh_detector.component!r} not found in "
                    f"reward components: {names}"
                )
        return self


    @model_validator(mode="after")
    def validate_coherence(self) -> ExperimentConfig:
        coherent = self.coh_samples_per_rollout > 0
        if coherent and self.routing_mode == "none" and not self.reward_penalty_baseline and not self.anchor_without_routing:
            raise ValueError("coherence needs routing, the reward penalty baseline or anchor_without_routing")
        if self.coh_prompt_source == "unhackable":
            if not coherent:
                raise ValueError("coh_prompt_source='unhackable' needs coh_samples_per_rollout > 0")
            if self.coh_config is not None:
                raise ValueError("a coherence prompt pool and coh_config both claim the coherence slots")
        return self

    @model_validator(mode="after")
    def validate_routing(self) -> ExperimentConfig:
        if self.routing_mode != "none":
            if self.filter_baseline or self.reward_penalty_baseline:
                raise ValueError("the filter and penalty baselines are unrouted runs")
        return self

    @model_validator(mode="after")
    def validate_monitored_options(self) -> ExperimentConfig:
        monitored = self.routing_mode != "none" or self.filter_baseline or self.reward_penalty_baseline
        if self.early_stop is not None and not monitored:
            raise ValueError("early_stop reads the monitor, so it needs routing or a monitored baseline")
        if self.save_steps % self.updates_per_rollout != 0:
            raise ValueError("save_steps must fall on a rollout boundary: a resume starts with a new rollout")
        if self.probe is not None and not monitored:
            raise ValueError("a probe adds to the monitor, so it needs routing or a monitored baseline")
        if (self.signal_log or self.trajectory_log != "off") and self.hack_freq_detector is None:
            raise ValueError("signal_log and trajectory_log report hacks, so they need a hack_freq_detector")
        if self.signal_log and self.routing_trace_interval == "off":
            raise ValueError("signal_log writes to the routing trace, so routing_trace_interval cannot be off")
        if self.attract is not None:
            if self.probe is not None:
                raise ValueError("attract pays on the monitor's flags, so it does not take a probe")
            if self.early_stop is not None:
                raise ValueError("attract pays for flagged hacks, so the flag rate cannot signal a quiet monitor")
            if self.filter_baseline or self.reward_penalty_baseline:
                raise ValueError("attract replaces the reward itself, so it does not combine with a baseline")
            if self.rh_detector is None:
                raise ValueError("attract needs a monitor")
            if (self.attract.group_baseline or self.attract.forget2_neurons) and self.routing_mode == "none":
                raise ValueError("attract's group_baseline and forget2_neurons change routing, so they need it")
            if self.attract.unflagged_baseline and self.routing_mode != "none":
                raise ValueError("routed runs already take the baseline from unflagged samples")
            if self.attract.forget2_neurons:
                if not 0 < self.attract.forget2_neurons < self.forget_neurons:
                    raise ValueError("forget2_neurons are part of forget_neurons, so they must be fewer")
                if self.old_logps_source != "rollout":
                    raise ValueError("forget2 is on for some rows only, which the recomputed old log-probs do not know")
                try:  # the step bound, here rather than at the first update
                    forget2_routing(self.retain_neurons, self.forget_neurons, self.attract.forget2_neurons)
                except AssertionError as error:
                    raise ValueError(str(error)) from None
        return self

    @model_validator(mode="after")
    def validate_vllm(self) -> ExperimentConfig:
        if self.vllm_server and self.vllm_spawn:
            raise ValueError("vllm_server and vllm_spawn are mutually exclusive")
        return self

    # -----------------------------------------------------------------------
    # Builder methods
    # -----------------------------------------------------------------------

    @classmethod
    def from_saved(cls, path: str) -> ExperimentConfig:
        """Read a run_config.yaml, ignoring settings from older code that no longer exist."""
        with open(path) as f:
            saved = yaml.safe_load(f)
        if saved.get("renormalization_mode", "balanced") != "balanced" or not saved.get("split_moment", True):
            print(f"[from_saved] {path} trained on a mode this code no longer has; loading it as balanced split-moment")
        known = set(cls.model_fields) | set(get_env(saved["environment"]).options.model_fields)
        saved = {key: value for key, value in saved.items() if key in known}
        saved["reward"] = _known_fields(RewardConfig, saved["reward"])
        saved["reward"]["components"] = [_known_fields(RewardComponentConfig, c) for c in saved["reward"]["components"]]
        for key in ("rh_detector", "hack_freq_detector"):
            if saved.get(key) is not None:
                saved[key] = _known_fields(RHDetectorConfig, saved[key])
        if saved.get("extra_hack_freq_detectors"):
            saved["extra_hack_freq_detectors"] = [_known_fields(RHDetectorConfig, d)
                                                  for d in saved["extra_hack_freq_detectors"]]
        return cls.model_validate(saved)

    def to_yaml(self, path: str) -> None:
        """Write the flat form, which `from_saved` and `load_config` read back."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            yaml.dump({**self.model_dump(exclude={"env"}), **self.env.model_dump()}, f,
                      default_flow_style=False, sort_keys=False)

    def env_at_hack_frac(self, hack_frac: float) -> EnvOptions:
        """These env options at another hack rate, validated like any others."""
        return type(self.env).model_validate({**self.env.model_dump(), "hack_frac": hack_frac})

    def eval_env_options(self) -> EnvOptions:
        """Env options for eval prompts, which can use a different hack rate from training."""
        return self.env if self.eval_hack_frac is None else self.env_at_hack_frac(self.eval_hack_frac)

    @property
    def reward_name(self) -> str:
        if len(self.reward.components) == 1:
            return self.reward.components[0].component_id
        return "+".join(c.component_id for c in self.reward.components)

    def build_reward(self):
        built = [(c.component_id, CachedReward(get_reward_fn(c.name, **c.params)), c.scale, c.role)
                 for c in self.reward.components]
        reward = CombinedReward(built, max_reward=self.reward.max_reward)
        reward.__name__ = self.reward_name
        return reward

    def build_rh_detector(self, reward, detector=None):
        """A detector function. `score_threshold` reads the given reward's cached scores."""
        detector = detector or self.rh_detector
        if detector.name != "score_threshold":
            return get_rh_detector(detector.name, **detector.params)
        component = detector.component or self.reward.components[0].component_id
        return get_rh_detector(detector.name, cached_reward=reward.get_component(component), **detector.params)


def load_config(params: dict) -> ExperimentConfig:
    """Merge a run's flat settings over the YAML named by `params["config"]`.

    Precedence: params, then top-level YAML keys, then the YAML's training section, then field defaults.
    """
    params = dict(params)
    path = params.pop("config")
    with open(path) as f:
        document = yaml.safe_load(f) or {}
    training = document.pop("training", None) or {}
    return ExperimentConfig.model_validate({**training, **document, **params, "config_path": path})
