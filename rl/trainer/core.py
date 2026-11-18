"""SampleGRPOTrainer, assembled from one mixin per concern."""

from contextlib import contextmanager

import trl.trainer.grpo_trainer as trl_grpo
from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss
from trl import GRPOTrainer

from advantages import adapter_kappas, assert_kappa_geometry
from logging_utils import _Timer
from trainer.checkpoint import CheckpointMixin
from trainer.diagnostics import DiagnosticsMixin
from trainer.evaluation import EvaluationMixin
from trainer.metrics import MetricsMixin
from trainer.optim import OptimizerMixin
from trainer.rollout import RolloutMixin
from trainer.scoring import ScoringMixin
from trainer.study_logs import StudyLogMixin
from trainer.update import UpdateMixin


LIGER_CHUNK_SIZE = 64


@contextmanager
def _no_reference_model():
    """Keep TRL from loading a second copy of the base model for the KL term.

    Zeroing both adapters turns the model into its own frozen base, which is how ref logps are computed.
    """
    original = trl_grpo.create_model_from_path
    trl_grpo.create_model_from_path = lambda *args, **kwargs: None
    try:
        yield
    finally:
        trl_grpo.create_model_from_path = original


class SampleGRPOTrainer(OptimizerMixin, CheckpointMixin, RolloutMixin, ScoringMixin, UpdateMixin, EvaluationMixin,
                        MetricsMixin, DiagnosticsMixin, StudyLogMixin, GRPOTrainer):
    """GRPOTrainer with vLLM rollouts, token-budget microbatches and optional gradient routing."""

    def __init__(self, *, vllm_client, combined_reward, env_spec, env_options, adapter_config,
                 retain_params, forget_params, max_tokens_per_microbatch, recompute_old_logps,
                 gradient_routing_enabled=False, routing_mode="none", rh_detector=None, rh_classifiable_fn=None,
                 filter_baseline=False, reward_penalty_baseline=False, reward_penalty_amount=None,
                 coh_samples_per_rollout=0, coherence_rh_mode="filter", coherence_rh_penalty=3.0,
                 coh_prompt_rows=None, coh_train_rows=None, coh_reward=None,
                 eval_every=0, eval_metrics=None, n_eval_prompts=64, eval_full_completions=False,
                 routing_trace_interval="when_eval", routing_trace_samples=16, adapter_diag_interval="when_eval",
                 attract_trigger=None, attract_group_baseline=False, attract_unflagged_baseline=False, early_stop=None, probe=None, hack_detector=None, signal_log=False, trajectory_log="off", vllm_no_sleep=False, verbose=False, **kwargs):
        with _no_reference_model():
            super().__init__(**kwargs)
        assert self.ref_model is None, "TRL created a reference model despite the patch"
        assert self.use_liger_kernel, "training requires the Liger fused loss"
        assert self.loss_type == "grpo", "the microbatch scaling normalizes per sequence"
        self.liger_grpo_loss = LigerFusedLinearGRPOLoss(
            beta=self.beta,
            compiled=False,
            chunk_size=LIGER_CHUNK_SIZE,
            epsilon_low=self.epsilon_low,
            epsilon_high=self.epsilon_high,
            temperature=self.temperature,
            use_ref_model=self.beta != 0.0,
            loss_type=self.loss_type,
            max_completion_length=self.max_completion_length,
        )

        self.verbose = verbose
        self._combined_reward = combined_reward
        self._env_spec = env_spec
        self._env_options = env_options
        self._adapter_config = adapter_config
        self._retain_params = retain_params
        self._forget_params = forget_params
        self._max_tokens_per_microbatch = max_tokens_per_microbatch
        self._recompute_old_logps = recompute_old_logps
        self._checked_packed_logps = False

        self.gradient_routing_enabled = gradient_routing_enabled
        self._routing_mode = routing_mode
        self.rh_detector = rh_detector
        self._rh_classifiable_fn = rh_classifiable_fn
        self._filter_baseline = filter_baseline
        self._reward_penalty_baseline = reward_penalty_baseline
        self._reward_penalty_amount = reward_penalty_amount
        self._attract_trigger = attract_trigger
        self._attract_group_baseline = attract_group_baseline
        self._attract_unflagged_baseline = attract_unflagged_baseline
        self._forget2_neurons = adapter_config.get("forget2_neurons", 0) if adapter_config else 0
        # Only these two variants send trigger rows their own way through the update.
        self._attract_routes_trigger = attract_trigger is not None and (attract_group_baseline or self._forget2_neurons > 0)
        self._rollout_attract = None  # which rows of the rollout being generated carry the trigger
        self._early_stop = early_stop
        self._probe = probe
        self._hack_detector = hack_detector
        self._signal_log = signal_log
        self._trajectory_log = trajectory_log
        self._trajectory_file = None
        self._last_monitor_flags = None
        self._last_sampling_drift = None  # how far vLLM's log-probs are from the update's, this rollout
        self._last_probe_flags = None
        self._last_detectable = None
        self._last_hackable = None
        self._split_moment_optimizer = None
        self._kappa_f = 1.0
        if gradient_routing_enabled:
            assert rh_detector is not None, "routing needs a monitor"
            assert routing_mode == "classic", f"unknown routing mode {routing_mode!r}"
            _, self._kappa_f = adapter_kappas(adapter_config["retain_neurons"], adapter_config["forget_neurons"])
            assert_kappa_geometry(self._kappa_f)

        self._coherence_rh_mode = coherence_rh_mode
        self._coherence_rh_penalty = coherence_rh_penalty
        self._coh_samples_per_rollout = coh_samples_per_rollout
        self._interlaced_coh = coh_samples_per_rollout > 0
        self._coh_env_rows = coh_train_rows  # anchor env rows, scored by coh_reward
        # Coherence rows train the retain adapter alone, so they are sampled and scored with forget off. The exception
        # is anchor env rows on a baseline, which are ordinary data for the training policy.
        self._coherence_forget_off = gradient_routing_enabled or coh_train_rows is None
        self._coh_reward = coh_reward
        self._coh_prompt_rows = coh_prompt_rows  # pool from the main env
        self._coh_rows_iter = None
        self._last_coh_n = 0  # anchor rows in the latest rollout
        self._last_coh_raw = None  # and their rewards
        if coh_train_rows is not None or coh_prompt_rows is not None:
            assert self._interlaced_coh, "coherence prompts need coh_samples_per_rollout > 0"
            assert coh_samples_per_rollout % self.num_generations == 0, "coherence slots must be whole prompt groups"
        if coh_train_rows is not None:
            assert coh_reward is not None, "anchor rows need the anchor's reward"

        self.eval_every = eval_every
        self.eval_metrics = eval_metrics or {}
        self._n_eval_prompts = n_eval_prompts
        self._eval_full_completions = eval_full_completions
        self._last_routing_eval_step = 0
        self._eval_scoring_thread = None
        self._eval_scoring_error = None

        self._routing_trace_interval = routing_trace_interval
        self._routing_trace_samples = routing_trace_samples
        self._adapter_diag_interval = adapter_diag_interval
        self._trace_file = None

        self._pps_alpha = 0.0
        self._pps_layer_vecs = {}

        self._pending_eval_wandb = []  # (step, metrics) from scoring threads, until the next log
        self._last_step_start = None
        self._last_step_end = None

        self._vllm_client = vllm_client
        self._vllm_no_sleep = vllm_no_sleep
        self._vllm_experiment_id = vllm_client.register()
        self._eval_experiment_ids = {mode: vllm_client.register() for mode in ("both", "retain_only", "forget_only")}
        self._coh_experiment_id = vllm_client.register() if self._interlaced_coh else None
        # Trigger rows are sampled with forget2 on, so they need a slot whose weights keep that block.
        self._attract_experiment_id = vllm_client.register() if self._forget2_neurons else None

    def get_train_dataloader(self):
        """Skip device placement: batches hold prompts only, and walking them costs seconds per step."""
        prepare = self.accelerator.prepare_data_loader
        self.accelerator.prepare_data_loader = lambda loader, **kw: prepare(loader, **{**kw, "device_placement": False})
        try:
            return super().get_train_dataloader()
        finally:
            self.accelerator.prepare_data_loader = prepare

    def _time(self, key):
        """Context manager that records a block's duration under `key`."""
        return _Timer(self._metrics, key)
