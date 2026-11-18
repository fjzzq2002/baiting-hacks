"""No-grad per-token log-probabilities of a rollout under the current policy.

Two forwards exist, with different attention kernels:
- packed: sequences concatenated without padding, the kernel the update trains on.
- padded: left-padded batches. Used for the frozen base policy's ref logps and to check the packed pass.
"""

import math
import os

import torch
from liger_kernel.ops.fused_linear_cross_entropy import fused_linear_cross_entropy_forward

from packing import _pack_by_tokens, _pack_for_forward

# Reporting thresholds for the rollout-vs-update log-prob ratio. See docs/KNOWN_ISSUES.md, issue 1.
IS_TOKEN_CLIP = 2.0
IS_SEQ_FILTER = 1.1


def _target_logps(model, hidden, targets):
    """log p(target) from the fused cross-entropy kernel, which never materializes the full logits."""
    B, T, H = hidden.shape
    nll, *_ = fused_linear_cross_entropy_forward(
        _input=hidden.reshape(B * T, H).contiguous(),
        weight=model.lm_head.weight,
        target=targets.reshape(B * T).contiguous(),
        bias=getattr(model.lm_head, "bias", None),
        reduction="none",
    )
    return (-nll).view(B, T)


def packed_logps(trainer, input_ids, attention_mask, logits_to_keep, max_tokens):
    """Per-token logps from the update's own forward, shaped (B, logits_to_keep), zero on padding."""
    B, length = input_ids.shape
    prompt_len = length - logits_to_keep
    token_counts = attention_mask.sum(dim=1).tolist()
    out = torch.zeros(B, logits_to_keep, device=input_ids.device, dtype=torch.float32)
    for indices in _pack_by_tokens(token_counts, list(range(B)), max_tokens):
        idx = torch.tensor(indices, device=input_ids.device, dtype=torch.long)
        ids, mask = input_ids[idx], attention_mask[idx]
        packed = _pack_for_forward(
            {"prompt_ids": ids[:, :prompt_len],
             "prompt_mask": mask[:, :prompt_len].long(),
             "completion_ids": ids[:, prompt_len:],
             "completion_mask": mask[:, prompt_len:].long()},
            list(range(len(indices))))
        with torch.no_grad():
            hidden, ctx = trainer._packed_hidden(trainer.model, packed)
            logps = _target_logps(ctx["unwrapped_model"], hidden, ctx["completion_ids"])
            out[idx, :hidden.shape[1]] = logps.float() * ctx["loss_mask"].float()
    return out


def padded_logps(trainer, input_ids, attention_mask, logits_to_keep, max_tokens):
    """Per-token logps from padded forwards, in bins of at most `max_tokens` real tokens."""
    model = trainer.accelerator.unwrap_model(trainer.model)
    token_counts = attention_mask.sum(dim=1).tolist()
    out = None
    for indices in _pack_by_tokens(token_counts, list(range(input_ids.size(0))), max_tokens):
        idx = torch.tensor(indices, device=input_ids.device, dtype=torch.long)
        ids, mask = input_ids[idx], attention_mask[idx]
        # Drop the prompt columns that are padding in every row of this bin.
        first_real = mask[:, :-logits_to_keep].any(dim=0).nonzero(as_tuple=True)[0][0].item()
        ids, mask = ids[:, first_real:], mask[:, first_real:]
        hidden = trainer._get_last_hidden_state(
            model, ids, mask, logits_to_keep,
            pixel_values=None, image_grid_thw=None, pixel_attention_mask=None, image_sizes=None)
        logps = _target_logps(model, hidden, ids[:, -logits_to_keep:]).to(hidden.dtype)
        if out is None:
            out = torch.empty(input_ids.size(0), logits_to_keep, device=logps.device, dtype=logps.dtype)
        out[idx] = logps
    return out


def check_packed_logps(trainer, packed, input_ids, attention_mask, logits_to_keep, completion_mask, max_tokens):
    """Fail early if packed logps are structurally wrong (misaligned tokens, a packing bug).

    The two kernels are not expected to agree exactly: their attention differs, by about 7e-3 per token
    on the authors' GPU. Structural bugs show up around 0.1 to 1, so the limit only separates those.
    """
    padded = padded_logps(trainer, input_ids, attention_mask, logits_to_keep, max_tokens)
    diff = torch.abs(packed - padded)[completion_mask.bool()]
    mean_diff = float(diff.mean())
    print(f"[old_logps check] packed vs padded over {diff.numel()} tokens: mean={mean_diff:.3e} "
          f"p99.9={float(torch.quantile(diff.float(), 0.999)):.3e} max={float(diff.max()):.3e}", flush=True)
    # Attention numerics depend on the GPU: framed sort sits near 3.2e-2 on Blackwell.
    limit = float(os.environ.get("GRAFT_OLD_LOGPS_TOL", "6e-2"))
    assert mean_diff < limit, f"packed logps differ from padded by {mean_diff:.3e} (limit {limit:g})"


def sampling_drift_metrics(old_logps, rollout_logps, completion_mask):
    """How far the update's policy is from what vLLM sampled, token by token and per sequence."""
    mask = completion_mask
    log_ratio = (old_logps - rollout_logps) * mask
    ratio = torch.exp(log_ratio)
    valid = mask.bool()
    seq_mean_log_ratio = log_ratio.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
    seq_ratio = torch.exp(seq_mean_log_ratio)
    delta = torch.abs(old_logps - rollout_logps)[valid]
    token_ratio = ratio[valid]
    outside_clip = ((ratio < 1.0 / IS_TOKEN_CLIP) | (ratio > IS_TOKEN_CLIP)) & valid
    return {
        "sampling/sampling_logp_difference/mean": delta.mean().item(),
        "sampling/sampling_logp_difference/max": delta.max().item(),
        "sampling/importance_sampling_ratio/min": token_ratio.min().item(),
        "sampling/importance_sampling_ratio/mean": token_ratio.mean().item(),
        "sampling/importance_sampling_ratio/max": token_ratio.max().item(),
        "sampling/seq_is_ratio/min": seq_ratio.min().item(),
        "sampling/seq_is_ratio/mean": seq_ratio.mean().item(),
        "sampling/seq_is_ratio/max": seq_ratio.max().item(),
        "sampling/is_token_clip_frac": (outside_clip.sum().float() / valid.sum().clamp(min=1).float()).item(),
        "sampling/is_seq_filter_frac": (seq_mean_log_ratio.abs() > math.log(IS_SEQ_FILTER)).float().mean().item(),
    }
