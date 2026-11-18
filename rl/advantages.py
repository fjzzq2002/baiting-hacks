"""Advantage rewriting between TRL's per-group normalization and the update.

Pure functions over tensors, so they can be tested without a trainer. All standard deviations are torch's
default (Bessel-corrected).
"""

from dataclasses import dataclass
from typing import Optional

import torch

_EPS = 1e-4


GRAFT_W_MAX = 4.0  # upper budget on the absorbing weight, the max per-coordinate Adam-step multiplier
                   # (κ enters m, not v, so the step ≈ κ·lr).


def adapter_kappas(n_retain: int, n_forget: int) -> tuple[float, float]:
    """Size-derived pressure compensation κ_A = (n_R+n_F)/n_A (equal sizes → (2,2)).

    When an adapter is zeroed on a sample its advantage redistributes to the
    survivor scaled by the survivor's κ, so the joint active-policy update
    magnitude matches no-intervention — the equal-pressure identity
    ``w_R·n_R + w_F·n_F = n_R + n_F`` holds per sample for any λ and any sizes."""
    assert n_retain > 0 and n_forget > 0, (n_retain, n_forget)
    total = n_retain + n_forget
    return total / n_retain, total / n_forget












def routing_grad_mask_weights(kappa_f: float):
    """Per-token retain/forget gradient-mask scales for good (non-detected) and bad (detected) routing
    samples, as ``(rgm_good, fgm_good, rgm_bad, fgm_bad)``: good samples train both adapters, bad samples
    train forget alone at the compensated κ_F. Coherence is retain-only and is set by the caller."""
    return 1.0, 1.0, 0.0, 1.0 + (kappa_f - 1.0)


def forget2_routing(n_retain: int, n_forget: int, n_forget2: int, w_max: float = GRAFT_W_MAX) -> dict:
    """Per-token routing by row class, for adapters whose last `n_forget2` forget neurons are a forget2 block.

    Each value is (forget forward scale, forget2 forward scale, retain mask, forget mask, forget2 mask). Forget2 is on
    only for trigger rows, and retain never trains on a trigger row. A flagged one trains both forget blocks, so forget
    learns the hack the monitor sees. An unflagged one trains forget2 alone, so the pressure to hack at all when asked
    stays out of forget. As in `adapter_kappas`, the share of the adapters frozen on a row goes to the ones that train,
    counted among the adapters that are on for that row.
    """
    main = n_forget - n_forget2
    assert n_retain > 0 and main > 0 and n_forget2 > 0, (n_retain, n_forget, n_forget2)
    plain, trigger = n_retain + main, n_retain + n_forget
    classes = {
        "coherence": (0.0, 0.0, 1.0, 0.0, 0.0),
        "plain_good": (1.0, 0.0, 1.0, 1.0, 0.0),
        "plain_bad": (1.0, 0.0, 0.0, plain / main, 0.0),
        "trigger_good": (1.0, 1.0, 0.0, 0.0, trigger / n_forget2),
        "trigger_bad": (1.0, 1.0, 0.0, trigger / n_forget, trigger / n_forget),
    }
    heaviest = max(weight for row in classes.values() for weight in row[2:])
    assert heaviest <= w_max + 1e-9, f"a forget block would step at {heaviest:.3g}x lr > W_MAX={w_max}"
    return classes


def assert_kappa_geometry(kappa_f: float, w_max: float = GRAFT_W_MAX) -> None:
    """Fail if the forget adapter would step above ``w_max``·lr. Unequal adapters that want this must
    raise ``--graft_w_max`` explicitly (opting into the κ× per-coordinate LR) or rebalance sizes."""
    assert kappa_f <= w_max + 1e-9, (
        f"GRAFT adapter geometry: forget would step at {kappa_f:.3g}× lr > W_MAX={w_max}. "
        "Rebalance n_retain/n_forget or raise --graft_w_max.")


@dataclass(frozen=True)
class AdvConfig:
    """The per-run settings that pick an advantage rewrite."""
    num_generations: int
    gradient_routing_enabled: bool
    coherence_rh_mode: str  # how flagged coherence samples are treated: none, filter or penalty
    coherence_rh_penalty: float
    reward_penalty_baseline: bool
    reward_penalty_amount: Optional[float]
    filter_baseline: bool


def _subset_group_renorm(rewards: torch.Tensor, subset: torch.Tensor,
                         G: int) -> torch.Tensor:
    """Per-group GRPO renorm over only the ``subset`` samples in each group.

    Subset samples get ``(r - mean_subset) / (std_subset + eps)``; every other
    sample (and every sample of an empty-subset group) gets 0.
    """
    grouped = rewards.view(-1, G)
    sub_g = subset.view(-1, G)
    out = torch.zeros_like(grouped)
    for i in range(grouped.shape[0]):
        m = sub_g[i]
        if m.sum() > 0:
            r = grouped[i][m]
            # A single-element subset has no variance; the unbiased std() would
            # be NaN (0/0), so the advantage is 0 by definition.
            std = r.std() if r.numel() > 1 else r.new_zeros(())
            out[i][m] = (r - r.mean()) / (std + _EPS)
    return out.view(-1)


def _full_group_renorm(rewards: torch.Tensor, G: int) -> torch.Tensor:
    """Per-group GRPO renorm over the full group."""
    grouped = rewards.view(-1, G)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True)
    return ((grouped - mean) / (std + _EPS)).view(-1)


def _baseline_nonflagged_var_all(rewards: torch.Tensor, nonflagged: torch.Tensor,
                                 G: int) -> torch.Tensor:
    """Per-group renorm with the baseline from non-flagged samples and the scale from the whole group.

    Every sample gets ``(r - baseline) / (std_all + eps)``, so the control variate reflects honest
    performance. A group with no non-flagged sample uses the full-group mean.
    """
    grouped = rewards.view(-1, G)
    nf = nonflagged.view(-1, G)
    out = torch.zeros_like(grouped)
    for i in range(grouped.shape[0]):
        r = grouped[i]
        m = nf[i]
        baseline = r[m].mean() if bool(m.any()) else r.mean()
        std = r.std() if r.numel() > 1 else r.new_zeros(())
        out[i] = (r - baseline) / (std + _EPS)
    return out.view(-1)




def compute_routed_advantages(
    *,
    raw_rewards: torch.Tensor,
    base_advantages: torch.Tensor,
    is_rh: torch.Tensor,
    is_coherence: torch.Tensor,
    penalty_baseline_raw_rewards: Optional[torch.Tensor],
    cfg: AdvConfig,
    is_attract: Optional[torch.Tensor] = None,
    attract_group_baseline: bool = False,
    attract_unflagged_baseline: bool = False,
) -> torch.Tensor:
    """Final per-sample advantages.

    Args:
        raw_rewards: pre-normalization reward sums, [n].
        base_advantages: stock per-group GRPO advantages, [n].
        is_rh: bool [n], samples the monitor flagged.
        is_coherence: bool [n], the coherence slice; all False without one.
        penalty_baseline_raw_rewards: the rewards the penalty baseline transforms, [n].
        is_attract: bool [n], rows whose reward is r * 1[flagged]. Without routing every group is then renormalized.
            With routing, the non-flagged baseline of a trigger group is 0, so flagged samples keep their reward
            and the rest get 0.
        attract_group_baseline: with routing, center a trigger group on its own mean instead, as without routing.
            Unflagged samples then get a negative advantage, and a group where every sample is flagged gets about 0.
        attract_unflagged_baseline: without routing, take every group's baseline from its unflagged samples, as with
            routing. The advantages are then the ones a routed attract run computes, with nothing routed.
    """
    G = cfg.num_generations
    advantages = base_advantages.clone()
    if is_attract is not None:
        raw_rewards = raw_rewards.masked_fill(is_attract & ~is_rh, 0.0)

    if cfg.gradient_routing_enabled:
        flagged_coherence = is_rh & is_coherence
        if flagged_coherence.any():
            if cfg.coherence_rh_mode == "penalty":
                penalized = raw_rewards.clone()
                penalized[flagged_coherence] -= cfg.coherence_rh_penalty
                renormed = _full_group_renorm(penalized, G)
                coherence_groups = is_coherence.view(-1, G).all(dim=1).repeat_interleave(G)
                advantages[coherence_groups] = renormed[coherence_groups]
            elif cfg.coherence_rh_mode == "filter":
                advantages[flagged_coherence] = 0.0
            elif cfg.coherence_rh_mode != "none":
                raise ValueError(f"Unknown coherence_rh_mode: {cfg.coherence_rh_mode!r}")
        # One vector shared by both adapters: baseline over non-flagged samples, scale over the whole group.
        # Coherence groups keep the handling above.
        routing_groups = (~is_coherence.view(-1, G).all(dim=1)).repeat_interleave(G)
        balanced = _baseline_nonflagged_var_all(raw_rewards, ~is_rh, G)
        advantages[routing_groups] = balanced[routing_groups]
        if is_attract is not None and attract_group_baseline:
            advantages[is_attract] = _full_group_renorm(raw_rewards, G)[is_attract]
    elif cfg.reward_penalty_baseline:
        penalized = penalty_baseline_raw_rewards.clone()
        penalized[is_rh] = 0.0 if cfg.reward_penalty_amount is None else penalized[is_rh] - cfg.reward_penalty_amount
        advantages = _full_group_renorm(penalized, G)
    elif cfg.filter_baseline:
        # Survivors are advantaged against each other, not against the hacks being dropped.
        advantages = _subset_group_renorm(raw_rewards, ~is_rh, G)
    elif is_attract is not None:
        advantages = _full_group_renorm(raw_rewards, G)
        if attract_unflagged_baseline:
            main_groups = (~is_coherence.view(-1, G).all(dim=1)).repeat_interleave(G)
            advantages[main_groups] = _baseline_nonflagged_var_all(raw_rewards, ~is_rh, G)[main_groups]

    return advantages
