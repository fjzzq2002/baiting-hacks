"""Split-moment AdamW: second moment (v) from the pre-routing gradient, first
moment (m) from the routed gradient.

Standard Adam maintains, per parameter,
    m_t = β1·m_{t-1} + (1-β1)·g_t            (first moment / momentum)
    v_t = β2·v_{t-1} + (1-β2)·g_t²           (second moment / variance estimate)
and steps along m̂ / (√v̂ + eps). ``SplitMomentAdamW`` feeds the two moments from
TWO different gradients of the same step:
    m  <- p.grad                  (the ROUTED gradient — gate-masked + redistributed)
    v  <- p._pre_routing_grad     (the NATURAL/pre-routing gradient — every sample
                                   reaches both adapters at scale 1)

Rationale (gradient-routing experiment): the update *direction* should follow the
routing (m from the routed grad), while the per-parameter *scale* (v) should
reflect the natural gradient magnitude rather than the post-routing one. The two
gradients are produced in a single backward by the fused path + the
``PreRoutingGradAccumulator`` capture (gradient_routing.py, trainer/update.py).

A tagged (routing) parameter with no captured ``_pre_routing_grad`` is a hard error. Untagged parameters use ``p.grad``
for both moments, like ordinary AdamW.

Decoupled weight decay (AdamW), matching torch.optim.AdamW. amsgrad / maximize /
fused / capturable are not supported (asserted off).
"""

import math

import torch
from torch.optim import AdamW


def clip_pre_routing_grads_(param_groups, max_norm, total_norm):
    """Scale every ``p._pre_routing_grad`` in ``param_groups`` by the same
    coefficient ``torch.nn.utils.clip_grad_norm_`` applies to ``.grad``:
    ``clip_coef = min(1, max_norm / (total_norm + 1e-6))``, where ``total_norm``
    is the pre-clip total norm that call returned. No-op when nothing clips
    (coef >= 1) or args are missing. Returns the applied coef (or None).

    Keeps gradient clipping a single shared event across Adam's two moments when
    they are fed from two different gradients (m <- .grad, v <- _pre_routing_grad).
    """
    if max_norm is None or total_norm is None:
        return None
    clip_coef = float(max_norm) / (float(total_norm) + 1e-6)
    if clip_coef >= 1.0:
        return None
    for group in param_groups:
        for p in group["params"]:
            b = getattr(p, "_pre_routing_grad", None)
            if b is not None:
                b.mul_(clip_coef)
    return clip_coef


class SplitMomentAdamW(AdamW):
    """AdamW with v from the pre-routing grad and m from the routed grad, plus per-window settings from `set_window`:

    - Participation factor c_A per role: it scales the v-source, so the adapter's per-window step is multiplied by
      1/c_A. With retain-only coherence in the batch, c_F = N / N_routing makes forget step at retain's per-example
      rate, instead of Adam normalizing its smaller accumulated v back up to the full rate.
    - Freeze: an adapter with no examples this window (`active[role]=False`) is skipped entirely: m, v, step count and
      weight decay. Its bias correction stays right, because `state["step"]` only advances on participating windows.
    - Step bound `w_max` on every coordinate of a tagged group.

    Groups carry a `graft_role` tag ("retain" or "forget").
    """

    def set_window(self, participation, active, *, w_max=None):
        """Set {role: c_A}, {role: bool} and the step bound for the next `step()`, which consumes them.

        `w_max` is a per-coordinate trust region: the denominator is floored at |m̂|/w_max, so no coordinate's step
        exceeds w_max * lr. This does nothing wherever the step is already within the bound.
        """
        self._window = {"c": participation, "active": active, "w_max": w_max}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        window = getattr(self, "_window", None)
        w_max = window["w_max"] if window is not None else None
        # Step diagnostics, accumulated across routing params:
        realized_p999 = 0.0      # 99.9th-pct PRE-clamp per-coord step / lr
        realized_abs_max = 0.0   # raw max PRE-clamp per-coord step / lr
        n_clamped = 0            # routing coords the clamp bit this window
        n_coords = 0             # total routing coords
        for group in self.param_groups:
            assert not group.get("amsgrad", False), "SplitMomentAdamW: amsgrad unsupported"
            assert not group.get("maximize", False), "SplitMomentAdamW: maximize unsupported"
            assert not group.get("fused", False) and not group.get("foreach", False), (
                "SplitMomentAdamW: fused/foreach unsupported (use the plain loop)")
            role = group.get("graft_role")
            # Per-role participation + freeze. Default c=1 / always-active for
            # untagged groups or when no window was set (= no participation).
            if window is not None and role is not None:
                if not window["active"].get(role, True):
                    continue  # FREEZE this adapter: skip m / v / step counter / wd
                c = float(window["c"].get(role, 1.0))
            else:
                c = 1.0
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                g_m = p.grad  # first moment <- routed gradient
                if g_m is None:
                    continue
                assert not g_m.is_sparse, "SplitMomentAdamW: sparse grads unsupported"
                # second moment <- pre-routing (natural) gradient.
                g_v = getattr(p, "_pre_routing_grad", None)
                if g_v is None:
                    assert role is None, (
                        "SplitMomentAdamW: a routing param has no captured "
                        "_pre_routing_grad — the pre-routing capture did not fire. "
                        "Refusing to silently fall back to plain AdamW (v<-.grad) "
                        "under routing (no silent fallback).")
                    g_v = g_m
                if c != 1.0:
                    g_v = g_v * c  # participation: scale the v-source (squared below)
                block = getattr(p, "_v_block_scale", None)
                if block is not None:
                    g_v = g_v * block  # a block of this parameter that saw fewer examples than the rest of its role

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                # AdamW: decoupled weight decay applied directly to the parameter.
                if wd != 0:
                    p.mul_(1.0 - lr * wd)

                exp_avg.mul_(beta1).add_(g_m, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_v, g_v, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1 ** t
                bias_correction2 = 1.0 - beta2 ** t
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / bias_correction1
                # The realized step in units of lr is |m̂| / (√v̂ + eps). The adapter-size guard bounds the mask weight,
                # not this: v can cancel across opposite-sign tokens, leaving a tiny v̂ under a large masked m̂. Both
                # the 99.9th percentile and the max are logged, since one near-zero v̂ can spike the max.
                if w_max is not None and role is not None:
                    mhat_abs = exp_avg.abs() / bias_correction1
                    realized = mhat_abs / denom            # PRE-clamp per-coord step / lr
                    rflat = realized.flatten().float()
                    q = (torch.quantile(rflat, 0.999).item() if rflat.numel() > 1
                         else float(rflat.item()))
                    realized_p999 = max(realized_p999, q)
                    realized_abs_max = max(realized_abs_max, float(realized.max().item()))
                    floor = mhat_abs / w_max
                    bit = floor > denom
                    n_clamped += int(bit.sum().item())
                    n_coords += bit.numel()
                    denom = torch.maximum(denom, floor)
                p.addcdiv_(exp_avg, denom, value=-step_size)

        if w_max is not None:
            self._last_realized_max = realized_p999          # 99.9th pct (PRE-clamp)
            self._last_realized_abs_max = realized_abs_max   # raw max (PRE-clamp)
            self._last_frac_clamped = (n_clamped / n_coords) if n_coords else 0.0
        self._window = None  # consumed; the next window must set_window() again
        return loss
