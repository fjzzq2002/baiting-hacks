"""Dual adapter implementation for gradient routing.

DualMLPAdapter adds two mini-SwiGLU networks to each MLP block (extra neurons in the intermediate dim).

Naming convention:
  - "retain": adapter we keep at inference (gradients zeroed on bad samples)
  - "forget": adapter we ablate at inference (absorbs reward-hacking behavior)
"""

import math
from contextlib import contextmanager

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Fused gradient routing (single forward + single backward over a heterogeneous
# microbatch). When active, dual adapters carry coherence + good + bad samples in
# one packed forward, routing each sample's gradient to the correct adapter(s) —
# reproducing the per-pass register_hook scheme of the homogeneous-microbatch
# path in a single pass. Per token we control two things:
#
#   forget_fwd_scale  (1,T,1): forward multiplier on the forget-adapter output
#       (0 for coherence tokens [forget off], train_forget_scale for routing
#       tokens). Supersedes the scalar module forget_scale while active.
#   {retain,forget}_grad_mask (1,T,1): per-token PARAMETER-gradient gate for that
#       adapter (1 = this token trains the adapter, 0 = it does not).
#
# Critical subtlety — why a plain activation-grad hook is wrong. Stock routing
# uses parameter register_hooks, which zero an adapter's OWN param gradient on a
# masked pass but leave that adapter's contribution to the *downstream*
# activation gradient (to lower layers) intact. Simply masking the adapter
# output's gradient (g <- g*m) would also remove the adapter's Jacobian from the
# gradient flowing to lower layers — diverging from stock for multi-layer
# adapters. So we gate the PARAMETER gradient only, via a stop-gradient
# decomposition that leaves the input gradient untouched (see _fused_decouple).
# All adapter modules in a packed forward share the same (1,T,H) token layout, so
# one set of per-token tensors applies to every layer. While inactive, adapters
# behave exactly as before (zero impact on existing runs).
# ---------------------------------------------------------------------------

_FUSED_ROUTING = {"active": False}


def set_fused_routing(forget_fwd_scale, retain_grad_mask, forget_grad_mask, forget2_fwd_scale=None, forget2_grad_mask=None):
    """Install per-token fused-routing tensors for the next packed forward(s).
    All are (1, T, 1) float tensors; see the module-level note above. The forget2 pair is for adapters with a
    forget2 block: its forward switch (it is also off wherever forget is) and its own gradient weight."""
    _FUSED_ROUTING["active"] = True
    _FUSED_ROUTING["forget_fwd_scale"] = forget_fwd_scale
    _FUSED_ROUTING["retain_grad_mask"] = retain_grad_mask
    _FUSED_ROUTING["forget_grad_mask"] = forget_grad_mask
    _FUSED_ROUTING["forget2_fwd_scale"] = forget2_fwd_scale
    _FUSED_ROUTING["forget2_grad_mask"] = forget2_grad_mask


def clear_fused_routing():
    _FUSED_ROUTING["active"] = False
    for key in ("forget_fwd_scale", "retain_grad_mask", "forget_grad_mask", "forget2_fwd_scale", "forget2_grad_mask"):
        _FUSED_ROUTING.pop(key, None)




def _fused_decouple(g_full, g_xdetached, mask):
    """Gate an adapter output's PARAMETER gradient per token without touching its
    input gradient.

    g_full = g(x; theta)            (full autograd graph)
    g_xdetached = g(x.detach(); theta)   (same value; gradient flows to theta only)
    mask = (1,T,1) per-token gate (1 keep, 0 drop) for theta's gradient.

    Returns a tensor with value g_full, parameter-gradient `mask * dg/dtheta`, and
    input-gradient the full `dg/dx`. Derivation:
        out = g_full - (1-m)*g_xdetached + (1-m)*g_full.detach()
      value:  g_full - (1-m)g + (1-m)g = g_full
      d/dtheta: dg - (1-m)dg + 0 = m*dg
      d/dx:     dg/dx - 0 + 0 = dg/dx   (g_xdetached has no x-grad; .detach() none)
    For m==1 tokens this is exactly g_full; for m==0 tokens the param gradient
    vanishes while the input gradient (downstream coupling) is preserved.
    """
    mc = mask.to(g_full.dtype)
    return g_full - (1.0 - mc) * g_xdetached + (1.0 - mc) * g_full.detach()




class DualMLPAdapter(nn.Module):
    """Wraps a frozen MLP block with two parallel mini-SwiGLU adapter networks.

    Each adapter has n_neurons intermediate dim: gate/up use kaiming init, down uses zeros
    (so adapter output starts at zero).

    Forward: base_out + retain_scale * down_retain(SiLU(gate_retain(x)) * up_retain(x))
                       + forget_scale * down_forget(SiLU(gate_forget(x)) * up_forget(x))

    With `forget2_neurons` > 0, the last that many forget neurons are the forget2 block: a second forget adapter that
    is on only where its own switch says so, and never where forget is off. Its weights are part of the forget
    matrices, so checkpoints and the sampler see one forget adapter. Outside fused routing its switch is
    `forget2_scale`, which defaults to off.
    """

    def __init__(self, base_mlp, hidden_size, retain_neurons, forget_neurons,
                 retain_scale=1.0, forget_scale=1.0, forget2_neurons=0):
        super().__init__()
        assert 0 <= forget2_neurons < max(forget_neurons, 1), (forget_neurons, forget2_neurons)
        self.base_mlp = base_mlp
        self.retain_neurons = retain_neurons
        self.forget_neurons = forget_neurons
        self.forget2_neurons = forget2_neurons
        self.retain_scale = retain_scale
        self.forget_scale = forget_scale
        self.forget2_scale = 0.0

        first_param = next(base_mlp.parameters())
        dtype = first_param.dtype
        device = first_param.device
        self.act = nn.SiLU()

        # Freeze base MLP
        for p in self.base_mlp.parameters():
            p.requires_grad = False

        # Retain adapter
        if retain_neurons > 0:
            self.gate_retain = nn.Linear(hidden_size, retain_neurons, bias=False, dtype=dtype, device=device)
            self.up_retain = nn.Linear(hidden_size, retain_neurons, bias=False, dtype=dtype, device=device)
            self.down_retain = nn.Linear(retain_neurons, hidden_size, bias=False, dtype=dtype, device=device)
            nn.init.kaiming_uniform_(self.gate_retain.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.up_retain.weight, a=math.sqrt(5))
            nn.init.zeros_(self.down_retain.weight)
        else:
            self.gate_retain = self.up_retain = self.down_retain = None

        # Forget adapter
        if forget_neurons > 0:
            self.gate_forget = nn.Linear(hidden_size, forget_neurons, bias=False, dtype=dtype, device=device)
            self.up_forget = nn.Linear(hidden_size, forget_neurons, bias=False, dtype=dtype, device=device)
            self.down_forget = nn.Linear(forget_neurons, hidden_size, bias=False, dtype=dtype, device=device)
            nn.init.kaiming_uniform_(self.gate_forget.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.up_forget.weight, a=math.sqrt(5))
            nn.init.zeros_(self.down_forget.weight)
            if forget2_neurons:
                # Which axis indexes neurons, for per-block optimizer settings.
                self.gate_forget.weight._forget_neuron_axis = self.up_forget.weight._forget_neuron_axis = 0
                self.down_forget.weight._forget_neuron_axis = 1
        else:
            self.gate_forget = self.up_forget = self.down_forget = None

        # PPS steering (default OFF): fixed vector added to this wrapper's
        # output — which flows into the post-block residual add — whenever
        # steer_alpha != 0. `steer` is a NON-persistent buffer: excluded from
        # named_parameters() (so it is never optimized, never saved by the
        # adapter-only checkpoint filter, and never picked up by
        # collect_routing_params) AND excluded from state_dict() (full
        # checkpoints stay byte-identical to pre-PPS code; old checkpoints
        # load without missing-key noise). Kept fp32 on the params' device;
        # cast to the activation dtype at add time (contract).
        self.register_buffer(
            "steer",
            torch.zeros(hidden_size, dtype=torch.float32, device=device),
            persistent=False,
        )
        self.steer_alpha = 0.0  # plain python float; NOT a Parameter/tensor
        # Decoder-layer index of the wrapped MLP; written by apply_dual_mlp.
        # set_pps_steering needs it to match this adapter against a
        # {layer: vector} dict.
        self._pps_layer_idx = None

    def _retain_core(self, x):
        return self.down_retain(self.act(self.gate_retain(x)) * self.up_retain(x))

    def _forget_core(self, x):
        return self.down_forget(self.act(self.gate_forget(x)) * self.up_forget(x))

    def _forget_block(self, x, start, stop):
        """The forget adapter restricted to neurons [start, stop)."""
        linear = nn.functional.linear
        hidden = self.act(linear(x, self.gate_forget.weight[start:stop])) * linear(x, self.up_forget.weight[start:stop])
        return linear(hidden, self.down_forget.weight[:, start:stop])

    def _forget_main(self, x):
        return self._forget_block(x, 0, self.forget_neurons - self.forget2_neurons)

    def _forget2_core(self, x):
        return self._forget_block(x, self.forget_neurons - self.forget2_neurons, self.forget_neurons)

    def forward(self, x):
        base_out = self.base_mlp(x)
        fused = _FUSED_ROUTING["active"]

        if not fused:
            if self.gate_retain is not None:
                retain_out = self._retain_core(x) * self.retain_scale
            else:
                retain_out = 0
            if self.gate_forget is None:
                forget_out = 0
            elif self.forget2_neurons:
                forget_out = (self._forget_main(x) + self._forget2_core(x) * self.forget2_scale) * self.forget_scale
            else:
                forget_out = self._forget_core(x) * self.forget_scale
            out = base_out + retain_out + forget_out
            if self.steer_alpha != 0.0:
                # PPS steering: add to the wrapper output (= the post-block
                # residual contribution), every token, fp32 vector cast to the
                # activation dtype at add time. Python-float guard: when
                # steer_alpha == 0.0 (the default) this branch never executes,
                # so non-PPS runs are byte-identical to pre-PPS code.
                out = out + self.steer_alpha * self.steer.to(out.dtype)
            return out

        # Fused routing: per-token parameter-gradient gating (forward value and
        # input gradient unchanged) + per-token forward forget-scale.
        xd = x.detach()
        if self.gate_retain is not None:
            retain_out = _fused_decouple(
                self._retain_core(x), self._retain_core(xd),
                _FUSED_ROUTING["retain_grad_mask"]) * self.retain_scale
        else:
            retain_out = 0
        if self.gate_forget is None:
            forget_out = 0
        elif self.forget2_neurons:
            forget_fwd = _FUSED_ROUTING["forget_fwd_scale"].to(base_out.dtype)
            forget_out = _fused_decouple(self._forget_main(x), self._forget_main(xd), _FUSED_ROUTING["forget_grad_mask"])
            forget2_out = _fused_decouple(self._forget2_core(x), self._forget2_core(xd), _FUSED_ROUTING["forget2_grad_mask"])
            forget_out = (forget_out + forget2_out * _FUSED_ROUTING["forget2_fwd_scale"].to(base_out.dtype)) * forget_fwd
        else:
            forget_out = _fused_decouple(
                self._forget_core(x), self._forget_core(xd),
                _FUSED_ROUTING["forget_grad_mask"]) \
                * _FUSED_ROUTING["forget_fwd_scale"].to(base_out.dtype)
        out = base_out + retain_out + forget_out
        if self.steer_alpha != 0.0:
            # PPS steering (see non-fused branch). PPS runs are single-policy
            # (routing_mode=none) so this never fires in fused/GR runs; the
            # guard leaves the fused-path math untouched when off.
            out = out + self.steer_alpha * self.steer.to(out.dtype)
        return out

    def get_retain_params(self):
        """Get retain adapter parameters."""
        if self.gate_retain is not None:
            return list(self.gate_retain.parameters()) + list(self.up_retain.parameters()) + list(self.down_retain.parameters())
        return []

    def get_forget_params(self):
        """Get forget adapter parameters."""
        if self.gate_forget is not None:
            return list(self.gate_forget.parameters()) + list(self.up_forget.parameters()) + list(self.down_forget.parameters())
        return []

    def natural_adapter_output(self, x, forget_fwd_scale, forget2_fwd_scale=None):
        """The adapter contribution (retain_out + forget_out) with NO routing
        decouple — the natural / pre-routing output. Matches forward()'s fused
        branch values: retain uses ``retain_scale``; forget uses ``forget_fwd_scale``
        (the fused branch substitutes the forward forget-scale for ``forget_scale``).
        Used by PreRoutingGradAccumulator to recover the natural gradient via
        autograd through the mini-SwiGLU (no hand-derived backward)."""
        out = None
        if self.gate_retain is not None:
            out = self._retain_core(x) * self.retain_scale
        if self.gate_forget is not None:
            if self.forget2_neurons:
                f = (self._forget_main(x) + self._forget2_core(x) * forget2_fwd_scale) * forget_fwd_scale
            else:
                f = self._forget_core(x) * forget_fwd_scale
            out = f if out is None else out + f
        return out


ALL_PROJECTIONS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
ATTENTION_PROJECTIONS = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_PROJECTIONS = ["gate_proj", "up_proj", "down_proj"]


# ---------------------------------------------------------------------------
# Shared model discovery utilities (used by both HF and vLLM paths)
# ---------------------------------------------------------------------------

def compute_layer_indices(num_layers: int, layer_start: float = 0.0,
                          layer_end: float = 1.0, layer_stride: int = 1) -> list[int]:
    """Convert fractional layer range to concrete indices."""
    start_idx = int(num_layers * layer_start)
    end_idx = int(num_layers * layer_end)
    return list(range(start_idx, end_idx, layer_stride))


def find_mlp_modules(model, layer_indices: list[int] | None = None):
    """Discover MLP submodules via named_modules() walk.

    Finds paths matching *.{int}.mlp — the standard structure for LLaMA,
    Qwen, Mistral, Gemma, and their vLLM counterparts.

    Args:
        model: Any transformer model (HF or vLLM).
        layer_indices: Concrete layer indices to select, or None for all.

    Returns:
        List of (layer_idx, path, parent_module, mlp_module) tuples, sorted
        by layer index.
    """
    if layer_indices is not None:
        target = set(layer_indices)
    else:
        target = None  # accept all

    entries = []
    for name, module in model.named_modules():
        parts = name.split(".")
        if parts[-1] != "mlp" or len(parts) < 2 or not parts[-2].isdigit():
            continue
        layer_idx = int(parts[-2])
        if target is not None and layer_idx not in target:
            continue
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        entries.append((layer_idx, name, parent, module))

    assert entries, (
        f"find_mlp_modules found no modules matching *.{{int}}.mlp pattern"
        + (f" for layer indices {sorted(target)}" if target else "")
    )
    return sorted(entries)








def apply_dual_mlp(model, retain_neurons, forget_neurons,
                   layer_start=0.0, layer_end=1.0, layer_stride=1, forget2_neurons=0):
    """Replace MLP modules with DualMLPAdapter.

    Uses find_mlp_modules() for architecture-agnostic discovery.
    Freezes all base model parameters; only adapter params are trainable.
    Returns list of modified layer indices.
    """
    for param in model.parameters():
        param.requires_grad = False

    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    layer_indices = compute_layer_indices(num_layers, layer_start, layer_end, layer_stride)
    mlp_entries = find_mlp_modules(model, layer_indices)

    modified = []
    for layer_idx, path, parent, base_mlp in mlp_entries:
        adapter = DualMLPAdapter(base_mlp, hidden_size, retain_neurons, forget_neurons, forget2_neurons=forget2_neurons)
        # PPS steering: each adapter records which decoder layer it wraps so
        # set_pps_steering can match it against a {layer: vector} dict.
        adapter._pps_layer_idx = layer_idx
        setattr(parent, "mlp", adapter)
        modified.append(layer_idx)

    return modified


_DUAL_ADAPTER_TYPES = (DualMLPAdapter,)


def has_dual_adapters(model):
    """Check if model has any dual adapter modules."""
    return any(isinstance(m, _DUAL_ADAPTER_TYPES) for m in model.modules())


def set_scales(model, retain_scale: float = 1.0, forget_scale: float = 1.0):
    """Set retain/forget adapter scales for inference-time ablation."""
    for module in model.modules():
        if isinstance(module, _DUAL_ADAPTER_TYPES):
            module.retain_scale = retain_scale
            module.forget_scale = forget_scale


def set_pps_steering(model, layer_to_vec: dict, alpha: float):
    """Write PPS steering vectors into every DualMLPAdapter (mirrors set_scales).

    layer_to_vec maps decoder-layer index -> steering vector (hidden_size,).
    Adapters whose layer appears in the dict get (vector, alpha); every other
    adapter gets (zeros, 0.0). ``layer_to_vec={}`` / ``alpha=0.0`` => steering
    fully OFF. Steering rides the wrapped-MLP output, which is the post-block
    residual-stream contribution.
    """
    adapters = [m for m in model.modules() if isinstance(m, DualMLPAdapter)]
    assert adapters, "set_pps_steering: model has no DualMLPAdapter modules"
    adapted = set()
    with torch.no_grad():
        for module in adapters:
            idx = module._pps_layer_idx
            assert idx is not None, (
                "set_pps_steering: adapter has no layer index; adapters must be "
                "installed via apply_dual_mlp")
            adapted.add(idx)
            vec = layer_to_vec.get(idx)
            if vec is not None:
                assert vec.shape == module.steer.shape, (
                    f"steering vector for layer {idx} has shape "
                    f"{tuple(vec.shape)}, expected {tuple(module.steer.shape)}")
                module.steer.copy_(vec)
                module.steer_alpha = float(alpha)
            else:
                module.steer.zero_()
                module.steer_alpha = 0.0
    missing = set(layer_to_vec) - adapted
    assert not missing, (
        f"set_pps_steering: layers {sorted(missing)} have steering vectors but "
        f"no DualMLPAdapter (adapted layers: {sorted(adapted)})")


@contextmanager
def disabled_dual_adapters(model):
    """Temporarily zero all adapter scales, restore on exit.

    With both scales at 0, every adapter forward reduces to the frozen
    base layer output — so a forward pass through `model` under this context is
    equivalent to running the unadaptered base model. Used to compute reference
    logprobs without instantiating a separate ref model copy.

    PPS steering is neutralized too (steer_alpha saved / set to 0.0 / restored)
    so the KL-reference forward is the pristine UNSTEERED base, not base+steer.
    """
    saved = []
    for module in model.modules():
        if isinstance(module, _DUAL_ADAPTER_TYPES):
            saved.append((module, module.retain_scale, module.forget_scale,
                          getattr(module, "steer_alpha", None)))
            module.retain_scale = 0.0
            module.forget_scale = 0.0
            if isinstance(module, DualMLPAdapter):
                module.steer_alpha = 0.0
    try:
        yield
    finally:
        for module, r, f, sa in saved:
            module.retain_scale = r
            module.forget_scale = f
            if sa is not None:
                module.steer_alpha = sa


def collect_routing_params(model):
    """Return (retain params, forget params) in model order, so optimizer state lines up across processes.

    Retain is everything trainable that is not forget. The base model is frozen, so that is each adapter's retain side.
    """
    forget = [p for m in model.modules() if isinstance(m, _DUAL_ADAPTER_TYPES) for p in m.get_forget_params()]
    forget_ids = {id(p) for p in forget}
    retain = [p for p in model.parameters() if p.requires_grad and id(p) not in forget_ids]
    return retain, forget


class PreRoutingGradAccumulator:
    """Accumulate the *natural* (pre-routing) parameter gradient of the dual
    adapters into per-parameter ``._pre_routing_grad`` buffers, during the same
    backward that the decoupled fused path uses to put the *routed* gradient in
    ``.grad``.

    Purpose: split-moment Adam (see ``SplitMomentAdamW``). Adam's second moment
    (v) is built from the pre-routing gradient (both adapters see every sample at
    scale 1 — no gate-mask, no redistribution); the first moment (m) is built
    from ``.grad`` (the routed gradient). Both come from ONE base backward.

    How. We need a second, differently-weighted reduction of the same per-token
    gradient pieces autograd already folds into ``.grad`` (routed) — but with the
    routing masks set to 1 (natural). Autograd only gives the one accumulation, so
    we tap each adapter's input ``x`` (forward pre-hook) and output-grad
    ``g = dL/dy`` (full backward hook; ``g`` is the module-output grad, upstream of
    the decouple's parameter gating, hence *natural*). After each microbatch's
    backward, ``flush()`` re-runs a *natural* forward of just that adapter
    (``natural_adapter_output``, no decouple) and applies ``g`` via
    ``torch.autograd.grad`` to get the natural parameter gradient — autograd
    handles the mini-SwiGLU with no hand-derived backward. The
    re-forward is over the tiny adapter only (50–200× smaller than the base MLP),
    reusing the one expensive base backward. Grads reduce straight into a
    param-sized buffer (never per-sample). ``g`` already carries the per-microbatch
    loss scale (same ``backward(loss·scale)``), so ``_pre_routing_grad`` is
    normalized consistently with ``.grad``.

    The forward forget-scale (``forget_fwd_scale``) is part of the natural forward
    and is passed per-token into ``flush``: 1 on routing
    tokens, 0 on coherence tokens (forget is forward-off there). So a coherence
    sample contributes weight-1 to the retain adapter's ``v`` and 0 to the forget
    adapter's — matching its retain-only routed gradient.
    """

    def __init__(self, model):
        self._handles = []
        self._saved = {}            # id(module) -> x (per current microbatch)
        self._captures = []         # [(module, x, g)] for the current microbatch
        self._params = []           # adapter params whose buffers we own
        self._install(model)

    def _install(self, model):
        for m in model.modules():
            if isinstance(m, DualMLPAdapter):
                self._handles.append(m.register_forward_pre_hook(self._save))
                self._handles.append(m.register_full_backward_hook(self._bwd))
                self._params += m.get_retain_params() + m.get_forget_params()

    def reset(self):
        """Clear the pre-routing buffers. Call once before the step's backwards."""
        for p in self._params:
            p._pre_routing_grad = None

    def _save(self, mod, args):
        self._saved[id(mod)] = args[0].detach()

    def _bwd(self, mod, grad_input, grad_output):
        x = self._saved.pop(id(mod), None)
        g = grad_output[0]
        if x is None or g is None:
            return
        # Stash for flush(): can't run autograd inside a backward hook. Clone g —
        # the grad buffer may be freed/reused once this backward completes.
        self._captures.append((mod, x, g.detach().clone()))

    def flush(self, forget_fwd_scale, forget2_fwd_scale=None):
        """Add this microbatch's natural adapter gradient to `_pre_routing_grad` and clear its captures.

        `forget_fwd_scale` is the microbatch's per-token forward forget scale, shaped [1, T, 1]: 0 on coherence tokens,
        so forget's v excludes them. `forget2_fwd_scale` does the same for a forget2 block, which only sees the tokens
        it is on for. Call once after each microbatch's backward.
        """
        for mod, x, g in self._captures:
            params = mod.get_retain_params() + mod.get_forget_params()
            with torch.enable_grad():
                out = mod.natural_adapter_output(x, forget_fwd_scale, forget2_fwd_scale)
                grads = torch.autograd.grad(out, params, grad_outputs=g, retain_graph=False, allow_unused=True)
            for p, gp in zip(params, grads):
                if gp is None:
                    continue
                buffer = getattr(p, "_pre_routing_grad", None)
                p._pre_routing_grad = gp.detach() if buffer is None else buffer.add_(gp.detach())
        self._captures = []

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        self._saved.clear()
        self._captures = []
