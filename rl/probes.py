"""A linear probe on activations, fit once on fixed trajectories and read on the initial model.

The initial model is the model RL starts from, with the adapters doing nothing. Fitting happens offline, by
tools/fit_probe.py, on no-intervention trajectories only, so a probe is never fit on rollouts from the run it monitors.
"""

from dataclasses import asdict, dataclass

import torch
from torch.nn.functional import binary_cross_entropy_with_logits

from gradient_routing import disabled_dual_adapters
from packing import _pack_by_tokens, _pack_for_forward

LAYER = 16
L2 = 1e-3
NEGATIVE_QUANTILE = 0.99
STD_FLOOR = 1e-4


class _ReachedLayer(Exception):
    pass


def auroc(scores, labels):
    """Probability that a positive outscores a negative, with ties counted half. None without both classes."""
    positives, negatives = scores[labels], scores[~labels]
    if not len(positives) or not len(negatives):
        return None
    order = torch.cat([positives, negatives]).sort().values
    low = torch.searchsorted(order, positives, right=False)
    high = torch.searchsorted(order, positives, right=True)
    n = len(positives)
    # Twice the rank sum is an integer, and a float32 sum of ranks drifts past 1 on tens of thousands of rows.
    return (int((low + high + 1).sum()) / 2 - n * (n + 1) / 2) / (n * len(negatives))


def pad_rows(prompts, completions, pad_id, device):
    """Token-id lists as the tensors a rollout holds: prompts padded on the left, completions on the right."""
    width_p, width_c = max(map(len, prompts)), max(max(map(len, completions)), 1)
    rows = {"prompt_ids": torch.full((len(prompts), width_p), pad_id, dtype=torch.long, device=device),
            "prompt_mask": torch.zeros(len(prompts), width_p, dtype=torch.long, device=device),
            "completion_ids": torch.full((len(prompts), width_c), pad_id, dtype=torch.long, device=device),
            "completion_mask": torch.zeros(len(prompts), width_c, dtype=torch.long, device=device)}
    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        if prompt:
            rows["prompt_ids"][i, -len(prompt):] = torch.tensor(prompt, device=device)
            rows["prompt_mask"][i, -len(prompt):] = 1
        if completion:
            rows["completion_ids"][i, :len(completion)] = torch.tensor(completion, device=device)
            rows["completion_mask"][i, :len(completion)] = 1
    return rows


def read_layers(model, rows, layers, max_tokens):
    """Decoder blocks' outputs, averaged over completion tokens, from the update's packed forward. One pass, stopped early.

    `rows` is shaped like a rollout (see `pad_rows`). Returns {layer: float32 (N, H)}, with zeros for an empty completion.
    """
    captured = {}
    deepest = max(layers)

    def capture(layer):
        def hook(_, __, out):
            captured[layer] = out[0] if isinstance(out, tuple) else out
            if layer == deepest:
                raise _ReachedLayer  # the blocks after this one would only be thrown away
        return hook

    hooks = [model.model.layers[layer - 1].register_forward_hook(capture(layer)) for layer in layers]
    counts = (rows["prompt_mask"].sum(1) + rows["completion_mask"].sum(1)).tolist()
    features = {}
    try:
        for indices in _pack_by_tokens(counts, list(range(len(counts))), max_tokens):
            packed = _pack_for_forward(rows, indices)
            try:
                with torch.no_grad():
                    model.model(input_ids=packed["packed_input_ids"], position_ids=packed["packed_position_ids"],
                                use_cache=False)
            except _ReachedLayer:
                pass
            for layer in layers:
                hidden = captured[layer][0].float()
                if layer not in features:
                    features[layer] = torch.zeros(len(counts), hidden.shape[-1], device=hidden.device)
                start = 0
                for index, (prompt_len, completion_len) in zip(indices, packed["seq_boundaries"]):
                    if completion_len:
                        features[layer][index] = hidden[start + prompt_len:start + prompt_len + completion_len].mean(0)
                    start += prompt_len + completion_len
    finally:
        for hook in hooks:
            hook.remove()
    return features


def read_activations(model, rows, layer, max_tokens):
    """One decoder block's output, as in `read_layers`."""
    return read_layers(model, rows, [layer], max_tokens)[layer]


@dataclass
class Probe:
    """Standardize, then a linear score. A row is flagged when its logit exceeds `threshold`."""
    layer: int
    mean: torch.Tensor
    std: torch.Tensor
    weights: torch.Tensor
    bias: float
    threshold: float

    def logits(self, features):
        return ((features - self.mean) / self.std) @ self.weights + self.bias

    def to(self, device):
        return Probe(self.layer, self.mean.to(device), self.std.to(device), self.weights.to(device), self.bias,
                     self.threshold)

    def save(self, path):
        torch.save(asdict(self), path)

    @classmethod
    def load(cls, path):
        return cls(**torch.load(path, map_location="cpu", weights_only=True))


def fit_probe(features, flagged, negative, layer=LAYER, l2=L2, quantile=NEGATIVE_QUANTILE, max_iter=200):
    """L2 logistic regression on monitor flags, with the threshold at a quantile of the negatives' logits.

    `flagged` and `negative` are boolean masks over `features`. Rows in neither are not used.
    """
    use = flagged | negative
    x, y = features[use], flagged[use].float()
    mean, std = x.mean(0), x.std(0, unbiased=False) + STD_FLOOR
    z = (x - mean) / std
    weights = torch.zeros(z.shape[1], device=z.device, requires_grad=True)
    bias = torch.zeros((), device=z.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([weights, bias], lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = binary_cross_entropy_with_logits(z @ weights + bias, y) + 0.5 * l2 * (weights * weights).sum()
        loss.backward()
        return loss

    with torch.enable_grad():
        optimizer.step(closure)
    probe = Probe(layer, mean.cpu(), std.cpu(), weights.detach().cpu(), float(bias.detach()), 0.0)
    probe.threshold = float(torch.quantile(probe.to(features.device).logits(features[negative]), quantile))
    return probe


class TrainingProbe:
    """Flags rollout rows during training. It holds a fixed probe and reads the training model with its adapters off,
    so what it sees does not follow the policy and no second model is loaded."""

    def __init__(self, path, max_tokens):
        self.probe = Probe.load(path)
        self.max_tokens = max_tokens

    def flags(self, model, rows):
        with disabled_dual_adapters(model):
            features = read_activations(model, rows, self.probe.layer, self.max_tokens)
        return (self.probe.to(features.device).logits(features) > self.probe.threshold).tolist()
