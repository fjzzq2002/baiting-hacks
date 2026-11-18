"""Unit tests for dual adapter mechanics (DualMLPAdapter)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import pytest
from transformers import AutoModelForCausalLM

from gradient_routing import (
    DualMLPAdapter,
    apply_dual_mlp,
    set_scales,
    collect_routing_params,
    has_dual_adapters,
)

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"


@pytest.fixture
def base_model():
    """Load the toy-env model (CPU)."""
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    model.eval()
    return model


# --- DualMLPAdapter scale tests ---


def test_dual_mlp_forget_scale_zero_matches_retain_only(base_model):
    """With forget_scale=0, DualMLPAdapter output should equal base_mlp + retain only."""
    base_mlp = base_model.model.layers[0].mlp
    hidden_size = base_model.config.hidden_size
    adapter = DualMLPAdapter(base_mlp, hidden_size, retain_neurons=8, forget_neurons=8)

    # Set non-zero retain weights so there's something to compare
    with torch.no_grad():
        adapter.down_retain.weight.normal_(0, 0.01)

    x = torch.randn(2, 4, hidden_size)

    adapter.forget_scale = 0.0
    output = adapter(x)

    # Manually compute base + retain only
    base_out = base_mlp(x)
    retain_intermediate = adapter.act(adapter.gate_retain(x)) * adapter.up_retain(x)
    retain_out = adapter.down_retain(retain_intermediate)
    expected = base_out + retain_out

    assert torch.allclose(output, expected, atol=1e-5), \
        f"Max diff: {(output - expected).abs().max().item()}"


def test_dual_mlp_retain_scale_zero_matches_forget_only(base_model):
    """With retain_scale=0, DualMLPAdapter output should equal base_mlp + forget only."""
    base_mlp = base_model.model.layers[0].mlp
    hidden_size = base_model.config.hidden_size
    adapter = DualMLPAdapter(base_mlp, hidden_size, retain_neurons=8, forget_neurons=8)

    with torch.no_grad():
        adapter.down_forget.weight.normal_(0, 0.01)

    x = torch.randn(2, 4, hidden_size)

    adapter.retain_scale = 0.0
    output = adapter(x)

    base_out = base_mlp(x)
    forget_intermediate = adapter.act(adapter.gate_forget(x)) * adapter.up_forget(x)
    forget_out = adapter.down_forget(forget_intermediate)
    expected = base_out + forget_out

    assert torch.allclose(output, expected, atol=1e-5), \
        f"Max diff: {(output - expected).abs().max().item()}"


# --- Model-level tests ---


def test_set_scales_reaches_every_adapter():
    """set_scales sets both scales on all dual adapters."""
    layers = nn.ModuleList(DualMLPAdapter(nn.Linear(8, 8, bias=False), 8, 4, 4) for _ in range(3))
    set_scales(layers, retain_scale=0.25, forget_scale=0.5)
    assert all(m.retain_scale == 0.25 and m.forget_scale == 0.5 for m in layers)
