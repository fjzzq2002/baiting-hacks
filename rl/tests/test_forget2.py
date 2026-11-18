"""The forget2 block: routing weights, the trigger-group baseline, the adapter's forward and gradients, and the
weights the sampler receives."""
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from advantages import AdvConfig, compute_routed_advantages, forget2_routing  # noqa: E402
from gradient_routing import DualMLPAdapter, PreRoutingGradAccumulator, clear_fused_routing, set_fused_routing  # noqa: E402
from vllm_client import _pack_weights_flat  # noqa: E402
from vllm_utils import WEIGHT_KEYS  # noqa: E402

HIDDEN, RETAIN, FORGET, FORGET2 = 12, 4, 8, 4
MAIN = FORGET - FORGET2


def test_routing_weights_for_equal_blocks():
    classes = forget2_routing(16, 32, 16)
    assert classes["coherence"] == (0.0, 0.0, 1.0, 0.0, 0.0)
    assert classes["plain_good"] == (1.0, 0.0, 1.0, 1.0, 0.0)
    assert classes["plain_bad"] == (1.0, 0.0, 0.0, 2.0, 0.0)
    assert classes["trigger_bad"] == (1.0, 1.0, 0.0, 1.5, 1.5)
    assert classes["trigger_good"] == (1.0, 1.0, 0.0, 0.0, 3.0)


def test_routing_keeps_the_total_pressure():
    n_retain, n_forget, n_forget2 = 12, 20, 10
    main = n_forget - n_forget2
    classes = forget2_routing(n_retain, n_forget, n_forget2)
    _, _, retain, forget, forget2 = classes["plain_bad"]
    assert retain * n_retain + forget * main + forget2 * n_forget2 == pytest.approx(n_retain + main)
    for name in ("trigger_bad", "trigger_good"):
        _, _, retain, forget, forget2 = classes[name]
        assert retain * n_retain + forget * main + forget2 * n_forget2 == pytest.approx(n_retain + n_forget)


def test_routing_refuses_a_block_that_would_step_too_far():
    with pytest.raises(AssertionError):
        forget2_routing(60, 8, 4)


def advantages(group_baseline):
    config = AdvConfig(num_generations=4, gradient_routing_enabled=True, coherence_rh_mode="none", coherence_rh_penalty=0.0,
                       reward_penalty_baseline=False, reward_penalty_amount=None, filter_baseline=False)
    rewards = torch.tensor([1.0, 2.0, 3.0, 6.0, 1.0, 1.2, 0.9, 1.1])
    flags = torch.tensor([0, 0, 1, 1, 1, 1, 0, 0], dtype=torch.bool)
    attract = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.bool)
    return compute_routed_advantages(raw_rewards=rewards, base_advantages=torch.zeros(8), is_rh=flags,
                                     is_coherence=torch.zeros(8, dtype=torch.bool), penalty_baseline_raw_rewards=None,
                                     cfg=config, is_attract=attract, attract_group_baseline=group_baseline)


def test_trigger_group_is_centered_on_zero_by_default():
    paid = torch.tensor([1.0, 1.2, 0.0, 0.0])
    assert torch.allclose(advantages(False)[4:], paid / (paid.std() + 1e-4))


def test_trigger_group_is_centered_on_its_own_mean_when_asked():
    out = advantages(True)
    paid = torch.tensor([1.0, 1.2, 0.0, 0.0])
    assert torch.allclose(out[4:], (paid - paid.mean()) / (paid.std() + 1e-4))
    assert torch.equal(out[:4], advantages(False)[:4])  # plain groups keep the unflagged baseline


def test_unrouted_attract_can_take_the_baseline_from_unflagged_samples():
    config = AdvConfig(num_generations=4, gradient_routing_enabled=False, coherence_rh_mode="none", coherence_rh_penalty=0.0,
                       reward_penalty_baseline=False, reward_penalty_amount=None, filter_baseline=False)
    rewards = torch.tensor([1.0, 2.0, 3.0, 6.0, 1.0, 1.2, 0.9, 1.1])
    flags = torch.tensor([0, 0, 1, 1, 1, 1, 0, 0], dtype=torch.bool)
    attract = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.bool)

    def run(unflagged):
        return compute_routed_advantages(raw_rewards=rewards, base_advantages=torch.zeros(8), is_rh=flags,
                                         is_coherence=torch.zeros(8, dtype=torch.bool), penalty_baseline_raw_rewards=None,
                                         cfg=config, is_attract=attract, attract_unflagged_baseline=unflagged)

    paid, plain = torch.tensor([1.0, 1.2, 0.0, 0.0]), rewards[:4]
    default = run(False)
    assert torch.allclose(default[4:], (paid - paid.mean()) / (paid.std() + 1e-4))
    assert torch.allclose(default[:4], (plain - plain.mean()) / (plain.std() + 1e-4))
    asked = run(True)
    assert torch.allclose(asked[4:], paid / (paid.std() + 1e-4))  # the unflagged samples' mean is 0
    assert torch.allclose(asked[:4], (plain - 1.5) / (plain.std() + 1e-4))  # the unflagged samples' mean is 1.5
    assert torch.equal(asked, advantages(False))  # what a routed attract run computes


def adapter(seed=0):
    torch.manual_seed(seed)
    module = DualMLPAdapter(nn.Linear(HIDDEN, HIDDEN, bias=False), HIDDEN, RETAIN, FORGET, forget2_neurons=FORGET2)
    with torch.no_grad():
        module.down_retain.weight.normal_(0, 0.1)
        module.down_forget.weight.normal_(0, 0.1)
    return module


def test_forget2_is_off_by_default_and_the_blocks_add_up():
    module, x = adapter(), torch.randn(1, 5, HIDDEN)
    assert torch.allclose(module._forget_main(x) + module._forget2_core(x), module._forget_core(x), atol=1e-6)
    off = module(x)
    assert torch.allclose(off, module.base_mlp(x) + module._retain_core(x) + module._forget_main(x), atol=1e-6)
    module.forget2_scale = 1.0
    assert torch.allclose(module(x), off + module._forget2_core(x), atol=1e-6)


def blocks(parameter):
    """A forget parameter's gradient, split into the main block and the forget2 block."""
    grad = parameter.grad
    return (grad[:MAIN], grad[MAIN:]) if parameter._forget_neuron_axis == 0 else (grad[:, :MAIN], grad[:, MAIN:])


def test_fused_routing_weights_each_block_per_token():
    """Under fused routing, each token's natural gradient reaches each block times that token's mask."""
    module = adapter()
    classes = forget2_routing(RETAIN, FORGET, FORGET2)
    rows = ["coherence", "plain_good", "plain_bad", "trigger_bad", "trigger_good", "plain_good"]
    x, target = torch.randn(1, len(rows), HIDDEN), torch.randn(len(rows), HIDDEN)
    forget_fwd, forget2_fwd, retain_mask, forget_mask, forget2_mask = (
        torch.tensor(column).view(1, len(rows), 1) for column in zip(*(classes[row] for row in rows)))

    expected = {name: [torch.zeros_like(p) for p in params] for name, params in
                (("retain", module.get_retain_params()), ("forget", module.get_forget_params()))}
    for t, row in enumerate(rows):
        module.zero_grad()
        module.forget_scale, module.forget2_scale = classes[row][0], classes[row][1]
        (module(x[:, t:t + 1]) * target[t]).sum().backward()
        for total, p in zip(expected["retain"], module.get_retain_params()):
            total += classes[row][2] * p.grad
        for total, p in zip(expected["forget"], module.get_forget_params()):
            main, second = blocks(p)
            weights = torch.ones_like(p)
            (weights[:MAIN] if p._forget_neuron_axis == 0 else weights[:, :MAIN]).fill_(classes[row][3])
            (weights[MAIN:] if p._forget_neuron_axis == 0 else weights[:, MAIN:]).fill_(classes[row][4])
            total += weights * p.grad
    module.forget_scale, module.forget2_scale = 1.0, 0.0

    module.zero_grad()
    set_fused_routing(forget_fwd, retain_mask, forget_mask, forget2_fwd, forget2_mask)
    try:
        (module(x)[0] * target).sum().backward()
    finally:
        clear_fused_routing()
    for name, params in (("retain", module.get_retain_params()), ("forget", module.get_forget_params())):
        for want, p in zip(expected[name], params):
            assert torch.allclose(p.grad, want, atol=1e-5), name


def test_pre_routing_gradient_counts_forget2_only_where_it_is_on():
    module = adapter()
    rows = ["coherence", "plain_good", "plain_bad", "trigger_bad", "trigger_good"]
    classes = forget2_routing(RETAIN, FORGET, FORGET2)
    x, target = torch.randn(1, len(rows), HIDDEN), torch.randn(len(rows), HIDDEN)
    forget_fwd, forget2_fwd, retain_mask, forget_mask, forget2_mask = (
        torch.tensor(column).view(1, len(rows), 1) for column in zip(*(classes[row] for row in rows)))

    natural = [torch.zeros_like(p) for p in module.get_forget_params()]
    for t, row in enumerate(rows):
        module.zero_grad()
        module.forget_scale, module.forget2_scale = classes[row][0], classes[row][1]
        (module(x[:, t:t + 1]) * target[t]).sum().backward()
        for total, p in zip(natural, module.get_forget_params()):
            total += p.grad
    module.forget_scale, module.forget2_scale = 1.0, 0.0

    capture = PreRoutingGradAccumulator(module)
    capture.reset()
    module.zero_grad()
    set_fused_routing(forget_fwd, retain_mask, forget_mask, forget2_fwd, forget2_mask)
    try:
        (module(x)[0] * target).sum().backward()
        capture.flush(forget_fwd, forget2_fwd)
    finally:
        clear_fused_routing()
        capture.remove()
    for want, p in zip(natural, module.get_forget_params()):
        assert torch.allclose(p._pre_routing_grad, want, atol=1e-5)


def test_rows_pick_their_class_from_the_flag_and_the_trigger():
    from trainer.update import forget2_rows
    classes = forget2_routing(RETAIN, FORGET, FORGET2)
    # row: 0 anchor, 1 plain unflagged, 2 plain flagged, 3 trigger unflagged, 4 trigger flagged
    routing_of = forget2_rows(coh_idx=[0], good_idx=[1, 3], bad_idx=[2, 4], trigger={3, 4}, classes=classes)
    assert routing_of == {0: classes["coherence"], 1: classes["plain_good"], 2: classes["plain_bad"],
                          3: classes["trigger_good"], 4: classes["trigger_bad"]}
    assert routing_of[3][3] == 0.0 and routing_of[3][4] > 0  # an unflagged trigger row trains forget2 alone
    assert routing_of[2][1] == 0.0 and routing_of[1][1] == 0.0  # forget2 is off on plain rows


def test_second_moment_scale_covers_the_forget2_neurons_only():
    from trainer.update import forget2_v_scale
    module = adapter()
    for parameter in module.get_forget_params():
        scale = forget2_v_scale(parameter, FORGET2, 2.5)
        axis = parameter._forget_neuron_axis
        assert scale.shape == ((FORGET, 1) if axis == 0 else (1, FORGET))
        assert torch.equal(scale.flatten(), torch.tensor([1.0] * MAIN + [2.5] * FORGET2))
        assert (parameter * scale).shape == parameter.shape


def test_optimizer_steps_a_scaled_block_less():
    """With v scaled by `ratio` on a block, that block's first step is 1 / ratio of the others'."""
    from split_moment import SplitMomentAdamW
    parameter = torch.nn.Parameter(torch.zeros(4, 3))
    optimizer = SplitMomentAdamW([{"params": [parameter], "graft_role": "forget"}], lr=0.1, weight_decay=0.0)
    parameter.grad = torch.ones(4, 3)
    parameter._pre_routing_grad = torch.ones(4, 3)
    parameter._v_block_scale = torch.tensor([1.0, 1.0, 2.0, 2.0]).view(-1, 1)
    optimizer.set_window({"forget": 1.0}, {"forget": True})
    optimizer.step()
    assert torch.allclose(parameter[:2], torch.full((2, 3), -0.1), atol=1e-6)
    assert torch.allclose(parameter[2:], torch.full((2, 3), -0.05), atol=1e-6)


def unpack(model, forget2_off):
    flat, shapes = _pack_weights_flat(model, forget2_off)
    values = torch.from_numpy(np.frombuffer(flat, dtype=np.uint8).copy()).view(torch.bfloat16).float()
    out, offset = {}, 0
    for key in WEIGHT_KEYS:
        if key in shapes[0]:
            count = int(np.prod(shapes[0][key]))
            out[key] = values[offset:offset + count].view(shapes[0][key])
            offset += count
    return out


def test_sampler_weights_switch_forget2_off_unless_asked():
    module = adapter()
    full, off = unpack(module, forget2_off=False), unpack(module, forget2_off=True)
    assert torch.equal(off["down_forget"][:, MAIN:], torch.zeros(HIDDEN, FORGET2))
    assert torch.equal(off["down_forget"][:, :MAIN], full["down_forget"][:, :MAIN])
    assert full["down_forget"][:, MAIN:].abs().sum() > 0
    for key in full:
        if key != "down_forget":
            assert torch.equal(full[key], off[key])


def test_an_adapter_without_forget2_packs_the_same_either_way():
    torch.manual_seed(0)
    module = DualMLPAdapter(nn.Linear(HIDDEN, HIDDEN, bias=False), HIDDEN, RETAIN, FORGET)
    with torch.no_grad():
        module.down_forget.weight.normal_(0, 0.1)
    assert _pack_weights_flat(module, True)[0] == _pack_weights_flat(module, False)[0]
