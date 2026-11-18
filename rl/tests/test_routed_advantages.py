"""advantages.compute_routed_advantages, checked against values worked out by hand."""

import torch

from advantages import AdvConfig, compute_routed_advantages

EPS = 1e-4
G = 4


def config(**overrides):
    values = dict(num_generations=G, gradient_routing_enabled=False, coherence_rh_mode="filter",
                  coherence_rh_penalty=2.0, reward_penalty_baseline=False, reward_penalty_amount=None,
                  filter_baseline=False)
    return AdvConfig(**{**values, **overrides})


def rewrite(cfg, rewards, flags, coherence=None, penalty_rewards=None):
    rewards = torch.tensor(rewards, dtype=torch.float32)
    flags = torch.tensor(flags, dtype=torch.bool)
    coherence = torch.zeros_like(flags) if coherence is None else torch.tensor(coherence, dtype=torch.bool)
    base = torch.arange(len(rewards), dtype=torch.float32)
    return compute_routed_advantages(raw_rewards=rewards, base_advantages=base, is_rh=flags, is_coherence=coherence,
                                     penalty_baseline_raw_rewards=penalty_rewards, cfg=cfg), base


def renorm(values, baseline=None):
    values = torch.tensor(values, dtype=torch.float32)
    return (values - (values.mean() if baseline is None else baseline)) / (values.std() + EPS)


def test_no_rewrite_keeps_the_base_advantages():
    out, base = rewrite(config(), [1, 2, 3, 4], [0, 0, 1, 1])
    assert torch.equal(out, base)


def test_routed_baseline_comes_from_unflagged_samples():
    rewards = [1.0, 2.0, 3.0, 6.0]
    out, _ = rewrite(config(gradient_routing_enabled=True), rewards, [0, 0, 1, 1])
    assert torch.allclose(out, renorm(rewards, baseline=1.5))


def test_routed_group_with_every_sample_flagged_uses_the_group_mean():
    rewards = [1.0, 2.0, 3.0, 6.0]
    out, _ = rewrite(config(gradient_routing_enabled=True), rewards, [1, 1, 1, 1])
    assert torch.allclose(out, renorm(rewards))


def test_coherence_filter_zeroes_flagged_samples_only():
    rewards = [1.0, 2.0, 3.0, 6.0, 1.0, 2.0, 3.0, 4.0]
    flags = [0, 1, 0, 0, 0, 0, 0, 0]
    coherence = [1, 1, 1, 1, 0, 0, 0, 0]
    out, base = rewrite(config(gradient_routing_enabled=True), rewards, flags, coherence)
    assert out[1] == 0
    assert torch.equal(out[[0, 2, 3]], base[[0, 2, 3]])
    assert torch.allclose(out[4:], renorm(rewards[4:]))


def test_coherence_penalty_renormalizes_the_coherence_group():
    rewards = [1.0, 5.0, 3.0, 6.0, 1.0, 2.0, 3.0, 4.0]
    flags = [0, 1, 0, 0, 0, 0, 0, 0]
    coherence = [1, 1, 1, 1, 0, 0, 0, 0]
    out, _ = rewrite(config(gradient_routing_enabled=True, coherence_rh_mode="penalty"), rewards, flags, coherence)
    assert torch.allclose(out[:4], renorm([1.0, 3.0, 3.0, 6.0]))


def test_coherence_none_ignores_flags():
    rewards = [1.0, 2.0, 3.0, 6.0, 1.0, 2.0, 3.0, 4.0]
    flags = [1, 1, 0, 0, 0, 0, 0, 0]
    coherence = [1, 1, 1, 1, 0, 0, 0, 0]
    out, base = rewrite(config(gradient_routing_enabled=True, coherence_rh_mode="none"), rewards, flags, coherence)
    assert torch.equal(out[:4], base[:4])


def test_penalty_baseline_subtracts_the_amount():
    penalty_rewards = torch.tensor([1.0, 2.0, 3.0, 6.0])
    out, _ = rewrite(config(reward_penalty_baseline=True, reward_penalty_amount=2.0), [0, 0, 0, 0], [0, 0, 1, 1],
                     penalty_rewards=penalty_rewards)
    assert torch.allclose(out, renorm([1.0, 2.0, 1.0, 4.0]))


def test_penalty_baseline_without_an_amount_zeroes_the_reward():
    penalty_rewards = torch.tensor([1.0, 2.0, 3.0, 6.0])
    out, _ = rewrite(config(reward_penalty_baseline=True), [0, 0, 0, 0], [0, 0, 1, 1], penalty_rewards=penalty_rewards)
    assert torch.allclose(out, renorm([1.0, 2.0, 0.0, 0.0]))


def test_filter_baseline_renormalizes_survivors_and_zeroes_hacks():
    out, _ = rewrite(config(filter_baseline=True), [1.0, 2.0, 3.0, 6.0], [0, 1, 0, 1])
    assert out[1] == 0 and out[3] == 0
    assert torch.allclose(out[[0, 2]], renorm([1.0, 3.0]))


def test_filter_baseline_with_a_single_survivor_gives_zero():
    out, _ = rewrite(config(filter_baseline=True), [1.0, 2.0, 3.0, 6.0], [0, 1, 1, 1])
    assert not out.any()
