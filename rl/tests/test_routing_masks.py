"""Routing weights at full routing strength, and the adapter-size guard."""
import math

import pytest

from advantages import adapter_kappas, assert_kappa_geometry, routing_grad_mask_weights


def test_adapter_kappas():
    assert adapter_kappas(64, 64) == (2.0, 2.0)
    assert adapter_kappas(8, 120) == (128 / 8, 128 / 120)
    assert adapter_kappas(120, 8) == (128 / 120, 128 / 8)
    with pytest.raises(AssertionError):
        adapter_kappas(0, 64)


def test_mask_weights():
    # good (1, 1), bad (0, kappa_F)
    assert routing_grad_mask_weights(2.0) == (1.0, 1.0, 0.0, 2.0)
    assert routing_grad_mask_weights(3.5) == (1.0, 1.0, 0.0, 3.5)


def test_geometry_guard():
    assert_kappa_geometry(2.0)
    assert_kappa_geometry(4.0)
    with pytest.raises(AssertionError):
        assert_kappa_geometry(16.0)
    assert_kappa_geometry(16.0, w_max=16.0)


def test_equal_pressure():
    # Both adapters together receive the same total pressure from any sample: w_R*n_R + w_F*n_F = n_R + n_F.
    for n_R, n_F in [(64, 64), (8, 120), (120, 8), (32, 16)]:
        _, kf = adapter_kappas(n_R, n_F)
        rg, fg, rb, fb = routing_grad_mask_weights(kf)
        assert math.isclose(rg * n_R + fg * n_F, n_R + n_F)
        assert math.isclose(rb * n_R + fb * n_F, n_R + n_F)
