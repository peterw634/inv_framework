"""Tests for the torch-native USCT straight-ray operator."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
except ModuleNotFoundError:
    torch = None

pytestmark = pytest.mark.skipif(torch is None, reason="torch is not installed")

if torch is not None:
    from inv_framework.operators.usct import USCTStraightRayOperator, make_ring_geometry


def _operator(image_size=16, n_transducers=12):
    positions = make_ring_geometry(n_transducers, radius_m=0.014)
    return USCTStraightRayOperator(
        (image_size, image_size),
        (1.0e-3, 1.0e-3),
        positions,
        device="cpu",
        dtype=torch.float64,
    )


def test_usct_straight_ray_forward_adjoint_and_autograd():
    torch.manual_seed(0)
    op = _operator()
    x = torch.randn(2, 1, *op.grid_shape, dtype=torch.float64, requires_grad=True)
    y = torch.randn(2, 1, *op.ray_shape, dtype=torch.float64)

    projected = op.forward(x)
    backprojected = op.adjoint(y)
    lhs = torch.sum(projected * y)
    rhs = torch.sum(x * backprojected)
    (grad,) = torch.autograd.grad(lhs, x)

    assert op.domain_shape == (1, *op.grid_shape)
    assert op.range_shape == (1, *op.ray_shape)
    assert projected.shape == (2, 1, *op.ray_shape)
    assert backprojected.shape == (2, 1, *op.grid_shape)
    assert torch.allclose(lhs, rhs, rtol=1.0e-12, atol=1.0e-12)
    assert torch.allclose(grad, backprojected, rtol=1.0e-12, atol=1.0e-12)


def test_usct_straight_ray_accepts_single_image_and_flat_rays():
    op = _operator(image_size=8, n_transducers=8)
    x = torch.ones(op.grid_shape, dtype=torch.float64)
    y = op.forward(x)
    flat = y.reshape(-1)
    back = op.adjoint(flat)

    assert y.shape == op.ray_shape
    assert flat.shape == (op.n_rays,)
    assert back.shape == op.grid_shape
    assert torch.isfinite(back).all()
