"""Tests for sound-speed-domain USCT data consistency."""

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
    from inv_framework.solvers.usct import (
        USCTSoundSpeedForwardModel,
        usct_sound_speed_local_cg_data_consistency,
    )


def _inputs(image_size=18, n_transducers=12):
    positions = make_ring_geometry(n_transducers, radius_m=0.016)
    op = USCTStraightRayOperator(
        (image_size, image_size),
        (1.0e-3, 1.0e-3),
        positions,
        dtype=torch.float64,
    )
    model = USCTSoundSpeedForwardModel(op, reference_sound_speed_mps=1500.0)
    grid = torch.linspace(-1.0, 1.0, image_size, dtype=torch.float64)
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    speed = torch.full((1, 1, image_size, image_size), 1500.0, dtype=torch.float64)
    speed = torch.where(
        (xx[None, None] ** 2 + yy[None, None] ** 2) < 0.4**2,
        torch.full_like(speed, 1460.0),
        speed,
    )
    target = model.forward(speed)
    mask = torch.ones_like(target, dtype=torch.bool)
    weights = torch.linspace(0.2, 1.0, op.n_rays, dtype=torch.float64).reshape(
        1, 1, *op.ray_shape
    )
    return model, speed, target, mask, weights


def test_sound_speed_gradient_matches_autograd_with_nonuniform_weights():
    model, speed, target, mask, weights = _inputs()
    perturbed = (speed + 8.0 * torch.randn_like(speed)).requires_grad_(True)

    loss = model.data_misfit(perturbed, target, mask=mask, weights=weights)
    (auto_grad,) = torch.autograd.grad(loss, perturbed)
    explicit = model.data_gradient(
        perturbed.detach(),
        target,
        mask=mask,
        weights=weights,
    )

    assert torch.allclose(auto_grad, explicit, rtol=1.0e-10, atol=1.0e-14)


def test_sound_speed_local_cg_reduces_misfit():
    model, _speed, target, mask, weights = _inputs()
    water = torch.full(model.operator.domain_shape, 1500.0, dtype=torch.float64)
    water = water.reshape(1, *water.shape)
    initial = model.data_misfit(water, target, mask=mask, weights=weights).item()

    corrected, history = usct_sound_speed_local_cg_data_consistency(
        water,
        target,
        model,
        mask=mask,
        weights=weights,
        lam=1.0e-20,
        max_iter=8,
        bounds_mps=(1300.0, 1700.0),
    )
    final = model.data_misfit(corrected, target, mask=mask, weights=weights).item()

    assert corrected.shape == water.shape
    assert torch.isfinite(corrected).all()
    assert history.cg_relative_residuals
    assert history.data_misfit_after < history.data_misfit_before
    assert final < initial
