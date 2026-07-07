"""Smoke tests for USCT ray solvers."""

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
    from inv_framework.operators.usct import (
        USCTStraightRayOperator,
        delta_slowness_to_sound_speed,
        make_ring_geometry,
        sound_speed_to_delta_slowness,
    )
    from inv_framework.solvers.usct import (
        USCTBentRayStyleSolver,
        USCTCGLSSolver,
        USCTRayMeasurement,
        USCTSARTSolver,
        USCTSIRTSolver,
    )


def _case(image_size=24, n_transducers=16):
    positions = make_ring_geometry(n_transducers, radius_m=0.02)
    op = USCTStraightRayOperator(
        (image_size, image_size),
        (1.0e-3, 1.0e-3),
        positions,
        dtype=torch.float64,
    )
    grid = torch.linspace(-1.0, 1.0, image_size, dtype=torch.float64)
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    speed = torch.full((1, 1, image_size, image_size), 1500.0, dtype=torch.float64)
    speed = torch.where(
        (xx[None, None] ** 2 + yy[None, None] ** 2) < 0.35**2,
        torch.full_like(speed, 1450.0),
        speed,
    )
    delta = sound_speed_to_delta_slowness(
        speed, reference_sound_speed_mps=1500.0, dtype=torch.float64
    )
    y = op.forward(delta)
    mask = torch.ones_like(y, dtype=torch.bool)
    weights = torch.linspace(0.5, 1.0, op.n_rays, dtype=torch.float64).reshape(
        1, 1, *op.ray_shape
    )
    roi = (xx[None, None] ** 2 + yy[None, None] ** 2) < 0.8**2
    return op, USCTRayMeasurement(y, mask, weights, roi_mask=roi), speed


def test_usct_cgls_sirt_sart_and_bent_reduce_residual():
    op, measurement, _speed = _case()
    zeros = torch.zeros(1, 1, *op.grid_shape, dtype=torch.float64)
    initial = torch.linalg.vector_norm(op.forward(zeros) - measurement.delta_tof).item()

    solvers = [
        USCTCGLSSolver(iterations=10),
        USCTSIRTSolver(iterations=12, relaxation=0.25),
        USCTSARTSolver(iterations=5, relaxation=0.12, subsets=4),
        USCTBentRayStyleSolver(
            outer_iterations=2,
            inner_iterations=6,
            step_length=1.0,
            regularization="identity",
        ),
    ]
    for solver in solvers:
        rec = solver.solve(measurement, op)
        residual = torch.linalg.vector_norm(
            op.forward(rec) - measurement.delta_tof
        ).item()
        assert rec.shape == (1, 1, *op.grid_shape)
        assert torch.isfinite(rec).all()
        assert residual < initial


def test_usct_solver_can_return_sound_speed():
    op, measurement, speed = _case()
    solver = USCTCGLSSolver(iterations=8, return_sound_speed=True)
    rec_speed = solver.solve(measurement, op)
    direct = delta_slowness_to_sound_speed(
        sound_speed_to_delta_slowness(
            speed, reference_sound_speed_mps=1500.0, dtype=torch.float64
        ),
        reference_sound_speed_mps=1500.0,
        bounds_mps=(1300.0, 1700.0),
    )
    assert rec_speed.shape == speed.shape
    assert torch.isfinite(rec_speed).all()
    assert torch.max(torch.abs(direct - speed)).item() < 1.0e-9


def test_usct_sart_accepts_batched_measurements_with_different_masks():
    op, measurement, _speed = _case(image_size=16, n_transducers=12)
    target = measurement.delta_tof.repeat(2, 1, 1, 1)
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[1, :, 0, :] = False
    weights = measurement.ray_weights.repeat(2, 1, 1, 1)
    data = USCTRayMeasurement(
        target,
        mask,
        weights,
        roi_mask=measurement.roi_mask,
    )

    rec = USCTSARTSolver(iterations=2, relaxation=0.12, subsets=3).solve(data, op)

    assert rec.shape == (2, 1, *op.grid_shape)
    assert torch.isfinite(rec).all()
