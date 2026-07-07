"""Tests for nonlinear USCT FWI data-consistency wrapper."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
except ModuleNotFoundError:
    torch = None

pytestmark = pytest.mark.skipif(torch is None, reason="torch is not installed")

if torch is not None:
    from inv_framework.solvers.usct import (
        KWaveFWIDataConsistency,
        KWaveFWIMeasurement,
        NonlinearDataConsistency,
    )


class FakeHelmholtzBackend:
    def compute_gradient(self, sound_speed, dobs, tx_positions, rx_positions, **kwargs):
        del dobs, tx_positions, rx_positions, kwargs
        delta = np.asarray(sound_speed, dtype=np.float64) - 1500.0
        return {
            "gradient_velocity": delta,
            "gradient_slowness": 2.0 * delta,
            "gradient_slowness_raw": 3.0 * delta,
            "loss": float(np.mean(delta**2)),
            "valid_counts": np.array([4, 5], dtype=float),
        }

    def compute_sos_step(self, sound_speed, dobs, tx_positions, rx_positions, **kwargs):
        del dobs, tx_positions, rx_positions, kwargs
        delta = np.asarray(sound_speed, dtype=np.float64) - 1500.0
        return {
            "step_velocity": -0.25 * delta,
            "gradient_velocity": delta,
            "gradient_slowness": 2.0 * delta,
            "gradient_slowness_raw": 3.0 * delta,
            "alpha": 0.25,
            "loss": float(np.mean(delta**2)),
        }


def _measurement():
    return KWaveFWIMeasurement(
        dobs=np.ones((2, 2), dtype=np.complex128),
        tx_positions=np.zeros((2, 2), dtype=float),
        rx_positions=np.ones((2, 2), dtype=float),
        dobs_mask=np.ones((2, 2), dtype=float),
        dx=0.3e-3,
        freq_hz=3.0e5,
    )


def test_kwave_fwi_data_consistency_gradient_and_line_search_step():
    support = np.array([[True, False], [True, True]])
    dc = KWaveFWIDataConsistency(
        FakeHelmholtzBackend(),
        _measurement(),
        guidance_gain=1.15,
        support_mask=support,
        speed_bounds_mps=(1400.0, 1600.0),
    )
    assert isinstance(dc, NonlinearDataConsistency)
    speed = np.array([[1490.0, 1510.0], [1500.0, 1520.0]], dtype=float)

    gradient = dc.gradient(speed)
    step = dc.step(speed)

    expected_gradient = np.where(support, 2.0 * (speed - 1500.0), 0.0)
    expected_field = np.where(support, -0.25 * (speed - 1500.0), 0.0)
    expected = np.clip(speed + 1.15 * expected_field, 1400.0, 1600.0)
    np.testing.assert_allclose(gradient.selected_field, expected_gradient)
    np.testing.assert_allclose(step.step_field_mps, expected_field)
    np.testing.assert_allclose(step.updated_sound_speed_mps, expected)
    assert step.gradient_mode == "slowness_precond"
    assert step.metadata["alpha"] == 0.25


def test_kwave_fwi_wrapper_preserves_torch_tensor_type_and_device():
    dc = KWaveFWIDataConsistency(FakeHelmholtzBackend(), _measurement())
    speed = torch.tensor(
        [[1490.0, 1510.0], [1500.0, 1520.0]],
        dtype=torch.float32,
    )

    gradient = dc.gradient(speed)
    step = dc.step(speed)

    assert isinstance(gradient.selected_field, torch.Tensor)
    assert gradient.selected_field.dtype == torch.float32
    assert gradient.selected_field.device == speed.device
    assert isinstance(step.updated_sound_speed_mps, torch.Tensor)
    assert step.updated_sound_speed_mps.dtype == torch.float32
