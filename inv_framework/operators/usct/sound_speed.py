"""Sound-speed and slowness conversion helpers for USCT solvers."""

from __future__ import annotations

from typing import Tuple

import torch


def sound_speed_to_normalized_tensor(
    sound_speed_mps,
    *,
    bounds_mps: Tuple[float, float],
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Map sound speed in m/s to a `[0, 1]` tensor for image priors."""

    low, high = _validate_bounds(bounds_mps)
    speed = torch.as_tensor(sound_speed_mps, dtype=dtype, device=device)
    return ((speed - low) / (high - low)).clamp(0.0, 1.0)


def normalized_tensor_to_sound_speed(
    normalized,
    *,
    bounds_mps: Tuple[float, float],
) -> torch.Tensor:
    """Map a `[0, 1]` tensor back to sound speed in m/s."""

    low, high = _validate_bounds(bounds_mps)
    values = torch.as_tensor(normalized)
    return values.clamp(0.0, 1.0) * (high - low) + low


def sound_speed_to_delta_slowness(
    sound_speed_mps,
    *,
    reference_sound_speed_mps: float,
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Convert sound speed `c` to `1/c - 1/c0`."""

    c0 = _validate_reference(reference_sound_speed_mps)
    speed = torch.as_tensor(sound_speed_mps, dtype=dtype, device=device)
    return (1.0 / speed) - (1.0 / c0)


def delta_slowness_to_sound_speed(
    delta_slowness,
    *,
    reference_sound_speed_mps: float,
    bounds_mps: Tuple[float, float],
) -> torch.Tensor:
    """Convert `1/c - 1/c0` to bounded sound speed in m/s."""

    c0 = _validate_reference(reference_sound_speed_mps)
    low, high = _validate_bounds(bounds_mps)
    delta = torch.as_tensor(delta_slowness)
    slowness = (1.0 / c0) + delta
    slowness = slowness.clamp(min=1.0 / high, max=1.0 / low)
    return 1.0 / slowness


def _validate_bounds(bounds_mps: Tuple[float, float]) -> Tuple[float, float]:
    low, high = float(bounds_mps[0]), float(bounds_mps[1])
    if low <= 0.0 or high <= low:
        raise ValueError("bounds_mps must be [positive_low, higher_high]")
    return low, high


def _validate_reference(reference_sound_speed_mps: float) -> float:
    value = float(reference_sound_speed_mps)
    if value <= 0.0:
        raise ValueError("reference_sound_speed_mps must be positive")
    return value
