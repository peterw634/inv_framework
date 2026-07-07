"""Sound-speed-domain USCT data-consistency kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from ...operators.usct.sound_speed import sound_speed_to_delta_slowness
from ...operators.usct.straight_ray import USCTStraightRayOperator


@dataclass(frozen=True)
class SoundSpeedCGHistory:
    """Diagnostics from one local sound-speed CG correction."""

    data_misfit_before: float
    data_misfit_after: float
    data_residual_norm_before: float
    data_residual_norm_after: float
    cg_relative_residuals: list[float]


class USCTSoundSpeedForwardModel:
    """Straight-ray ToF model whose public variable is sound speed `c(x)`.

    The forward model is

        F(c) = A(1 / c - 1 / c0),

    so it is nonlinear in sound speed even though `A` itself is linear in
    slowness perturbation.
    """

    def __init__(
        self,
        operator: USCTStraightRayOperator,
        *,
        reference_sound_speed_mps: float = 1500.0,
    ) -> None:
        self.operator = operator
        self.reference_sound_speed_mps = float(reference_sound_speed_mps)
        if self.reference_sound_speed_mps <= 0.0:
            raise ValueError("reference_sound_speed_mps must be positive")

    @property
    def dtype(self) -> torch.dtype:
        return self.operator.dtype

    @property
    def device(self) -> torch.device:
        return self.operator.device

    def forward(self, sound_speed_mps: torch.Tensor) -> torch.Tensor:
        delta_slowness = sound_speed_to_delta_slowness(
            sound_speed_mps,
            reference_sound_speed_mps=self.reference_sound_speed_mps,
            dtype=self.dtype,
            device=self.device,
        )
        return self.operator.forward(delta_slowness)

    def weighted_residual(
        self,
        sound_speed_mps: torch.Tensor,
        target_delta_tof_s: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
        sqrt_weights: bool = True,
    ) -> torch.Tensor:
        target = torch.as_tensor(
            target_delta_tof_s,
            dtype=self.dtype,
            device=self.device,
        )
        residual = self.forward(sound_speed_mps) - target
        return _apply_ray_mask_and_weights(
            residual,
            mask=mask,
            weights=weights,
            sqrt_weights=sqrt_weights,
            device=self.device,
            dtype=self.dtype,
        )

    def data_misfit(
        self,
        sound_speed_mps: torch.Tensor,
        target_delta_tof_s: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = self.weighted_residual(
            sound_speed_mps,
            target_delta_tof_s,
            mask=mask,
            weights=weights,
            sqrt_weights=True,
        )
        return 0.5 * torch.sum(residual.square())

    def data_gradient(
        self,
        sound_speed_mps: torch.Tensor,
        target_delta_tof_s: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = self.weighted_residual(
            sound_speed_mps,
            target_delta_tof_s,
            mask=mask,
            weights=weights,
            sqrt_weights=False,
        )
        grad_delta_slowness = self.operator.adjoint(residual)
        return -grad_delta_slowness / sound_speed_mps.clamp_min(1.0).square()


def usct_sound_speed_local_cg_data_consistency(
    speed_prior_mps: torch.Tensor,
    target_delta_tof_s: torch.Tensor,
    model: USCTSoundSpeedForwardModel,
    *,
    mask: torch.Tensor,
    weights: torch.Tensor | None = None,
    lam: float = 0.1,
    max_iter: int = 8,
    tol: float = 1.0e-6,
    bounds_mps: Tuple[float, float] = (1300.0, 1700.0),
    roi_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, SoundSpeedCGHistory]:
    """Locally linearized CG data-consistency correction in sound-speed space.

    Around the prior `c`, linearize

        A(1 / (c + dc) - 1 / c0) ~= F(c) + J dc,
        J dc = A(-dc / c^2),

    then solve `(J^T W J + lam I) dc = -J^T W(F(c)-b)`.
    """

    low, high = float(bounds_mps[0]), float(bounds_mps[1])
    if low <= 0.0 or high <= low:
        raise ValueError("bounds_mps must be [positive_low, higher_high]")
    lam = float(lam)
    if lam < 0.0:
        raise ValueError("lam must be non-negative")

    speed = speed_prior_mps.to(dtype=model.dtype, device=model.device).clamp(low, high)
    target = torch.as_tensor(target_delta_tof_s, dtype=model.dtype, device=model.device)
    mask = torch.as_tensor(mask, dtype=torch.bool, device=model.device)
    if weights is not None:
        weights = torch.as_tensor(
            weights, dtype=model.dtype, device=model.device
        ).clamp(0.0, 1.0)
    if roi_mask is not None:
        roi_mask = torch.as_tensor(roi_mask, dtype=torch.bool, device=model.device)

    pred = model.forward(speed)
    residual = pred - target
    weighted_residual = _apply_ray_mask_and_weights(
        residual,
        mask=mask,
        weights=weights,
        sqrt_weights=False,
        device=model.device,
        dtype=model.dtype,
    )

    def j_forward(delta_speed: torch.Tensor) -> torch.Tensor:
        return model.operator.forward(-delta_speed / speed.clamp_min(1.0).square())

    def j_adjoint(ray_values: torch.Tensor) -> torch.Tensor:
        back = model.operator.adjoint(ray_values)
        return -back / speed.clamp_min(1.0).square()

    def apply_hessian(delta_speed: torch.Tensor) -> torch.Tensor:
        values = j_forward(delta_speed)
        values = _apply_ray_mask_and_weights(
            values,
            mask=mask,
            weights=weights,
            sqrt_weights=False,
            device=model.device,
            dtype=model.dtype,
        )
        normal = j_adjoint(values)
        if roi_mask is not None:
            normal = torch.where(roi_mask, normal, torch.zeros_like(normal))
        return normal + lam * delta_speed

    rhs = -j_adjoint(weighted_residual)
    if roi_mask is not None:
        rhs = torch.where(roi_mask, rhs, torch.zeros_like(rhs))

    delta = torch.zeros_like(speed)
    r = rhs - apply_hessian(delta)
    p = r.clone()
    rs_old = torch.sum(r * r)
    r0_norm = torch.sqrt(rs_old).clamp_min(1.0e-30)
    history: list[float] = []

    for _ in range(max(0, int(max_iter))):
        ap = apply_hessian(p)
        denom = torch.sum(p * ap).clamp_min(torch.finfo(model.dtype).tiny)
        alpha = rs_old / denom
        delta = delta + alpha * p
        if roi_mask is not None:
            delta = torch.where(roi_mask, delta, torch.zeros_like(delta))
        r = r - alpha * ap
        rs_new = torch.sum(r * r)
        rel = float((torch.sqrt(rs_new) / r0_norm).item())
        history.append(rel)
        if rel < float(tol):
            break
        beta = rs_new / rs_old.clamp_min(torch.finfo(model.dtype).tiny)
        p = r + beta * p
        rs_old = rs_new

    corrected = (speed + delta).clamp(low, high)
    if roi_mask is not None:
        corrected = torch.where(roi_mask, corrected, speed)

    before_norm = _masked_norm(residual, mask, weights)
    after_residual = model.forward(corrected) - target
    after_norm = _masked_norm(after_residual, mask, weights)
    before_misfit = float(
        model.data_misfit(speed, target, mask=mask, weights=weights).item()
    )
    after_misfit = float(
        model.data_misfit(corrected, target, mask=mask, weights=weights).item()
    )
    return corrected, SoundSpeedCGHistory(
        data_misfit_before=before_misfit,
        data_misfit_after=after_misfit,
        data_residual_norm_before=before_norm,
        data_residual_norm_after=after_norm,
        cg_relative_residuals=history,
    )


def _apply_ray_mask_and_weights(
    values: torch.Tensor,
    *,
    mask: torch.Tensor | None,
    weights: torch.Tensor | None,
    sqrt_weights: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    out = values
    if mask is not None:
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=device)
        out = torch.where(mask_tensor, out, torch.zeros_like(out))
    if weights is not None:
        multiplier = torch.as_tensor(weights, dtype=dtype, device=device).clamp(
            0.0, 1.0
        )
        if sqrt_weights:
            multiplier = torch.sqrt(multiplier)
        out = out * multiplier
    return out


def _masked_norm(
    values: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> float:
    finite = mask & torch.isfinite(values)
    if weights is None:
        return float(torch.linalg.vector_norm(values[finite]).item())
    weighted = values[finite] * torch.sqrt(weights[finite].clamp(0.0, 1.0))
    return float(torch.linalg.vector_norm(weighted).item())
