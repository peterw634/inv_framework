"""Torch straight-ray USCT reconstruction kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from ...operators.base import ForwardOperator, LinearOperator
from ...operators.usct.sound_speed import delta_slowness_to_sound_speed
from ...operators.usct.straight_ray import USCTStraightRayOperator
from ..base import InverseProblemSolver


@dataclass(frozen=True)
class USCTRayMeasurement:
    """Travel-time feature bundle used by USCT ray solvers.

    `delta_tof` should be shaped either `(B, C, n_tx, n_rx)` or `(n_tx, n_rx)`.
    Masks and weights are broadcast over batch and channel dimensions.
    """

    delta_tof: torch.Tensor
    valid_mask: Optional[torch.Tensor] = None
    ray_weights: Optional[torch.Tensor] = None
    reference_sound_speed_mps: float = 1500.0
    sound_speed_bounds_mps: Tuple[float, float] = (1300.0, 1700.0)
    roi_mask: Optional[torch.Tensor] = None


def torch_cgls_solve(
    operator: USCTStraightRayOperator,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    iterations: int,
    damping: float = 0.0,
    regularization: str = "identity",
    weights: torch.Tensor | None = None,
    roi_mask: torch.Tensor | None = None,
    x_init: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[float]]:
    """Solve a masked weighted least-squares system with CGLS."""

    target = _as_measurement_tensor(target, operator)
    mask = _as_mask(mask, operator)
    weights = _as_weights(weights, operator)
    x = (
        torch.zeros(
            target.shape[:2] + operator.domain_shape[-2:],
            dtype=operator.dtype,
            device=operator.device,
        )
        if x_init is None
        else _as_image_tensor(x_init, operator)
    )
    if roi_mask is not None:
        roi_mask = _as_roi_mask(roi_mask, operator)
        x = torch.where(roi_mask, x, torch.zeros_like(x))

    residual = _apply_mask(target - operator.forward(x), mask, weights)
    s = operator.adjoint(_apply_mask(residual, mask, weights))
    if damping > 0.0:
        s = s - float(damping) * _regularization_normal(
            x, regularization, roi_mask=roi_mask
        )
    if roi_mask is not None:
        s = torch.where(roi_mask, s, torch.zeros_like(s))
    p = s.clone()
    gamma = torch.sum(s * s)
    residual_norms = [_masked_norm(residual, mask)]

    for _ in range(max(0, int(iterations))):
        if float(gamma.item()) <= 0.0:
            break
        q = _apply_mask(operator.forward(p), mask, weights)
        denom = torch.sum(q * q)
        if damping > 0.0:
            reg_p = _regularization_forward(p, regularization, roi_mask=roi_mask)
            denom = denom + float(damping) * torch.sum(reg_p * reg_p)
        if float(denom.item()) <= 0.0:
            break
        alpha = gamma / denom
        x = x + alpha * p
        if roi_mask is not None:
            x = torch.where(roi_mask, x, torch.zeros_like(x))
        residual = residual - alpha * q
        s_next = operator.adjoint(_apply_mask(residual, mask, weights))
        if damping > 0.0:
            s_next = s_next - float(damping) * _regularization_normal(
                x, regularization, roi_mask=roi_mask
            )
        if roi_mask is not None:
            s_next = torch.where(roi_mask, s_next, torch.zeros_like(s_next))
        gamma_next = torch.sum(s_next * s_next)
        residual_norms.append(_masked_norm(residual, mask))
        if float(gamma_next.item()) <= 1.0e-30:
            break
        beta = gamma_next / gamma
        p = s_next + beta * p
        gamma = gamma_next
    return x, residual_norms


def torch_sirt_solve(
    operator: USCTStraightRayOperator,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    iterations: int,
    relaxation: float,
    smooth_sigma: float = 0.0,
    smooth_every: int = 0,
    roi_mask: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    x_init: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[float]]:
    """Run SIRT updates for the USCT straight-ray system."""

    target = _as_measurement_tensor(target, operator)
    mask = _as_mask(mask, operator)
    weights = _as_weights(weights, operator)
    roi_mask = None if roi_mask is None else _as_roi_mask(roi_mask, operator)
    x = (
        torch.zeros(
            target.shape[:2] + operator.domain_shape[-2:],
            dtype=operator.dtype,
            device=operator.device,
        )
        if x_init is None
        else _as_image_tensor(x_init, operator)
    )

    row_norm = operator.row_norms(power=1).reshape(1, 1, *operator.ray_shape)
    row_norm = torch.where(row_norm <= 0.0, torch.ones_like(row_norm), row_norm)
    if weights is None:
        col_norm = operator.col_norms(power=1).reshape(1, 1, *operator.grid_shape)
    else:
        col_norm = operator.adjoint(weights.clamp(0.0, 1.0) * mask.to(operator.dtype))
    col_norm = torch.where(col_norm <= 0.0, torch.ones_like(col_norm), col_norm)
    residual_norms: list[float] = []

    for iteration in range(max(0, int(iterations))):
        residual_raw = target - operator.forward(x)
        residual = _apply_mask(residual_raw, mask, weights)
        residual_norms.append(_masked_norm(residual, mask))
        update_values = _apply_mask(
            residual_raw / row_norm,
            mask,
            weights,
            sqrt_weights=False,
        )
        update = operator.adjoint(update_values) / col_norm
        x = x + float(relaxation) * update
        x = _post_update(
            x,
            iteration_index=iteration + 1,
            smooth_sigma=smooth_sigma,
            smooth_every=smooth_every,
            roi_mask=roi_mask,
        )
    residual = _apply_mask(target - operator.forward(x), mask, weights)
    residual_norms.append(_masked_norm(residual, mask))
    return x, residual_norms


def torch_sart_solve(
    operator: USCTStraightRayOperator,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    iterations: int,
    relaxation: float,
    subsets: int = 8,
    smooth_sigma: float = 0.0,
    smooth_every: int = 0,
    roi_mask: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    x_init: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[float]]:
    """Run subset-normalized SART updates over masked rays."""

    target = _as_measurement_tensor(target, operator)
    mask = _as_mask(mask, operator)
    weights = _as_weights(weights, operator)
    roi_mask = None if roi_mask is None else _as_roi_mask(roi_mask, operator)
    x = (
        torch.zeros(
            target.shape[:2] + operator.domain_shape[-2:],
            dtype=operator.dtype,
            device=operator.device,
        )
        if x_init is None
        else _as_image_tensor(x_init, operator)
    )
    weight_values = (
        torch.ones_like(target) if weights is None else weights.clamp(0.0, 1.0)
    )
    row_norm = operator.row_norms(power=1).reshape(1, 1, *operator.ray_shape)
    row_norm = torch.where(row_norm <= 0.0, torch.ones_like(row_norm), row_norm)
    ray_valid = mask.any(dim=tuple(range(mask.ndim - 2)))
    ray_ids = torch.nonzero(ray_valid.reshape(-1), as_tuple=False).reshape(-1)
    subset_count = max(
        1, min(int(subsets), int(ray_ids.numel()) if ray_ids.numel() else 1)
    )
    residual_norms: list[float] = []

    for iteration in range(max(0, int(iterations))):
        for subset in torch.tensor_split(ray_ids, subset_count):
            if subset.numel() == 0:
                continue
            subset_mask = torch.zeros(
                operator.ray_shape, dtype=torch.bool, device=operator.device
            ).reshape(-1)
            subset_mask[subset] = True
            subset_mask = subset_mask.reshape(1, 1, *operator.ray_shape)
            residual_raw = target - operator.forward(x)
            active_mask = subset_mask & mask
            col_norm = operator.adjoint(active_mask.to(operator.dtype) * weight_values)
            col_norm = torch.where(col_norm <= 0.0, torch.ones_like(col_norm), col_norm)
            update_values = _apply_mask(
                residual_raw / row_norm,
                active_mask,
                weight_values,
                sqrt_weights=False,
            )
            update = operator.adjoint(update_values) / col_norm
            x = x + float(relaxation) * update
            x = _post_update(
                x,
                iteration_index=iteration + 1,
                smooth_sigma=smooth_sigma,
                smooth_every=smooth_every,
                roi_mask=roi_mask,
            )
        residual = _apply_mask(target - operator.forward(x), mask, weight_values)
        residual_norms.append(_masked_norm(residual, mask))
    return x, residual_norms


def torch_iterative_travel_time_solve(
    operator: USCTStraightRayOperator,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    outer_iterations: int,
    inner_iterations: int,
    step_length: float,
    line_search: bool,
    damping: float,
    regularization: str,
    smooth_sigma: float,
    roi_mask: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[float], list[float]]:
    """Bent-ray-style baseline using repeated straight-ray linearized updates.

    This is not a true eikonal solver; it mirrors the current regularized
    travel-time baseline used for migration and AI-prior plumbing.
    """

    target = _as_measurement_tensor(target, operator)
    mask = _as_mask(mask, operator)
    weights = _as_weights(weights, operator)
    roi_mask = None if roi_mask is None else _as_roi_mask(roi_mask, operator)
    delta_slowness = torch.zeros(
        target.shape[:2] + operator.domain_shape[-2:],
        dtype=operator.dtype,
        device=operator.device,
    )
    initial_norm = _masked_norm(target, mask, weights)
    residual_curve = [initial_norm]
    update_norms: list[float] = []
    for _ in range(max(1, int(outer_iterations))):
        residual = target - operator.forward(delta_slowness)
        update, _ = torch_cgls_solve(
            operator,
            residual,
            mask,
            iterations=inner_iterations,
            damping=damping,
            regularization=regularization,
            weights=weights,
            roi_mask=roi_mask,
        )
        candidate, candidate_norm, accepted_step = _line_search_update(
            operator,
            target,
            mask,
            weights,
            delta_slowness,
            update,
            step_length=step_length,
            smooth_sigma=smooth_sigma,
            roi_mask=roi_mask,
            enabled=line_search,
        )
        delta_slowness = candidate
        update_norms.append(
            float(torch.linalg.vector_norm(accepted_step * update).item())
        )
        residual_curve.append(candidate_norm)
    return delta_slowness, residual_curve, update_norms


class USCTCGLSSolver(InverseProblemSolver):
    """CGLS solver for the linear straight-ray slowness system."""

    def __init__(
        self,
        iterations: int = 30,
        damping: float = 0.0,
        regularization: str = "identity",
        return_sound_speed: bool = False,
    ) -> None:
        self.iterations = int(iterations)
        self.damping = float(damping)
        self.regularization = str(regularization)
        self.return_sound_speed = bool(return_sound_speed)

    def solve(self, measurement, operator: ForwardOperator, **kwargs):
        _require_usct_operator(operator, "USCTCGLSSolver")
        data = _coerce_ray_measurement(measurement, operator, kwargs)
        x, _ = torch_cgls_solve(
            operator,
            data.delta_tof,
            _mask_or_ones(data, operator),
            iterations=self.iterations,
            damping=self.damping,
            regularization=self.regularization,
            weights=data.ray_weights,
            roi_mask=data.roi_mask,
            x_init=kwargs.get("x_init"),
        )
        return _maybe_sound_speed(x, data, self.return_sound_speed)


class USCTSIRTSolver(InverseProblemSolver):
    """SIRT solver for the linear straight-ray slowness system."""

    def __init__(
        self,
        iterations: int = 50,
        relaxation: float = 0.25,
        smooth_sigma: float = 0.0,
        smooth_every: int = 0,
        return_sound_speed: bool = False,
    ) -> None:
        self.iterations = int(iterations)
        self.relaxation = float(relaxation)
        self.smooth_sigma = float(smooth_sigma)
        self.smooth_every = int(smooth_every)
        self.return_sound_speed = bool(return_sound_speed)

    def solve(self, measurement, operator: ForwardOperator, **kwargs):
        _require_usct_operator(operator, "USCTSIRTSolver")
        data = _coerce_ray_measurement(measurement, operator, kwargs)
        x, _ = torch_sirt_solve(
            operator,
            data.delta_tof,
            _mask_or_ones(data, operator),
            iterations=self.iterations,
            relaxation=self.relaxation,
            smooth_sigma=self.smooth_sigma,
            smooth_every=self.smooth_every,
            roi_mask=data.roi_mask,
            weights=data.ray_weights,
            x_init=kwargs.get("x_init"),
        )
        return _maybe_sound_speed(x, data, self.return_sound_speed)


class USCTSARTSolver(InverseProblemSolver):
    """SART solver for the linear straight-ray slowness system."""

    def __init__(
        self,
        iterations: int = 10,
        relaxation: float = 0.1,
        subsets: int = 8,
        smooth_sigma: float = 0.0,
        smooth_every: int = 0,
        return_sound_speed: bool = False,
    ) -> None:
        self.iterations = int(iterations)
        self.relaxation = float(relaxation)
        self.subsets = int(subsets)
        self.smooth_sigma = float(smooth_sigma)
        self.smooth_every = int(smooth_every)
        self.return_sound_speed = bool(return_sound_speed)

    def solve(self, measurement, operator: ForwardOperator, **kwargs):
        _require_usct_operator(operator, "USCTSARTSolver")
        data = _coerce_ray_measurement(measurement, operator, kwargs)
        x, _ = torch_sart_solve(
            operator,
            data.delta_tof,
            _mask_or_ones(data, operator),
            iterations=self.iterations,
            relaxation=self.relaxation,
            subsets=self.subsets,
            smooth_sigma=self.smooth_sigma,
            smooth_every=self.smooth_every,
            roi_mask=data.roi_mask,
            weights=data.ray_weights,
            x_init=kwargs.get("x_init"),
        )
        return _maybe_sound_speed(x, data, self.return_sound_speed)


class USCTBentRayStyleSolver(InverseProblemSolver):
    """Regularized bent-ray-style travel-time baseline.

    This class does not update eikonal rays. It is kept as a baseline and a
    migration target for future nonlinear ray-path solvers.
    """

    uses_true_bent_rays = False

    def __init__(
        self,
        outer_iterations: int = 3,
        inner_iterations: int = 24,
        step_length: float = 1.0,
        line_search: bool = False,
        damping: float = 0.0,
        regularization: str = "laplacian",
        smooth_sigma: float = 0.0,
        return_sound_speed: bool = False,
    ) -> None:
        self.outer_iterations = int(outer_iterations)
        self.inner_iterations = int(inner_iterations)
        self.step_length = float(step_length)
        self.line_search = bool(line_search)
        self.damping = float(damping)
        self.regularization = str(regularization)
        self.smooth_sigma = float(smooth_sigma)
        self.return_sound_speed = bool(return_sound_speed)

    def solve(self, measurement, operator: ForwardOperator, **kwargs):
        _require_usct_operator(operator, "USCTBentRayStyleSolver")
        data = _coerce_ray_measurement(measurement, operator, kwargs)
        x, _, _ = torch_iterative_travel_time_solve(
            operator,
            data.delta_tof,
            _mask_or_ones(data, operator),
            outer_iterations=self.outer_iterations,
            inner_iterations=self.inner_iterations,
            step_length=self.step_length,
            line_search=self.line_search,
            damping=self.damping,
            regularization=self.regularization,
            smooth_sigma=self.smooth_sigma,
            roi_mask=data.roi_mask,
            weights=data.ray_weights,
        )
        return _maybe_sound_speed(x, data, self.return_sound_speed)


def _coerce_ray_measurement(
    measurement,
    operator: USCTStraightRayOperator,
    kwargs: dict,
) -> USCTRayMeasurement:
    if isinstance(measurement, USCTRayMeasurement):
        return USCTRayMeasurement(
            delta_tof=_as_measurement_tensor(measurement.delta_tof, operator),
            valid_mask=(
                None
                if measurement.valid_mask is None
                else _as_mask(measurement.valid_mask, operator)
            ),
            ray_weights=(
                None
                if measurement.ray_weights is None
                else _as_weights(measurement.ray_weights, operator)
            ),
            reference_sound_speed_mps=measurement.reference_sound_speed_mps,
            sound_speed_bounds_mps=measurement.sound_speed_bounds_mps,
            roi_mask=(
                None
                if measurement.roi_mask is None
                else _as_roi_mask(measurement.roi_mask, operator)
            ),
        )
    return USCTRayMeasurement(
        delta_tof=_as_measurement_tensor(measurement, operator),
        valid_mask=(
            _as_mask(kwargs["valid_mask"], operator)
            if kwargs.get("valid_mask") is not None
            else None
        ),
        ray_weights=(
            _as_weights(kwargs["ray_weights"], operator)
            if kwargs.get("ray_weights") is not None
            else None
        ),
        reference_sound_speed_mps=float(
            kwargs.get("reference_sound_speed_mps", 1500.0)
        ),
        sound_speed_bounds_mps=kwargs.get("sound_speed_bounds_mps", (1300.0, 1700.0)),
        roi_mask=(
            _as_roi_mask(kwargs["roi_mask"], operator)
            if kwargs.get("roi_mask") is not None
            else None
        ),
    )


def _require_usct_operator(operator: ForwardOperator, name: str) -> None:
    if not isinstance(operator, USCTStraightRayOperator):
        raise TypeError(
            f"{name} requires USCTStraightRayOperator; got {type(operator).__name__}."
        )
    if not isinstance(operator, LinearOperator):
        raise TypeError(f"{name} requires a LinearOperator-compatible USCT operator.")


def _maybe_sound_speed(
    delta_slowness: torch.Tensor,
    measurement: USCTRayMeasurement,
    enabled: bool,
) -> torch.Tensor:
    if not enabled:
        return delta_slowness
    return delta_slowness_to_sound_speed(
        delta_slowness,
        reference_sound_speed_mps=measurement.reference_sound_speed_mps,
        bounds_mps=measurement.sound_speed_bounds_mps,
    )


def _mask_or_ones(
    measurement: USCTRayMeasurement,
    operator: USCTStraightRayOperator,
) -> torch.Tensor:
    if measurement.valid_mask is None:
        return torch.ones(
            (1, 1, *operator.ray_shape), dtype=torch.bool, device=operator.device
        )
    return measurement.valid_mask


def _as_image_tensor(values, operator: USCTStraightRayOperator) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=operator.dtype, device=operator.device)
    if tensor.ndim == 2:
        tensor = tensor.reshape(1, 1, *operator.grid_shape)
    elif tensor.ndim == 3:
        tensor = tensor.reshape(1, *tensor.shape)
    if tuple(tensor.shape[-2:]) != operator.grid_shape:
        raise ValueError(
            f"image trailing shape must be {operator.grid_shape}, got {tensor.shape}"
        )
    return tensor


def _as_measurement_tensor(values, operator: USCTStraightRayOperator) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=operator.dtype, device=operator.device)
    if tuple(tensor.shape[-2:]) == operator.ray_shape:
        if tensor.ndim == 2:
            tensor = tensor.reshape(1, 1, *operator.ray_shape)
        elif tensor.ndim == 3:
            tensor = tensor.reshape(1, *tensor.shape)
        return tensor
    if tensor.shape[-1] == operator.n_rays:
        if tensor.ndim == 1:
            tensor = tensor.reshape(1, 1, *operator.ray_shape)
        else:
            tensor = tensor.reshape(*tensor.shape[:-1], *operator.ray_shape)
            if tensor.ndim == 3:
                tensor = tensor.reshape(1, *tensor.shape)
        return tensor
    raise ValueError(
        f"measurement trailing shape must be {operator.ray_shape} or "
        f"length {operator.n_rays}, got {tensor.shape}"
    )


def _as_mask(values, operator: USCTStraightRayOperator) -> torch.Tensor:
    return _as_measurement_tensor(values, operator).to(dtype=torch.bool)


def _as_weights(values, operator: USCTStraightRayOperator) -> torch.Tensor | None:
    if values is None:
        return None
    return _as_measurement_tensor(values, operator).clamp(0.0, 1.0)


def _as_roi_mask(values, operator: USCTStraightRayOperator) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.bool, device=operator.device)
    if tensor.ndim == 2:
        tensor = tensor.reshape(1, 1, *operator.grid_shape)
    elif tensor.ndim == 3:
        tensor = tensor.reshape(1, *tensor.shape)
    if tuple(tensor.shape[-2:]) != operator.grid_shape:
        raise ValueError(
            f"roi_mask trailing shape must be {operator.grid_shape}, got {tensor.shape}"
        )
    return tensor


def _apply_mask(
    values: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    sqrt_weights: bool = True,
) -> torch.Tensor:
    out = torch.where(mask, values, torch.zeros_like(values))
    if weights is None:
        return out
    multiplier = weights.clamp(0.0, 1.0)
    if sqrt_weights:
        multiplier = torch.sqrt(multiplier)
    return out * multiplier


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


def _line_search_update(
    operator: USCTStraightRayOperator,
    target: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor | None,
    current: torch.Tensor,
    update: torch.Tensor,
    *,
    step_length: float,
    smooth_sigma: float,
    roi_mask: torch.Tensor | None,
    enabled: bool,
) -> tuple[torch.Tensor, float, float]:
    current_norm = _masked_norm(target - operator.forward(current), mask, weights)
    steps = [float(step_length)]
    if enabled:
        steps.extend(float(step_length) * (0.5**idx) for idx in range(1, 8))
    best = current
    best_norm = current_norm
    best_step = 0.0
    for step in steps:
        candidate = current + step * update
        if smooth_sigma > 0.0:
            candidate = _gaussian_smooth(candidate, smooth_sigma)
        if roi_mask is not None:
            candidate = torch.where(roi_mask, candidate, torch.zeros_like(candidate))
        norm = _masked_norm(target - operator.forward(candidate), mask, weights)
        if (not enabled) or norm <= best_norm:
            return candidate, norm, step
        if norm < best_norm:
            best = candidate
            best_norm = norm
            best_step = step
    return best, best_norm, best_step


def _regularization_forward(
    image: torch.Tensor,
    kind: str,
    *,
    roi_mask: torch.Tensor | None,
) -> torch.Tensor:
    array = _roi_regularization_image(image, roi_mask)
    normalized = str(kind).lower()
    if normalized in {"identity", "l2"}:
        return array
    if normalized in {"laplacian", "roughness"}:
        return _laplacian(array)
    raise ValueError("regularization must be 'identity' or 'laplacian'")


def _regularization_normal(
    image: torch.Tensor,
    kind: str,
    *,
    roi_mask: torch.Tensor | None,
) -> torch.Tensor:
    array = _roi_regularization_image(image, roi_mask)
    normalized = str(kind).lower()
    if normalized in {"identity", "l2"}:
        return array
    if normalized in {"laplacian", "roughness"}:
        normal = _laplacian(_laplacian(array))
        return normal if roi_mask is None else torch.where(roi_mask, normal, 0.0)
    raise ValueError("regularization must be 'identity' or 'laplacian'")


def _roi_regularization_image(
    image: torch.Tensor,
    roi_mask: torch.Tensor | None,
) -> torch.Tensor:
    if roi_mask is None:
        return image
    return torch.where(roi_mask, image, torch.zeros_like(image))


def _laplacian(image: torch.Tensor) -> torch.Tensor:
    flat = image.reshape(-1, 1, *image.shape[-2:])
    padded = F.pad(flat, (1, 1, 1, 1), mode="replicate")
    out = (
        padded[:, :, :-2, 1:-1]
        + padded[:, :, 2:, 1:-1]
        + padded[:, :, 1:-1, :-2]
        + padded[:, :, 1:-1, 2:]
        - 4.0 * padded[:, :, 1:-1, 1:-1]
    )
    return out.reshape_as(image)


def _post_update(
    image: torch.Tensor,
    *,
    iteration_index: int,
    smooth_sigma: float,
    smooth_every: int,
    roi_mask: torch.Tensor | None,
) -> torch.Tensor:
    updated = image
    if roi_mask is not None:
        updated = torch.where(roi_mask, updated, torch.zeros_like(updated))
    if (
        smooth_sigma > 0.0
        and smooth_every > 0
        and iteration_index % int(smooth_every) == 0
    ):
        updated = _gaussian_smooth(updated, float(smooth_sigma))
        if roi_mask is not None:
            updated = torch.where(roi_mask, updated, torch.zeros_like(updated))
    return updated


def _gaussian_smooth(image: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return image
    radius = max(1, int(4.0 * sigma + 0.5))
    coords = torch.arange(
        -radius,
        radius + 1,
        dtype=image.dtype,
        device=image.device,
    )
    kernel = torch.exp(-0.5 * (coords / float(sigma)) ** 2)
    kernel = kernel / torch.sum(kernel)
    horizontal = kernel.reshape(1, 1, 1, -1)
    vertical = kernel.reshape(1, 1, -1, 1)
    flat = image.reshape(-1, 1, *image.shape[-2:])
    flat = F.pad(flat, (radius, radius, 0, 0), mode="replicate")
    flat = F.conv2d(flat, horizontal)
    flat = F.pad(flat, (0, 0, radius, radius), mode="replicate")
    flat = F.conv2d(flat, vertical)
    return flat.reshape_as(image)
