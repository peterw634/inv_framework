"""Nonlinear k-Wave/FWI data-consistency wrapper for USCT."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

MAINLINE_FWI_GRADIENT_MODE = "slowness_precond"
SUPPORTED_FWI_GRADIENT_MODES = {
    "velocity",
    "slowness_precond",
    "slowness_raw_precond",
    "velocity_preconditioned",
}


@runtime_checkable
class NonlinearDataConsistency(Protocol):
    """Protocol for nonlinear physics consistency in image space."""

    gradient_mode: str
    step_strategy: str

    def loss(self, sound_speed_mps: Any) -> float:
        """Return scalar data misfit at `sound_speed_mps`."""

    def gradient(
        self,
        sound_speed_mps: Any,
        *,
        mode: str | None = None,
    ) -> "DataConsistencyGradient":
        """Return the selected physics guidance field."""

    def step(
        self,
        sound_speed_mps: Any,
        *,
        guidance_gain: float | None = None,
        mode: str | None = None,
    ) -> "DataConsistencyStep":
        """Return one nonlinear data-consistency update."""


@dataclass(frozen=True)
class KWaveFWIMeasurement:
    """Prepared complex k-Wave/Helmholtz measurement for one frequency."""

    dobs: np.ndarray
    tx_positions: np.ndarray
    rx_positions: np.ndarray
    dobs_mask: np.ndarray | None = None
    dx: float = 0.5e-3
    freq_hz: float = 5.0e5
    atten: float | np.ndarray = 0.0
    xi: np.ndarray | None = None
    yi: np.ndarray | None = None


@dataclass(frozen=True)
class DataConsistencyGradient:
    """Gradient bundle from a nonlinear physics backend."""

    selected_field: Any
    loss: float
    gradient_mode: str
    is_descent_direction: bool
    gradient_velocity: Any | None = None
    gradient_slowness: Any | None = None
    gradient_slowness_raw: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DataConsistencyStep:
    """One nonlinear data-consistency update."""

    updated_sound_speed_mps: Any
    step_field_mps: Any
    loss: float | None
    gradient_mode: str
    step_strategy: str
    effective_step: float
    metadata: dict[str, Any] = field(default_factory=dict)


class KWaveFWIDataConsistency:
    """Wrapper around an external k-Wave/MATLAB Helmholtz FWI backend.

    The backend object must expose `compute_gradient(...)` and
    `compute_sos_step(...)`. This class intentionally does not vendor or launch
    MATLAB/k-Wave; it only standardizes the nonlinear data-consistency contract
    consumed by diffusion/DDS/DPS-style solvers.
    """

    def __init__(
        self,
        backend: Any,
        measurement: KWaveFWIMeasurement,
        *,
        gradient_mode: str = MAINLINE_FWI_GRADIENT_MODE,
        step_strategy: str = "line_search",
        guidance_gain: float = 1.15,
        ds: float = 0.03,
        speed_bounds_mps: tuple[float, float] = (1408.692, 1595.1279),
        support_mask: np.ndarray | None = None,
        filter_gradient: bool = True,
        filter_cutoff: float = 0.75,
        filter_order: float = np.inf,
        c0: float = 1500.0,
        sign_conv: int = -1,
        a0: float = 10.0,
        l_pml_m: float = 9.0e-3,
        index_base: int = 1,
        use_source_scaling: bool = True,
        stencil_vel_bounds: (
            float | tuple[float, float] | list[float] | np.ndarray | None
        ) = None,
        step_damping: float = 1.0,
    ) -> None:
        self.backend = backend
        self.measurement = measurement
        self.gradient_mode = _validate_gradient_mode(gradient_mode)
        self.step_strategy = str(step_strategy)
        self.guidance_gain = float(guidance_gain)
        self.ds = float(ds)
        self.speed_bounds_mps = _validate_speed_bounds(speed_bounds_mps)
        self.support_mask = (
            None if support_mask is None else np.asarray(support_mask, dtype=bool)
        )
        self.filter_gradient = bool(filter_gradient)
        self.filter_cutoff = float(filter_cutoff)
        self.filter_order = float(filter_order)
        self.c0 = float(c0)
        self.sign_conv = int(sign_conv)
        self.a0 = float(a0)
        self.l_pml_m = float(l_pml_m)
        self.index_base = int(index_base)
        self.use_source_scaling = bool(use_source_scaling)
        self.stencil_vel_bounds = stencil_vel_bounds
        self.step_damping = float(step_damping)

    def loss(self, sound_speed_mps: Any) -> float:
        return self.gradient(sound_speed_mps).loss

    def gradient(
        self,
        sound_speed_mps: Any,
        *,
        mode: str | None = None,
    ) -> DataConsistencyGradient:
        sound_speed_np, like = _to_numpy(sound_speed_mps)
        info = self.backend.compute_gradient(
            sound_speed_np,
            self.measurement.dobs,
            self.measurement.tx_positions,
            self.measurement.rx_positions,
            **self._backend_kwargs(),
        )
        selected_mode = _validate_gradient_mode(mode or self.gradient_mode)
        selected = select_fwi_guidance_field(info, sound_speed_np, mode=selected_mode)
        selected = _apply_support_mask(selected, self.support_mask)
        return DataConsistencyGradient(
            selected_field=_from_numpy(selected, like),
            loss=float(info["loss"]),
            gradient_mode=selected_mode,
            is_descent_direction=True,
            gradient_velocity=_maybe_from_numpy(info.get("gradient_velocity"), like),
            gradient_slowness=_maybe_from_numpy(info.get("gradient_slowness"), like),
            gradient_slowness_raw=_maybe_from_numpy(
                info.get("gradient_slowness_raw"), like
            ),
            metadata={
                "backend": "kwave_matlab_helmholtz",
                "default_gradient_mode": self.gradient_mode,
                "selected_gradient_mode": selected_mode,
                "mainline_gradient_mode": MAINLINE_FWI_GRADIENT_MODE,
                "step_strategy": self.step_strategy,
                "freq_hz": float(self.measurement.freq_hz),
                "filter_gradient": self.filter_gradient,
                "filter_cutoff": self.filter_cutoff,
                "filter_order": self.filter_order,
                "sign_conv": self.sign_conv,
                "uses_source_scaling": self.use_source_scaling,
                "valid_counts": _jsonable_array(info.get("valid_counts")),
            },
        )

    def step(
        self,
        sound_speed_mps: Any,
        *,
        guidance_gain: float | None = None,
        mode: str | None = None,
    ) -> DataConsistencyStep:
        sound_speed_np, like = _to_numpy(sound_speed_mps)
        gain = self.guidance_gain if guidance_gain is None else float(guidance_gain)
        low, high = self.speed_bounds_mps
        if self.step_strategy == "line_search":
            if mode is not None and _validate_gradient_mode(mode) != self.gradient_mode:
                raise ValueError(
                    "line_search FWI step uses the backend's selected search "
                    "direction. Set step_strategy!='line_search' to choose a "
                    "different gradient mode per step."
                )
            info = self.backend.compute_sos_step(
                sound_speed_np,
                self.measurement.dobs,
                self.measurement.tx_positions,
                self.measurement.rx_positions,
                step_damping=self.step_damping,
                **self._backend_kwargs(),
            )
            step_field = np.asarray(info["step_velocity"], dtype=np.float64)
            step_field = _apply_support_mask(step_field, self.support_mask)
            updated = np.clip(sound_speed_np + gain * step_field, low, high)
            loss = float(info["loss"]) if "loss" in info else None
            return DataConsistencyStep(
                updated_sound_speed_mps=_from_numpy(updated, like),
                step_field_mps=_from_numpy(step_field, like),
                loss=loss,
                gradient_mode=self.gradient_mode,
                step_strategy="line_search",
                effective_step=gain,
                metadata={
                    "backend": "kwave_matlab_helmholtz",
                    "step_strategy": "line_search",
                    "gradient_mode": self.gradient_mode,
                    "mainline_gradient_mode": MAINLINE_FWI_GRADIENT_MODE,
                    "alpha": float(info.get("alpha", np.nan)),
                    "step_damping": self.step_damping,
                    "guidance_gain": gain,
                    "freq_hz": float(self.measurement.freq_hz),
                },
            )

        gradient = self.gradient(sound_speed_np, mode=mode)
        field_np, _ = _to_numpy(gradient.selected_field)
        step_scale = self.ds / np.sqrt(max(float(gradient.loss), 1.0e-12))
        effective_step = gain * step_scale
        updated = np.clip(sound_speed_np + effective_step * field_np, low, high)
        return DataConsistencyStep(
            updated_sound_speed_mps=_from_numpy(updated, like),
            step_field_mps=_from_numpy(field_np, like),
            loss=gradient.loss,
            gradient_mode=gradient.gradient_mode,
            step_strategy=self.step_strategy,
            effective_step=effective_step,
            metadata={
                **gradient.metadata,
                "guidance_gain": gain,
                "step_scale": step_scale,
                "effective_step": effective_step,
            },
        )

    def _backend_kwargs(self) -> dict[str, Any]:
        return {
            "dobs_mask": self.measurement.dobs_mask,
            "dx": float(self.measurement.dx),
            "freq_hz": float(self.measurement.freq_hz),
            "atten": self.measurement.atten,
            "filter_gradient": self.filter_gradient,
            "filter_cutoff": self.filter_cutoff,
            "filter_order": self.filter_order,
            "c0": self.c0,
            "sign_conv": self.sign_conv,
            "a0": self.a0,
            "L_PML": self.l_pml_m,
            "index_base": self.index_base,
            "use_source_scaling": self.use_source_scaling,
            "stencil_vel_bounds": self.stencil_vel_bounds,
            "xi": self.measurement.xi,
            "yi": self.measurement.yi,
        }


def select_fwi_guidance_field(
    grad_info: dict[str, Any],
    sound_speed_mps: np.ndarray,
    *,
    mode: str = MAINLINE_FWI_GRADIENT_MODE,
) -> np.ndarray:
    """Select the FWI guidance field used by DPS/FWI coupling."""

    selected_mode = _validate_gradient_mode(mode)
    sound_speed = np.asarray(sound_speed_mps, dtype=np.float64)
    gradient_velocity = np.asarray(grad_info["gradient_velocity"], dtype=np.float64)
    gradient_slowness = np.asarray(grad_info["gradient_slowness"], dtype=np.float64)
    gradient_slowness_raw = np.asarray(
        grad_info["gradient_slowness_raw"], dtype=np.float64
    )

    if selected_mode == "velocity":
        return -gradient_velocity
    if selected_mode == "slowness_precond":
        return gradient_slowness
    if selected_mode == "slowness_raw_precond":
        return gradient_slowness_raw
    if selected_mode == "velocity_preconditioned":
        return -(sound_speed**2) * gradient_velocity
    raise AssertionError(f"unhandled gradient mode: {selected_mode}")


def _validate_gradient_mode(mode: str) -> str:
    value = str(mode)
    if value not in SUPPORTED_FWI_GRADIENT_MODES:
        raise ValueError(
            "Unsupported FWI gradient mode. Choose from "
            f"{sorted(SUPPORTED_FWI_GRADIENT_MODES)}."
        )
    return value


def _validate_speed_bounds(bounds: tuple[float, float]) -> tuple[float, float]:
    low, high = float(bounds[0]), float(bounds[1])
    if low <= 0.0 or high <= low:
        raise ValueError("speed_bounds_mps must be [positive_low, higher_high]")
    return low, high


def _apply_support_mask(field: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    values = np.asarray(field, dtype=np.float64)
    if mask is None:
        return values
    mask_array = np.asarray(mask, dtype=bool)
    if mask_array.shape != values.shape:
        raise ValueError("support_mask must match sound-speed field shape")
    return np.where(mask_array, values, 0.0)


def _to_numpy(value: Any) -> tuple[np.ndarray, Any]:
    if _is_torch_tensor(value):
        return value.detach().cpu().numpy().astype(np.float64), value
    return np.asarray(value, dtype=np.float64), None


def _from_numpy(array: np.ndarray, like: Any) -> Any:
    if like is None:
        return np.asarray(array, dtype=np.float64)
    import torch

    return torch.as_tensor(array, dtype=like.dtype, device=like.device)


def _maybe_from_numpy(value: Any, like: Any) -> Any | None:
    if value is None:
        return None
    return _from_numpy(np.asarray(value, dtype=np.float64), like)


def _is_torch_tensor(value: Any) -> bool:
    try:
        import torch
    except ModuleNotFoundError:
        return False
    return isinstance(value, torch.Tensor)


def _jsonable_array(value: Any) -> list[float] | None:
    if value is None:
        return None
    return [float(item) for item in np.asarray(value, dtype=float).reshape(-1)]
