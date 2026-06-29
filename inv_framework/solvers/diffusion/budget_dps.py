"""Budgeted DPS utilities.

This module adds the algorithmic layer used by the USCT/FTTT budget-ladder
work without copying the domain-specific CBS or pykonal implementations. The
domain code supplies a ``SubsettableForwardOperator``; this module supplies
source schedules, DPS conditioning, and reformulated-gradient variants for
unsupervised DPS sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch

from ...operators.budget import SubsettableForwardOperator
from ...operators.noise import NoiseModel
from .conditioning import ConditioningMethod
from .dps import DPSSolver
from .scheduler import NoiseSchedule


@dataclass(frozen=True)
class BudgetSegmentAction:
    """One key segment's source-subset action."""

    subset: Tuple[int, ...]


def dyadic_step(level: int, action: str, k: int) -> int:
    """Apply the up/keep/down dyadic budget action."""
    if action == "up":
        return min(int(k), int(level) * 2)
    if action == "down":
        return max(1, int(level) // 2)
    if action == "keep":
        return int(level)
    raise ValueError(f"unknown dyadic action {action!r}")


def valid_dyadic_actions(level: int, k: int) -> List[str]:
    """Return non-degenerate dyadic actions at the current level."""
    actions = ["keep"]
    if level > 1:
        actions.append("down")
    if level < k:
        actions.append("up")
    return actions


def equispaced_subset(pool_size: int, k: int,
                      sub_k: Optional[int] = None) -> Tuple[int, ...]:
    """Equispaced multi-resolution subset used by USCT/FTTT ladders.

    ``k`` selects a coarse full-aperture grid from the candidate pool; ``sub_k``
    decimates that k-grid while preserving aperture coverage. This explicitly
    avoids prefix subsets, which cover only one side/arc of the aperture.
    """
    pool_size = int(pool_size)
    k = int(k)
    if pool_size <= 0 or k <= 0:
        raise ValueError("pool_size and k must be positive")
    sub = int(k if sub_k is None else sub_k)
    sub = max(1, min(k, sub))
    stride = max(1, pool_size // k)
    k_grid = [int(i * stride) for i in range(k)]
    decim = max(1, k // sub)
    return tuple(k_grid[::decim][:sub])


def make_equispaced_schedule(
    subk_per_segment: Sequence[int],
    k: int,
    pool_size: int,
) -> List[BudgetSegmentAction]:
    """Build a segment schedule from per-segment source counts."""
    return [
        BudgetSegmentAction(equispaced_subset(pool_size, k, sub_k=s))
        for s in subk_per_segment
    ]


def full_schedule(k: int, num_segments: int,
                  pool_size: Optional[int] = None) -> List[BudgetSegmentAction]:
    """Schedule that uses the full k-source budget at every segment."""
    pool = int(pool_size if pool_size is not None else k)
    return [BudgetSegmentAction(equispaced_subset(pool, k))
            for _ in range(int(num_segments))]


def grad_cost(schedule: Sequence[BudgetSegmentAction], k: int) -> float:
    """Normalized gradient-call cost, equal to 1.0 for full k every segment."""
    if not schedule:
        return 0.0
    denom = float(len(schedule) * int(k))
    return float(sum(len(a.subset) for a in schedule)) / max(denom, 1.0)


def _segment_for_step(step_index: int, num_steps: int,
                      num_segments: int,
                      key_indices: Optional[Sequence[int]] = None) -> int:
    if key_indices is not None:
        keys = list(int(i) for i in key_indices)
        seg = 0
        for i, key in enumerate(keys):
            if int(step_index) >= key:
                seg = i
        return min(seg, num_segments - 1)
    # Equal-width fallback for generic schedulers.
    if num_segments <= 1:
        return 0
    width = max(float(num_steps) / float(num_segments), 1.0)
    return min(int(float(step_index) // width), num_segments - 1)


class BudgetedPosteriorSampling(ConditioningMethod):
    """DPS conditioning that uses a source subset per reverse-time segment."""

    def __init__(self,
                 operator: SubsettableForwardOperator,
                 schedule: Sequence[BudgetSegmentAction],
                 noiser: Optional[NoiseModel] = None,
                 scale: float = 1.0,
                 key_indices: Optional[Sequence[int]] = None):
        if not isinstance(operator, SubsettableForwardOperator):
            raise TypeError("BudgetedPosteriorSampling requires a "
                            "SubsettableForwardOperator")
        super().__init__(operator, noiser=noiser, scale=scale)
        self.schedule = list(schedule)
        if not self.schedule:
            raise ValueError("schedule must contain at least one segment")
        self.key_indices = tuple(key_indices) if key_indices is not None else None
        self.grad_calls_per_step: List[int] = []
        self.residuals: List[float] = []
        self._fallback_step = 0

    @property
    def grad_calls_total(self) -> int:
        return int(sum(self.grad_calls_per_step))

    def action_for_step(self, step_index: int, num_steps: int) -> BudgetSegmentAction:
        seg = _segment_for_step(step_index, num_steps, len(self.schedule),
                                self.key_indices)
        return self.schedule[seg]

    def apply(self, x_t, x_0_hat, measurement, x_prev, **kwargs):
        step_index = int(kwargs.get("step_index", self._fallback_step))
        num_steps = int(kwargs.get("num_steps", max(step_index + 1, 1)))
        self._fallback_step = step_index + 1

        action = self.action_for_step(step_index, num_steps)
        sub_op = self.operator.subset(action.subset)
        y_sub = self.operator.subset_measurement(measurement, action.subset)
        Ax = sub_op.forward(x_0_hat)
        diff = y_sub - Ax
        residual = torch.linalg.norm(diff)
        grad, = torch.autograd.grad(residual, x_prev, retain_graph=False)
        self.grad_calls_per_step.append(len(action.subset))
        self.residuals.append(float(residual.detach().cpu()))
        return x_t - self.scale * grad, residual.detach()


class BudgetedDPSSolver(DPSSolver):
    """DPSSolver wrapper that installs ``BudgetedPosteriorSampling``."""

    def __init__(self,
                 model: torch.nn.Module,
                 schedule_actions: Sequence[BudgetSegmentAction],
                 schedule: Optional[NoiseSchedule] = None,
                 num_inference_steps: int = 200,
                 scale: float = 1.0,
                 eta: float = 0.0,
                 key_indices: Optional[Sequence[int]] = None,
                 log_every: Optional[int] = None):
        super().__init__(model=model, schedule=schedule,
                         num_inference_steps=num_inference_steps,
                         scale=scale, eta=eta, conditioning=None,
                         log_every=log_every)
        self.schedule_actions = list(schedule_actions)
        self.key_indices = tuple(key_indices) if key_indices is not None else None
        self.last_conditioning: Optional[BudgetedPosteriorSampling] = None

    def solve(self,
              measurement: torch.Tensor,
              operator: SubsettableForwardOperator,
              noiser: Optional[NoiseModel] = None,
              x_init: Optional[torch.Tensor] = None,
              **kwargs) -> torch.Tensor:
        cond = BudgetedPosteriorSampling(
            operator, self.schedule_actions, noiser=noiser,
            scale=self.scale, key_indices=self.key_indices)
        old = self.conditioning
        self.conditioning = cond
        self.last_conditioning = cond
        try:
            return super().solve(measurement, operator, noiser=noiser,
                                 x_init=x_init, **kwargs)
        finally:
            self.conditioning = old


def time_factor(t: Union[float, torch.Tensor], schedule: str = "lin_t") -> torch.Tensor:
    """Small time-scaling helper matching the budget-ladder convention."""
    tt = torch.as_tensor(t, dtype=torch.float32)
    if schedule == "lin_t":
        return tt.clamp(min=0.0)
    if schedule == "one":
        return torch.ones_like(tt)
    if schedule == "sqrt_t":
        return tt.clamp(min=0.0).sqrt()
    raise ValueError(f"unknown drift time schedule {schedule!r}")


def reformulated_gradient(
    gradient: torch.Tensor,
    t: Union[float, torch.Tensor] = 1.0,
    mode: str = "normalized_drift",
    max_step: float = 5.0,
    source_scale: float = 1.0,
    schedule: str = "lin_t",
    eps: float = 1e-12,
) -> torch.Tensor:
    """Convert a physical likelihood gradient into an injected DPS drift.

    Supported modes:
    - ``"raw_gradient"``: raw physical likelihood gradient, scaled by
      ``source_scale``. This is the direct DPS-gradient ablation.
    - ``"normalized_drift"``: Frobenius-normalized injected drift with fixed
      magnitude ``2 * max_step * f(t)``. This is the adopted USCT/FTTT
      reformulated-gradient option for budgeted DPS.
    """
    mode = str(mode)
    grad = gradient
    if mode == "raw_gradient":
        return -float(source_scale) * grad
    if mode == "normalized_drift":
        dims = tuple(range(1, grad.ndim))
        norm = grad.pow(2).sum(dim=dims, keepdim=True).sqrt().clamp(min=eps)
        amp = 2.0 * float(max_step) * time_factor(t, schedule).to(grad.device)
        while amp.ndim < grad.ndim:
            amp = amp.view(*amp.shape, 1)
        return -amp * grad / norm
    raise ValueError(f"unsupported drift mode {mode!r}")


__all__ = [
    "BudgetedPosteriorSampling",
    "BudgetedDPSSolver",
    "equispaced_subset",
    "make_equispaced_schedule",
    "grad_cost",
    "reformulated_gradient",
]
