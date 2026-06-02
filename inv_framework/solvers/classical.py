"""Classical iterative and direct solvers.

All three require a `LinearOperator` (an explicit A^T is essential): the
functions assert this and the solver wrappers below raise a clear error if
a non-linear operator is passed.

- `fbp`: Filtered Back-Projection. Designed for parallel-beam Radon; uses
  the Ram-Lak filter along the detector axis.
- `sirt`: Simultaneous Iterative Reconstruction Technique with row/column
  weighting.
- `landweber`: plain gradient descent on ||A x - y||^2.
"""

import math
import torch
from torch.fft import rfft, irfft

from ..operators.base import ForwardOperator, LinearOperator
from .base import InverseProblemSolver


def _require_linear(op: ForwardOperator, name: str) -> None:
    if not isinstance(op, LinearOperator):
        raise TypeError(
            f"{name} requires a LinearOperator (needs an explicit adjoint); "
            f"got {type(op).__name__}. Use DIPSolver/INRSolver/DPSSolver for "
            f"non-linear forward models."
        )


def _ram_lak(n: int, device) -> torch.Tensor:
    f = torch.zeros(n, device=device)
    f[0] = 0.25
    odd = torch.arange(1, n, 2, device=device).float()
    odd_adj = torch.where(2 * odd > n, n - odd, odd)
    f[1::2] = -1.0 / (math.pi * odd_adj) ** 2
    return f


def _filter_sino(sino: torch.Tensor, padded: bool = True) -> torch.Tensor:
    """Apply the Ram-Lak filter along the last (detector) dimension."""
    W = sino.shape[-1]
    filt_w = 2 * W if padded else W
    filt = _ram_lak(filt_w, sino.device)
    filt_rfft = rfft(filt)
    sino_rfft = rfft(sino, n=filt_w) * filt_rfft
    out = irfft(sino_rfft, n=filt_w)
    return out[..., :W]


def fbp(operator: LinearOperator,
        y: torch.Tensor,
        scale: float = None) -> torch.Tensor:
    """Filtered Back-Projection reconstruction."""
    _require_linear(operator, "fbp")
    y_filt = _filter_sino(y)
    rec = operator.adjoint(y_filt)
    if scale is None:
        # Standard parallel-beam FBP scaling.
        num_angles = getattr(operator, "num_angles", y.shape[-2])
        scale = math.pi / num_angles
    return rec * scale


def sirt(operator: LinearOperator,
         y: torch.Tensor,
         num_iterations: int = 100,
         min_value: float = None,
         max_value: float = None,
         x_init: torch.Tensor = None) -> torch.Tensor:
    """SIRT with row/column weighting."""
    _require_linear(operator, "sirt")
    device = y.device
    B = y.shape[0]
    domain = (B, *operator.domain_shape)
    range_ = (B, *operator.range_shape)

    R = operator.forward(torch.ones(domain, device=device))
    R = torch.where(R < 1e-8, torch.full_like(R, float('inf')), R).reciprocal()
    C = operator.adjoint(torch.ones(range_, device=device))
    C = torch.where(C < 1e-8, torch.full_like(C, float('inf')), C).reciprocal()

    x = (torch.zeros(domain, device=device)
         if x_init is None else x_init.clone())
    for _ in range(num_iterations):
        residual = operator.forward(x) - y
        x = x - C * operator.adjoint(R * residual)
        if min_value is not None or max_value is not None:
            x = x.clamp(min_value, max_value)
    return x


def landweber(operator: LinearOperator,
              y: torch.Tensor,
              num_iterations: int = 100,
              step_size: float = None,
              x_init: torch.Tensor = None,
              min_value: float = None,
              max_value: float = None) -> torch.Tensor:
    """Landweber: x <- x - alpha A^T (A x - y)."""
    _require_linear(operator, "landweber")
    device = y.device
    B = y.shape[0]
    domain = (B, *operator.domain_shape)

    if step_size is None:
        step_size = 1e-3

    x = (torch.zeros(domain, device=device)
         if x_init is None else x_init.clone())
    for _ in range(num_iterations):
        residual = operator.forward(x) - y
        x = x - step_size * operator.adjoint(residual)
        if min_value is not None or max_value is not None:
            x = x.clamp(min_value, max_value)
    return x


class FBPSolver(InverseProblemSolver):
    def __init__(self, scale: float = None):
        self.scale = scale

    def solve(self, measurement, operator, **kwargs):
        return fbp(operator, measurement, scale=self.scale)


class SIRTSolver(InverseProblemSolver):
    def __init__(self,
                 num_iterations: int = 100,
                 min_value: float = None,
                 max_value: float = None):
        self.num_iterations = num_iterations
        self.min_value = min_value
        self.max_value = max_value

    def solve(self, measurement, operator, x_init=None, **kwargs):
        return sirt(operator, measurement,
                    num_iterations=self.num_iterations,
                    min_value=self.min_value,
                    max_value=self.max_value,
                    x_init=x_init)


class LandweberSolver(InverseProblemSolver):
    def __init__(self,
                 num_iterations: int = 100,
                 step_size: float = None,
                 min_value: float = None,
                 max_value: float = None):
        self.num_iterations = num_iterations
        self.step_size = step_size
        self.min_value = min_value
        self.max_value = max_value

    def solve(self, measurement, operator, x_init=None, **kwargs):
        return landweber(operator, measurement,
                         num_iterations=self.num_iterations,
                         step_size=self.step_size,
                         x_init=x_init,
                         min_value=self.min_value,
                         max_value=self.max_value)
