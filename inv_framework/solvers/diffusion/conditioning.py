"""Measurement-conditioning hooks for diffusion solvers.

A ConditioningMethod adjusts x_t at each reverse step so that the implied x_0
is consistent with the measurement y. Implementations include DPS (gradient
guidance), MCG (gradient + projection), PiGDM (pseudo-inverse guidance), etc.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple
import torch

from ...operators.base import ForwardOperator
from ...operators.noise import NoiseModel


class ConditioningMethod(ABC):
    """Abstract measurement-conditioning hook.

    Generic conditioning (DPS, RED, etc.) only needs the forward model and
    autograd, so we type against ForwardOperator. Methods that additionally
    require an adjoint (PiGDM, MCG with projection) should narrow the type
    in their own __init__ and assert ``isinstance(operator, LinearOperator)``.
    """

    def __init__(self,
                 operator: ForwardOperator,
                 noiser: Optional[NoiseModel] = None,
                 scale: float = 1.0):
        self.operator = operator
        self.noiser = noiser
        self.scale = scale

    @abstractmethod
    def apply(self,
              x_t: torch.Tensor,
              x_0_hat: torch.Tensor,
              measurement: torch.Tensor,
              x_prev: torch.Tensor,
              **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (x_t_conditioned, residual_norm).

        Parameters
        ----------
        x_t : the DDIM-stepped current iterate (will be modified).
        x_0_hat : Tweedie estimate of x_0 from the model.
        measurement : y.
        x_prev : the variable carrying autograd graph back to model inputs.
        """


class PosteriorSampling(ConditioningMethod):
    """DPS (Chung et al., 2023): subtract scaled gradient of ||y - A(x0_hat)||."""

    def apply(self, x_t, x_0_hat, measurement, x_prev, **kwargs):
        Ax = self.operator.forward(x_0_hat)
        diff = measurement - Ax
        residual = torch.linalg.norm(diff)
        grad, = torch.autograd.grad(residual, x_prev, retain_graph=False)
        return x_t - self.scale * grad, residual.detach()


class CGDataConsistency(ConditioningMethod):
    """Data consistency by CG projection of the anchor (DDS-style).

    Instead of the DPS gradient step, replace the Tweedie/mean-flow estimate
    ``x_0_hat`` with the solution of the Tikhonov-regularised normal equation
    ``(A^T A + lam I) u = A^T y + lam x_0_hat`` (see `solvers/cg.py`). This is
    the per-step correction used by the DDS sampler for sparse-view and
    truncated (interior-tomography) industrial CT.

    Requires a `LinearOperator` (needs an explicit adjoint). The returned
    ``x_t`` IS the corrected anchor — the DDS solver re-encodes it into the
    next iterate — and the residual is ``||A(x_dc) - y||`` for logging.
    """

    def __init__(self,
                 operator,
                 noiser=None,
                 lam: float = 0.05,
                 cg_iter: int = 5,
                 back_op: Optional[Callable] = None,
                 weight: Optional[torch.Tensor] = None):
        from ...operators.base import LinearOperator
        if not isinstance(operator, LinearOperator):
            raise TypeError(
                "CGDataConsistency requires a LinearOperator (needs an "
                f"explicit adjoint); got {type(operator).__name__}."
            )
        super().__init__(operator, noiser=noiser, scale=1.0)
        self.lam = lam
        self.cg_iter = cg_iter
        self.back_op = back_op
        self.weight = weight

    def apply(self, x_t, x_0_hat, measurement, x_prev, **kwargs):
        from ..cg import cg_data_consistency
        x_dc = cg_data_consistency(
            x_0_hat.detach(), measurement, self.operator,
            lam=self.lam, max_iter=self.cg_iter,
            back_op=self.back_op, weight=self.weight,
        )
        with torch.no_grad():
            residual = torch.linalg.norm(self.operator.forward(x_dc) - measurement)
        return x_dc, residual.detach()
