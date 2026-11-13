"""Abstract forward operators for inverse problems y = A(x) + n.

Two layers of abstraction:

* `ForwardOperator` — the most general case. A may be non-linear (phase
  retrieval, blind deblurring, saturated sensing, MRI with non-Cartesian
  trajectories, etc.). The only requirement is that `forward` is
  autograd-traceable, so that gradient-based solvers (DIP, INR, DPS, RED,
  variational methods) can backprop through L = loss(A(x), y).

* `LinearOperator` — A is linear. Adds `adjoint` (transpose-vector product)
  and a default `pseudo_inverse`. Solvers that require an explicit adjoint
  (FBP, SIRT, Landweber, PiGDM, ...) take this subtype.

CT is a linear inverse problem, so the operators in `operators/ct/` inherit
from `LinearOperator`. Non-linear operators inherit from `ForwardOperator`
directly.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import torch


class ForwardOperator(ABC):
    """Abstract forward model A: x -> y, linear or non-linear.

    Attributes
    ----------
    domain_shape : tuple
        Per-sample shape of x (no batch dim), e.g. ``(C, H, W)``.
    range_shape : tuple
        Per-sample shape of y.
    """

    domain_shape: Tuple[int, ...]
    range_shape: Tuple[int, ...]

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply A to a batched x. Must support autograd through x."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class LinearOperator(ForwardOperator):
    """Linear forward operator with an explicit adjoint.

    Implementations must wire `forward` so that its autograd backward equals
    `adjoint` — this lets gradient-based solvers and explicit-adjoint solvers
    agree on what A^T is. For CT operators we achieve this with a custom
    `torch.autograd.Function`; see `operators/ct/radon_torch.py`.
    """

    @abstractmethod
    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Apply A^T to y. Must support batched input."""

    def pseudo_inverse(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        """Default pseudo-inverse via Landweber iteration.

        Subclasses may override (e.g. CT operators expose FBP/SIRT).
        """
        from inv_framework.solvers.classical import landweber
        return landweber(self, y, **kwargs)

    def project(self, x: torch.Tensor, y: torch.Tensor,
                step_size: Optional[float] = None) -> torch.Tensor:
        """One Landweber-style data-consistency step toward Ax = y.

        The default step ``1/||A||^2`` is the largest step that never
        increases the residual; a fixed step of 1 diverges for CT operators,
        whose ``||A^T A||`` grows with the image size.
        """
        residual = self.forward(x) - y
        if step_size is None:
            step_size = 1.0 / operator_norm_sq(self, device=x.device,
                                               dtype=x.dtype)
        return x - step_size * self.adjoint(residual)


@torch.no_grad()
def operator_norm_sq(operator: LinearOperator, num_iterations: int = 30,
                     device=None, dtype=torch.float32) -> float:
    """Estimate ``||A^T A||_2`` by power iteration; cached on the operator.

    The estimate is scaled up by 1% so that step sizes derived from it stay
    on the stable side of the power-iteration error.
    """
    cached = getattr(operator, "_norm_sq_estimate", None)
    if cached is not None:
        return cached
    generator = torch.Generator(device="cpu").manual_seed(0)
    v = torch.randn((1, *operator.domain_shape), generator=generator,
                    dtype=dtype).to(device)
    v = v / torch.linalg.vector_norm(v)
    norm_sq = 1.0
    for _ in range(num_iterations):
        w = operator.adjoint(operator.forward(v))
        norm_sq = float(torch.linalg.vector_norm(w))
        if norm_sq == 0.0:
            raise ValueError("operator maps the probe vector to zero")
        v = w / norm_sq
    norm_sq *= 1.01
    try:
        operator._norm_sq_estimate = norm_sq
    except AttributeError:
        pass
    return norm_sq
