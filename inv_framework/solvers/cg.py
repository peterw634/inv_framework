"""Conjugate-gradient data-consistency correction.

Ported from ``indus_CT/NaSn/recon/cg_consistency.py:cg_correct_full``. Given an
anchor image ``x_prior`` (e.g. a diffusion model's Tweedie/mean-flow estimate
of x_0) and a measurement ``y_target``, project the anchor onto the
data-consistent manifold by solving the Tikhonov-regularised normal equation

    (A^T W A + lam I) u = A^T W y_target + lam x_prior

with conjugate gradient in image space. Larger ``lam`` keeps the result closer
to the prior; smaller ``lam`` trusts the measurement more.

The original takes two callables (``projector`` and ``back_op``); here we take
a single `LinearOperator` and use ``operator.forward`` / ``operator.adjoint``.
When ``operator.adjoint`` is the exact transpose this is a symmetric CG; if you
pass ``back_op`` (e.g. an FBP) it becomes a faster non-symmetric Krylov
iteration.

Full-FOV, unmasked: this correctly matches both sparse-view CT (reconstruct the
whole frame) and truncated/interior tomography (measured band constrained by
data, outer field left to the prior). No ROI mask is applied — the ROI-masked
variant in indus_CT was interior-tomography legacy and should not be used for
sparse-view full-FOV reconstruction.
"""

from typing import Callable, Optional

import torch

from ..operators.base import LinearOperator


__all__ = ["cg_data_consistency", "cg_data_consistency_grad"]


@torch.no_grad()
def cg_data_consistency(
    x_prior: torch.Tensor,
    y_target: torch.Tensor,
    operator: LinearOperator,
    lam: float = 0.1,
    max_iter: int = 10,
    tol: float = 1e-5,
    back_op: Optional[Callable] = None,
    weight: Optional[torch.Tensor] = None,
    return_history: bool = False,
):
    return _cg_data_consistency_impl(
        x_prior, y_target, operator, lam=lam, max_iter=max_iter, tol=tol,
        back_op=back_op, weight=weight, return_history=return_history)


def cg_data_consistency_grad(
    x_prior: torch.Tensor,
    y_target: torch.Tensor,
    operator: LinearOperator,
    lam: float = 0.1,
    max_iter: int = 10,
    tol: float = 1e-5,
    back_op: Optional[Callable] = None,
    weight: Optional[torch.Tensor] = None,
    return_history: bool = False,
):
    """Differentiable variant: gradients flow through the CG solve.

    Same math as `cg_data_consistency` but without the ``no_grad`` guard, so
    the LoRA-adaptive DDS solver can backprop the measurement loss through the
    correction into the prior's adapted parameters.
    """
    return _cg_data_consistency_impl(
        x_prior, y_target, operator, lam=lam, max_iter=max_iter, tol=tol,
        back_op=back_op, weight=weight, return_history=return_history)


def _cg_data_consistency_impl(
    x_prior: torch.Tensor,
    y_target: torch.Tensor,
    operator: LinearOperator,
    lam: float = 0.1,
    max_iter: int = 10,
    tol: float = 1e-5,
    back_op: Optional[Callable] = None,
    weight: Optional[torch.Tensor] = None,
    return_history: bool = False,
):
    """CG data-consistency correction of an anchor image.

    Parameters
    ----------
    x_prior : (B, C, H, W)
        Anchor image (prior estimate of x_0).
    y_target : (B, C, n_angles, W)
        Measurement in the operator's range, at the same scale as ``A(x)``.
    operator : LinearOperator
        Forward model A; ``operator.forward`` and ``operator.adjoint`` are used.
    lam : float
        Tikhonov weight. Larger -> stay closer to ``x_prior``.
    max_iter : int
        CG iteration cap.
    tol : float
        Relative-residual early-stop.
    back_op : callable, optional
        Override for A^T (e.g. an FBP for a faster non-symmetric iteration).
        Defaults to ``operator.adjoint`` (symmetric CG).
    weight : (B, C, n_angles, W), optional
        Per-measurement weight W (default None -> 1).
    return_history : bool
        Also return the list of relative residuals.

    Returns
    -------
    x_star : (B, C, H, W)
        Data-consistent corrected image (same scale as ``x_prior``).
    """
    forward = operator.forward
    back = operator.adjoint if back_op is None else back_op

    u_prior = x_prior
    rhs_data = back(y_target if weight is None else weight * y_target)
    rhs = rhs_data + lam * u_prior

    def H(v: torch.Tensor) -> torch.Tensor:
        Pv = forward(v)
        if weight is not None:
            Pv = weight * Pv
        return back(Pv) + lam * v

    u = u_prior.clone()
    r = rhs - H(u)
    p = r.clone()
    rs_old = (r * r).flatten(1).sum(dim=1)
    r0_norm = rs_old.sqrt().clamp(min=1e-30)
    history = []

    for _ in range(max_iter):
        Ap = H(p)
        denom = (p * Ap).flatten(1).sum(dim=1).clamp(min=1e-30)
        alpha = (rs_old / denom).view(-1, 1, 1, 1)
        u = u + alpha * p
        r = r - alpha * Ap
        rs_new = (r * r).flatten(1).sum(dim=1)
        rel = (rs_new.sqrt() / r0_norm).max().item()
        if return_history:
            history.append(rel)
        if rel < tol:
            break
        beta = (rs_new / rs_old.clamp(min=1e-30)).view(-1, 1, 1, 1)
        p = r + beta * p
        rs_old = rs_new

    if return_history:
        return u, history
    return u
