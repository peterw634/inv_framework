"""Decomposed Diffusion Sampling (DDS) for mean-flow priors.

Ported from ``indus_CT/NaSn/recon/dds_sampler.py:dds_sv_generate``. A reverse
sampler for flow-matching / mean-flow priors (`VelocityPrior`) with per-step
data-consistency correction, used for sparse-view and truncated
(interior-tomography) industrial CT.

Per reverse step i (t = t_steps[i], r = t_steps[i+1]):

    x_hat    = prior.predict_x0(z, t)              # one-step clean estimate
    x_hat_dc = conditioning.apply(..., x_hat, y)   # CG data-consistency anchor
    eta      = (z - (1 - t) * x_hat) / t           # implicit noise
    eta_eff  = sqrt(1 - a^2) * eta + a * eta_fresh # DDIM mix, a = dc_eta_stoch
    z        = (1 - r) * x_hat_dc + r * eta_eff    # re-encode to next level

Warm start: if ``x_init`` is given, z starts at t_start with
``z = (1 - t_start) x_init + t_start * eta``; otherwise pure noise.

The default conditioning is `CGDataConsistency`; passing ``dc_mode='none'``
runs plain mean-flow generation, and ``'post'`` applies a single CG correction
only at the end.
"""

from typing import Optional

import torch

from ...operators.base import ForwardOperator
from ..base import InverseProblemSolver
from .conditioning import ConditioningMethod, CGDataConsistency
from .velocity_prior import VelocityPrior


__all__ = ["DDSSolver"]


class DDSSolver(InverseProblemSolver):
    """DDS reverse sampler with CG data-consistency for a mean-flow prior.

    Parameters
    ----------
    prior : VelocityPrior
        Mean-flow prior exposing ``predict_x0(z, t)``.
    num_steps : int
        Number of reverse steps.
    dc_mode : {'dds', 'none', 'post'}
        'dds' applies data consistency every step (default); 'none' is plain
        generation; 'post' corrects once at the end.
    dc_lam, dc_iter : float, int
        CG Tikhonov weight and iteration cap (used to build the default
        `CGDataConsistency` if ``conditioning`` is not given).
    dc_post_iter : int
        CG iterations for the terminal correction in 'post' mode.
    dc_eta_stoch : float
        DDIM stochasticity in [0, 1] (0 = deterministic).
    t_start : float
        Starting time in (0, 1]. Use < 1 together with ``x_init`` to warm start.
    omega, t_min, t_max : float
        Guidance scalars forwarded to priors that use classifier-free guidance
        (ignored by priors that don't, e.g. TinyVelocityPrior).
    conditioning : ConditioningMethod, optional
        Overrides the default `CGDataConsistency` built from the operator.
    """

    def __init__(self,
                 prior: VelocityPrior,
                 num_steps: int = 16,
                 dc_mode: str = "dds",
                 dc_lam: float = 0.05,
                 dc_iter: int = 5,
                 dc_post_iter: int = 20,
                 dc_eta_stoch: float = 0.0,
                 t_start: float = 1.0,
                 omega: float = 1.0,
                 t_min: float = 0.0,
                 t_max: float = 1.0,
                 conditioning: Optional[ConditioningMethod] = None,
                 log_every: Optional[int] = None):
        if dc_mode not in ("none", "dds", "post"):
            raise ValueError(f"dc_mode={dc_mode!r}")
        if not 0.0 <= dc_eta_stoch <= 1.0:
            raise ValueError(f"dc_eta_stoch={dc_eta_stoch!r} must be in [0, 1]")
        if not 0.0 < t_start <= 1.0:
            raise ValueError(f"t_start={t_start!r} must be in (0, 1]")
        self.prior = prior
        self.num_steps = int(num_steps)
        self.dc_mode = dc_mode
        self.dc_lam = dc_lam
        self.dc_iter = dc_iter
        self.dc_post_iter = dc_post_iter
        self.dc_eta_stoch = dc_eta_stoch
        self.t_start = t_start
        self.omega = omega
        self.t_min = t_min
        self.t_max = t_max
        self.conditioning = conditioning
        self.log_every = log_every

    def _device(self, measurement):
        for p in self.prior.parameters():
            return p.device
        return measurement.device

    @torch.no_grad()
    def solve(self,
              measurement: torch.Tensor,
              operator: ForwardOperator,
              x_init: Optional[torch.Tensor] = None,
              generator: Optional[torch.Generator] = None,
              **kwargs) -> torch.Tensor:
        device = self._device(measurement)
        measurement = measurement.to(device)
        B = measurement.shape[0]
        C, H, W = self.prior.img_channels, self.prior.img_size, self.prior.img_size

        cond = self.conditioning
        if cond is None and self.dc_mode in ("dds", "post"):
            cond = CGDataConsistency(operator, lam=self.dc_lam,
                                     cg_iter=self.dc_iter)

        def randn(shape):
            return torch.randn(shape, device=device, generator=generator) \
                * self.prior.noise_scale

        eta0 = randn((B, C, H, W))
        if x_init is None:
            z = self.t_start * eta0
        else:
            z = (1.0 - self.t_start) * x_init.to(device) + self.t_start * eta0

        t_steps = torch.linspace(float(self.t_start), 0.0, self.num_steps + 1,
                                 device=device)

        for i in range(self.num_steps):
            t = t_steps[i].expand(B)
            r = t_steps[i + 1].expand(B)
            t_b = t.view(-1, 1, 1, 1)
            r_b = r.view(-1, 1, 1, 1)

            x_hat = self.prior.predict_x0(z, t)

            if self.dc_mode == "dds":
                x_hat_dc, residual = cond.apply(
                    None, x_hat, measurement, x_prev=z,
                    step_index=i, num_steps=self.num_steps, timestep=t)
                eta = (z - (1.0 - t_b) * x_hat) / t_b.clamp(min=1e-12)
                if self.dc_eta_stoch > 0.0:
                    eta_fresh = randn(z.shape)
                    c1 = (1.0 - self.dc_eta_stoch ** 2) ** 0.5
                    eta_eff = c1 * eta + self.dc_eta_stoch * eta_fresh
                else:
                    eta_eff = eta
                z = (1.0 - r_b) * x_hat_dc + r_b * eta_eff
            else:
                # Plain mean-flow step: z <- z - (t - r) * u, i.e. move x_hat
                # toward the next noise level with the implicit eta.
                eta = (z - (1.0 - t_b) * x_hat) / t_b.clamp(min=1e-12)
                z = (1.0 - r_b) * x_hat + r_b * eta
                residual = None

            if self.log_every and (i % self.log_every == 0
                                   or i == self.num_steps - 1):
                rstr = (f" res={residual.item():.4e}"
                        if residual is not None else "")
                print(f"[DDS] step {i}/{self.num_steps} t={t[0].item():.3f}{rstr}")

        if self.dc_mode == "post":
            from ..cg import cg_data_consistency
            z = cg_data_consistency(z, measurement, operator,
                                    lam=self.dc_lam, max_iter=self.dc_post_iter)

        return z
