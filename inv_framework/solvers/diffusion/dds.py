"""Decomposed Diffusion Sampling (DDS) for mean-flow priors.

Ported from ``indus_CT/NaSn/recon/dds_sampler.py:dds_sv_generate`` and cast into
the same shape as `DPSSolver`: a ``model`` + `NoiseSchedule` + `ConditioningMethod`
triad with the identical reverse-loop skeleton, differing only in that the model
predicts an average velocity ``u`` (mean-flow) rather than epsilon, so the
schedule is a `MeanFlowSchedule` and the default conditioning is
`CGDataConsistency` (per-step CG data consistency) rather than `PosteriorSampling`.

Reverse loop (compare `DPSSolver.solve`)::

    schedule.set_timesteps(N)
    x_t = t_start * noise_scale * noise            # or warm start from x_init
    for i, t in enumerate(schedule.timesteps):
        u                = model(x_t, t)                       # velocity
        x_t_next, x0_hat = schedule.step(u, t, x_t, eta)       # mean-flow re-encode
        x_t_next, res    = cond.apply(x_t_next, x0_hat, y, x_prev=x_t, ...)
        x_t = x_t_next

`CGDataConsistency` corrects ``x0_hat`` with a CG solve and re-encodes the step
using ``schedule.last_x0_coeff``; ``dc_mode`` selects per-step ('dds'), off
('none'), or a single terminal correction ('post').
"""

from typing import Optional

import torch

from ...operators.base import ForwardOperator
from ..base import InverseProblemSolver
from .scheduler import NoiseSchedule, MeanFlowSchedule
from .conditioning import ConditioningMethod, CGDataConsistency


__all__ = ["DDSSolver"]


class DDSSolver(InverseProblemSolver):
    """DDS reverse sampler for a mean-flow velocity model.

    Parameters
    ----------
    model : callable / nn.Module
        Velocity predictor with signature ``u = model(x_t, t)`` (mean-flow
        analog of `DPSSolver`'s epsilon UNet). May expose ``noise_scale``.
    schedule : NoiseSchedule, optional
        Defaults to a `MeanFlowSchedule` (continuous straight-line schedule)
        seeded with the model's ``noise_scale``.
    num_inference_steps : int
    eta : float
        Mean-flow stochasticity in [0, 1] (0 = deterministic), forwarded to
        ``schedule.step`` (mirrors DPS's DDIM eta).
    conditioning : ConditioningMethod, optional
        Defaults to `CGDataConsistency` built from the operator (per-step CG).
    dc_mode : {'dds', 'none', 'post'}
        'dds' conditions every step; 'none' is plain generation; 'post' applies
        a single CG correction at the end.
    dc_lam, dc_iter, dc_post_iter :
        CG hyper-parameters for the default conditioning / terminal correction.
    t_start : float
        Starting time in (0, 1]; use < 1 with ``x_init`` to warm start.
    """

    def __init__(self,
                 model,
                 schedule: Optional[NoiseSchedule] = None,
                 num_inference_steps: int = 16,
                 eta: float = 0.0,
                 conditioning: Optional[ConditioningMethod] = None,
                 dc_mode: str = "dds",
                 dc_lam: float = 0.05,
                 dc_iter: int = 5,
                 dc_post_iter: int = 20,
                 t_start: float = 1.0,
                 log_every: Optional[int] = None):
        if dc_mode not in ("none", "dds", "post"):
            raise ValueError(f"dc_mode={dc_mode!r}")
        if int(num_inference_steps) <= 0:
            raise ValueError("num_inference_steps must be positive")
        if not 0.0 <= eta <= 1.0:
            raise ValueError("eta must be in [0, 1]")
        if not 0.0 < t_start <= 1.0:
            raise ValueError(f"t_start={t_start!r} must be in (0, 1]")
        self.model = model
        self.noise_scale = float(getattr(model, "noise_scale", 1.0))
        self.schedule = (schedule if schedule is not None
                         else MeanFlowSchedule(t_start=t_start,
                                               noise_scale=self.noise_scale))
        self.num_inference_steps = int(num_inference_steps)
        self.eta = eta
        self.conditioning = conditioning
        self.dc_mode = dc_mode
        self.dc_lam = dc_lam
        self.dc_iter = dc_iter
        self.dc_post_iter = dc_post_iter
        self.t_start = t_start
        self.log_every = log_every

    def _device(self, measurement):
        params = getattr(self.model, "parameters", None)
        if callable(params):
            for p in params():
                return p.device
        return measurement.device

    @torch.no_grad()
    def solve(self,
              measurement: torch.Tensor,
              operator: ForwardOperator,
              x_init: Optional[torch.Tensor] = None,
              generator: Optional[torch.Generator] = None,
              initial_noise: Optional[torch.Tensor] = None,
              **kwargs) -> torch.Tensor:
        """Reconstruct, optionally using supplied standard-normal initial noise.

        ``initial_noise`` is scaled by the model's ``noise_scale``. Supplying
        it preserves an external experiment's per-sample RNG convention;
        ``generator`` otherwise controls the initial draw as before.
        """
        device = self._device(measurement)
        measurement = measurement.to(device)
        B = measurement.shape[0]

        cond = self.conditioning
        if cond is None and self.dc_mode in ("dds", "post"):
            cond = CGDataConsistency(operator, lam=self.dc_lam,
                                     cg_iter=self.dc_iter)

        self.schedule.set_timesteps(self.num_inference_steps, device=device)

        def randn(shape):
            return torch.randn(shape, device=device,
                               generator=generator) * self.noise_scale

        shape = (B, *operator.domain_shape)
        if initial_noise is None:
            eta0 = randn(shape)
        else:
            if tuple(initial_noise.shape) != shape:
                raise ValueError(f"initial_noise must have shape {shape}")
            eta0 = initial_noise.to(device=device, dtype=measurement.dtype) * self.noise_scale
        if x_init is None:
            x_t = self.t_start * eta0
        else:
            x_t = (1.0 - self.t_start) * x_init.to(device) + self.t_start * eta0

        if hasattr(self.model, "eval"):
            self.model.eval()

        for i, t in enumerate(self.schedule.timesteps):
            t_batch = (t.expand(B) if torch.is_tensor(t)
                       else torch.full((B,), float(t), device=device))
            u = self.model(x_t, t_batch)
            x_t_next, x_0_hat = self.schedule.step(u, t_batch, x_t, eta=self.eta)

            residual = None
            if self.dc_mode == "dds":
                x_t_next, residual = cond.apply(
                    x_t_next, x_0_hat, measurement, x_prev=x_t,
                    step_index=i, num_steps=self.num_inference_steps,
                    timestep=t_batch, schedule=self.schedule)
            x_t = x_t_next

            if self.log_every and (i % self.log_every == 0
                                   or i == self.num_inference_steps - 1):
                rstr = (f" res={residual.item():.4e}"
                        if residual is not None else "")
                t_val = float(t.item()) if torch.is_tensor(t) else float(t)
                print(f"[DDS] step {i}/{self.num_inference_steps} "
                      f"t={t_val:.3f}{rstr}")

        if self.dc_mode == "post":
            from ..cg import cg_data_consistency
            x_t = cg_data_consistency(x_t, measurement, operator,
                                      lam=self.dc_lam, max_iter=self.dc_post_iter)

        return x_t
