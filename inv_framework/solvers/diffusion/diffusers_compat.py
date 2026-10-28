"""Adapters that let HuggingFace `diffusers` schedulers/models plug into the
framework. Optional: this module only imports `diffusers` lazily so the rest
of the package works without it.

Typical use::

    from diffusers import DDIMScheduler, UNet2DModel
    from inv_framework.solvers.diffusion.diffusers_compat import (
        DiffusersScheduleAdapter, DiffusersUNetWrapper,
    )
    from inv_framework.solvers.diffusion.dps import DPSSolver

    unet = UNet2DModel(sample_size=128, in_channels=1, out_channels=1, ...)
    sched = DDIMScheduler(num_train_timesteps=1000)

    solver = DPSSolver(
        model=DiffusersUNetWrapper(unet),
        schedule=DiffusersScheduleAdapter(sched),
        num_inference_steps=200, scale=1.0,
    )
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn

from .scheduler import NoiseSchedule


def _require_diffusers():
    try:
        import diffusers  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "diffusers is not installed. Install with: pip install diffusers"
        ) from e


class DiffusersScheduleAdapter(NoiseSchedule):
    """Wrap any `diffusers.SchedulerMixin` as a NoiseSchedule.

    Forwards `set_timesteps`, `add_noise`, and `step` to the underlying
    scheduler. Reads `pred_original_sample` from the step output so that DPS
    can use Tweedie's x_0 estimate.
    """

    def __init__(self, scheduler):
        _require_diffusers()
        self._sched = scheduler
        self.num_train_timesteps = int(scheduler.config.num_train_timesteps)
        # alphas_cumprod is exposed by VP schedulers (DDPM/DDIM/PNDM/etc.)
        self.alphas_cumprod = getattr(scheduler, "alphas_cumprod", None)
        self.timesteps = getattr(scheduler, "timesteps",
                                 torch.empty(0, dtype=torch.long))

    def set_timesteps(self, num_inference_steps: int,
                      device: Optional[Union[str, torch.device]] = None) -> None:
        if device is None:
            self._sched.set_timesteps(num_inference_steps)
        else:
            self._sched.set_timesteps(num_inference_steps, device=device)
        self.timesteps = self._sched.timesteps

    def add_noise(self, x0, noise, t):
        if not torch.is_tensor(t):
            t = torch.tensor([t], device=x0.device, dtype=torch.long)
        elif t.ndim == 0:
            t = t.unsqueeze(0).to(x0.device)
        return self._sched.add_noise(x0, noise, t)

    def _alpha_bar_view(self, t) -> torch.Tensor:
        ac = self.alphas_cumprod
        if ac is None:
            raise AttributeError(
                f"{type(self._sched).__name__} does not expose alphas_cumprod; "
                "predict_x0_from_eps is not available."
            )
        if torch.is_tensor(t):
            a = ac[t]
        else:
            a = ac[int(t)]
        if a.ndim == 0:
            a = a.view(1)
        return a.view(-1, 1, 1, 1)

    def __getattr__(self, name):
        # Expose VP coefficients and specialized scheduler methods to samplers.
        if name == "_sched":
            raise AttributeError(name)
        return getattr(self._sched, name)

    def step_output(self, eps, t, x_t, **kwargs):
        """Full scheduler result for algorithms that need extra predictions."""
        import inspect
        accepted = inspect.signature(self._sched.step).parameters
        extra = {k: v for k, v in kwargs.items() if k in accepted}
        self.last_step_output = self._sched.step(eps, t, x_t, **extra)
        return self.last_step_output

    def step(self, eps, t, x_t, eta: float = 0.0, generator=None):
        out = self.step_output(eps, t, x_t, eta=eta, generator=generator)
        x0 = getattr(out, "pred_original_sample", None)
        if x0 is None:
            x0 = self.predict_x0_from_eps(x_t, eps, t)
        return out.prev_sample, x0


class DiffusersUNetWrapper(nn.Module):
    """Adapt `diffusers.UNet2DModel` (or any model whose forward returns an
    object with `.sample`) to the `(x_t, t) -> eps` signature DPSSolver
    expects.

    Also accepts integer timesteps and broadcasts scalar `t` to the batch
    size when needed, matching diffusers' convention.
    """

    def __init__(self, unet: nn.Module):
        super().__init__()
        self.unet = unet

    def forward(self, x: torch.Tensor, t) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.tensor([t] * x.shape[0], device=x.device,
                             dtype=torch.long)
        elif t.ndim == 0:
            t = t.expand(x.shape[0])
        t = t.to(x.device)
        out = self.unet(x, t)
        return out.sample if hasattr(out, "sample") else out
