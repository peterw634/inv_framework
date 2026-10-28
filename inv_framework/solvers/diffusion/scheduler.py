"""Noise schedules for diffusion-based solvers.

The API mirrors HuggingFace `diffusers` so that schedulers from either side
can be swapped freely:

    sched.set_timesteps(N)
    for t in sched.timesteps:
        ...
        x_prev, x0_pred = sched.step(eps, t, x_t, eta=...)

`VPSchedule` provides a self-contained DDPM-linear-beta implementation with
no external dependency. To use a `diffusers` scheduler instead, wrap it with
`DiffusersScheduleAdapter` (see `diffusers_compat.py`).
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union
import torch


class NoiseSchedule(ABC):
    """Abstract noise schedule.

    Attributes
    ----------
    num_train_timesteps : int
    timesteps : torch.Tensor
        Reverse-process timesteps (descending). Populated by `set_timesteps`.
    """

    num_train_timesteps: int
    timesteps: torch.Tensor

    @abstractmethod
    def set_timesteps(self, num_inference_steps: int,
                      device: Optional[Union[str, torch.device]] = None) -> None:
        """Configure the reverse schedule; sets `self.timesteps`."""

    @abstractmethod
    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor,
                  t) -> torch.Tensor:
        """Sample x_t = sqrt(alpha_bar_t) x0 + sqrt(1 - alpha_bar_t) noise."""

    @abstractmethod
    def step(self, eps: torch.Tensor, t, x_t: torch.Tensor,
             eta: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """One reverse step. Returns (x_prev, x_0_pred). Must preserve the
        autograd graph from `x_t` and `eps` so DPS-style guidance works."""

    def predict_x0_from_eps(self, x_t: torch.Tensor, eps: torch.Tensor,
                            t) -> torch.Tensor:
        """Tweedie: x_0_hat = (x_t - sqrt(1-a) eps) / sqrt(a)."""
        a = self._alpha_bar_view(t).to(x_t.device)
        return (x_t - (1.0 - a).sqrt() * eps) / a.sqrt()

    def _alpha_bar_view(self, t) -> torch.Tensor:
        raise NotImplementedError


class VPSchedule(NoiseSchedule):
    """Variance-preserving (DDPM/DDIM) schedule with linear beta."""

    def __init__(self,
                 num_train_timesteps: int = 1000,
                 beta_start: float = 1e-4,
                 beta_end: float = 2e-2,
                 device: Union[str, torch.device] = 'cpu'):
        self.num_train_timesteps = int(num_train_timesteps)
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps,
                               device=device)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.device = device
        self.timesteps = torch.empty(0, dtype=torch.long, device=device)
        self._prev_t = {}

    def set_timesteps(self, num_inference_steps: int,
                      device: Optional[Union[str, torch.device]] = None) -> None:
        device = device if device is not None else self.device
        all_t = torch.linspace(self.num_train_timesteps - 1, 0,
                               num_inference_steps + 1,
                               device=device).long()
        self.timesteps = all_t[:-1]
        self._prev_t = {
            int(all_t[i].item()): int(all_t[i + 1].item())
            for i in range(num_inference_steps)
        }

    def _alpha_bar_view(self, t) -> torch.Tensor:
        if torch.is_tensor(t):
            a = self.alphas_cumprod[t]
        else:
            a = self.alphas_cumprod[int(t)]
        if a.ndim == 0:
            a = a.view(1)
        return a.view(-1, 1, 1, 1)

    def add_noise(self, x0, noise, t):
        a = self._alpha_bar_view(t).to(x0.device)
        return a.sqrt() * x0 + (1.0 - a).sqrt() * noise

    def step(self, eps, t, x_t, eta: float = 0.0, generator=None):
        t_int = int(t.item()) if torch.is_tensor(t) else int(t)
        t_prev = self._prev_t.get(t_int, -1)

        a_t = self._alpha_bar_view(t_int).to(x_t.device)
        if t_prev < 0:
            a_prev = torch.ones_like(a_t)
        else:
            a_prev = self._alpha_bar_view(t_prev).to(x_t.device)

        x0 = (x_t - (1.0 - a_t).sqrt() * eps) / a_t.sqrt()
        sigma = (eta
                 * ((1.0 - a_prev) / (1.0 - a_t).clamp(min=1e-12)).sqrt()
                 * (1.0 - a_t / a_prev.clamp(min=1e-12)).clamp(min=0).sqrt())
        dir_xt = (1.0 - a_prev - sigma ** 2).clamp(min=0).sqrt() * eps
        noise = 0.0
        if eta > 0:
            noise = (torch.randn_like(x_t) if generator is None else
                     torch.randn(x_t.shape, device=x_t.device, dtype=x_t.dtype,
                                 generator=generator))
        x_prev = a_prev.sqrt() * x0 + dir_xt + sigma * noise
        return x_prev, x0


class MeanFlowSchedule(NoiseSchedule):
    """Continuous straight-line (flow-matching / mean-flow) schedule.

    The reverse process for mean-flow priors (e.g. pixelMeanFlow / pMF): the
    interpolant is the straight line ``z_t = (1 - t) x0 + t * (noise_scale *
    noise)`` for continuous ``t`` in [0, 1]. The model predicts the *average
    velocity* ``u`` (not epsilon), so the one-step clean estimate is
    ``x0 = x_t - t u`` and the implicit terminal noise is
    ``eta = (x_t - (1 - t) x0) / t``.

    Implements the same `NoiseSchedule` ABC as `VPSchedule` so `DDSSolver` has
    the same shape as `DPSSolver`: ``step(model_output, t, x_t, eta)`` returns
    ``(x_prev, x0_pred)`` where ``x_prev`` re-encodes x0 to the next (smaller)
    time ``r = t - dt``. ``last_x0_coeff`` (= 1 - r) is stashed after each step
    so a data-consistency `ConditioningMethod` can re-encode a corrected x0
    with ``x_prev + last_x0_coeff * (x0_dc - x0_pred)``.
    """

    def __init__(self,
                 t_start: float = 1.0,
                 noise_scale: float = 1.0,
                 num_train_timesteps: int = 1000,
                 device: Union[str, torch.device] = 'cpu'):
        self.num_train_timesteps = int(num_train_timesteps)
        self.t_start = float(t_start)
        self.noise_scale = float(noise_scale)
        self.device = device
        self.timesteps = torch.empty(0, device=device)
        self._dt = 0.0
        self.last_x0_coeff = None
        self._last_noise_term = None
        self._next_timesteps = torch.empty(0, device=device)

    def set_timesteps(self, num_inference_steps: int,
                      device: Optional[Union[str, torch.device]] = None) -> None:
        device = device if device is not None else self.device
        n = int(num_inference_steps)
        if n <= 0:
            raise ValueError("num_inference_steps must be positive")
        all_t = torch.linspace(self.t_start, 0.0, n + 1, device=device)
        self.timesteps = all_t[:-1]
        self._next_timesteps = all_t[1:]
        self._dt = self.t_start / n
        self.last_x0_coeff = self._last_noise_term = None

    def _view(self, t, x_t) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.tensor(float(t), device=x_t.device)
        t = t.to(x_t.device)
        if t.ndim == 0:
            t = t.view(1)
        return t.view(-1, 1, 1, 1)

    def next_t(self, t) -> torch.Tensor:
        """Look up the next grid point without subtraction round-off."""
        if not self.timesteps.numel():
            raise RuntimeError("call set_timesteps before next_t")
        t = torch.as_tensor(t, device=self.timesteps.device)
        index = torch.searchsorted(-self.timesteps, -t.contiguous())
        index = index.clamp(max=self.timesteps.numel() - 1)
        return self._next_timesteps[index]

    def add_noise(self, x0, noise, t):
        t_b = self._view(t, x0)
        return (1.0 - t_b) * x0 + t_b * (self.noise_scale * noise)

    def predict_x0(self, x_t, u, t):
        """One-step clean estimate x0 = x_t - t u (Tweedie analog)."""
        t_b = self._view(t, x_t)
        return x_t - t_b * u

    def step(self, model_output, t, x_t, eta: float = 0.0):
        u = model_output
        t_b = self._view(t, x_t)
        r_b = self._view(self.next_t(t), x_t)
        x0 = x_t - t_b * u
        eta_impl = (x_t - (1.0 - t_b) * x0) / t_b.clamp(min=1e-12)
        if eta > 0.0:
            noise = torch.randn_like(x_t) * self.noise_scale
            c1 = (1.0 - eta ** 2) ** 0.5
            eta_eff = c1 * eta_impl + eta * noise
        else:
            eta_eff = eta_impl
        x_prev = (1.0 - r_b) * x0 + r_b * eta_eff
        self.last_x0_coeff = 1.0 - r_b
        self._last_noise_term = r_b * eta_eff
        return x_prev, x0

    def reencode_x0(self, x0):
        """Replace the last anchor while keeping that step's sampled noise.

        Direct re-encoding avoids cancellation in x_prev + c*(x_dc-x0),
        which matters for calibrated CT data with a large dynamic range.
        """
        if self._last_noise_term is None:
            raise RuntimeError("call step before reencode_x0")
        return self.last_x0_coeff * x0 + self._last_noise_term
