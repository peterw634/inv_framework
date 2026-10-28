"""Velocity-field prior for mean-flow / flow-matching solvers.

Mean-flow models (e.g. pixelMeanFlow / pMF) are the flow-matching analog of an
epsilon-predictor: given a noisy ``x_t`` and time ``t`` they predict the
*average velocity* ``u`` along the straight-line interpolant
``x_t = (1 - t) x0 + t * noise``. The clean estimate is ``x0 = x_t - t u`` — but
that arithmetic lives in `MeanFlowSchedule`, so a velocity prior only has to
map ``(x_t, t) -> u``, exactly mirroring how `DPSSolver`'s model maps
``(x_t, t) -> eps``.

A velocity "model" for `DDSSolver` is therefore any callable with signature
``model(x_t, t) -> u`` that also exposes a ``noise_scale`` attribute (the std
of the terminal noise ``x_1 = noise_scale * N(0, I)``). `VelocityPrior` is the
optional ABC documenting that contract; `TinyVelocityPrior` is a small
`TinyUNet`-backed instance so the DDS / CG / LoRA pipeline runs end-to-end with
no pMF checkpoint. Real pMF models plug in via `pmf_compat.PMFVelocityPrior`.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from ...models.tiny_unet import TinyUNet


class VelocityPrior(ABC):
    """Optional base for a mean-flow velocity model ``(x_t, t) -> u``.

    Attributes
    ----------
    noise_scale : float
        Std of the terminal noise ``x_1 = noise_scale * N(0, I)``. `DDSSolver`
        and `MeanFlowSchedule` read it to initialise / mix noise.
    """

    noise_scale: float = 1.0

    @abstractmethod
    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the average velocity ``u`` at ``(x_t, t)``. ``t`` is a
        per-sample tensor of shape (B,) with values in [0, 1]."""

    def __call__(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.forward(x_t, t)


class TinyVelocityPrior(nn.Module):
    """A small `TinyUNet`-backed velocity model ``(x_t, t) -> u``.

    Lets the DDS / CG / LoRA pipeline run end-to-end with no pMF checkpoint.
    Its ``forward`` returns the velocity ``u`` directly; the clean estimate
    ``x0 = x_t - t u`` and the reverse step are handled by `MeanFlowSchedule`.
    Being an `nn.Module`, it is callable as ``model(x_t, t)`` just like the
    epsilon UNet used by `DPSSolver`.
    """

    def __init__(self, img_channels: int = 1, img_size: int = 64,
                 base: int = 32, noise_scale: float = 1.0):
        super().__init__()
        self.img_channels = int(img_channels)
        self.img_size = int(img_size)
        self.noise_scale = float(noise_scale)
        self.net = TinyUNet(channels=img_channels, base=base)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.full((x_t.shape[0],), float(t), device=x_t.device)
        elif t.ndim == 0:
            t = t.expand(x_t.shape[0])
        # TinyUNet expects a timestep-embedding-scale argument.
        return self.net(x_t, t * 1000.0)
