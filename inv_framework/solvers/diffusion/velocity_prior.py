"""Velocity-field prior interface for mean-flow / flow-matching solvers.

Mean-flow models (e.g. pixelMeanFlow / pMF) are not epsilon-predictors: they
predict an *average velocity* ``u`` along a straight-line interpolant
``z_t = (1 - t) x_0 + t * noise``, so the one-step clean estimate is

    x_0_hat = z_t - t * u(z_t, t)

The DDS sampler in this framework depends only on this ``predict_x0`` map, not
on any particular model. `VelocityPrior` is that minimal interface; concrete
priors (the real pMF via `pmf_compat.PMFVelocityPrior`, or the small
`TinyVelocityPrior` below for tests/demos) implement it.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from ...models.tiny_unet import TinyUNet


class VelocityPrior(ABC):
    """Abstract mean-flow prior.

    Attributes
    ----------
    img_channels : int
    img_size : int
    noise_scale : float
        Std of the terminal noise ``z_1 = noise_scale * N(0, I)``; DDS uses it
        to initialise the reverse process and to mix DDIM stochasticity.
    """

    img_channels: int
    img_size: int
    noise_scale: float = 1.0

    @abstractmethod
    def predict_x0(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """One-step clean estimate x_0_hat = z - t * u(z, t).

        ``z`` is (B, C, H, W); ``t`` is a per-sample tensor of shape (B,).
        Must be autograd-traceable through any trainable parameters (LoRA)
        while allowing ``z`` to be detached by the caller.
        """

    def parameters(self):
        """Trainable parameters (empty by default). Overridden by nn-backed
        priors so the DDS solver can locate the device."""
        return iter(())


class TinyVelocityPrior(VelocityPrior, nn.Module):
    """A small `TinyUNet`-backed velocity field for tests and demos.

    Lets the DDS / CG / LoRA pipeline run end-to-end with no pMF checkpoint.
    ``predict_x0`` treats the network output as the average velocity ``u`` and
    returns ``z - t * u``.
    """

    def __init__(self, img_channels: int = 1, img_size: int = 64,
                 base: int = 32, noise_scale: float = 1.0):
        nn.Module.__init__(self)
        self.img_channels = int(img_channels)
        self.img_size = int(img_size)
        self.noise_scale = float(noise_scale)
        self.net = TinyUNet(channels=img_channels, base=base)

    def u_fn(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # TinyUNet scales t into a timestep-embedding range for stability.
        return self.net(z, t * 1000.0)

    def predict_x0(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_b = t.view(-1, 1, 1, 1).to(z.dtype)
        u = self.u_fn(z, t)
        return z - t_b * u

    def parameters(self, recurse: bool = True):
        return nn.Module.parameters(self, recurse=recurse)
