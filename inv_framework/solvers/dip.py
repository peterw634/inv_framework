"""Deep Image Prior solver.

Fit a CNN f_theta(z) (fixed random z) so that A f_theta(z) matches y. The
network's implicit smoothness prior regularises the otherwise underdetermined
inverse problem. Reference: Ulyanov et al., 2018.
"""

from typing import Callable, Optional
import torch
import torch.nn as nn

from ..operators.base import ForwardOperator
from ..models.skip_unet import SkipUNet
from .base import InverseProblemSolver


class DIPSolver(InverseProblemSolver):
    """Deep Image Prior.

    Parameters
    ----------
    model_factory : callable, optional
        Returns an nn.Module mapping z -> x. Defaults to SkipUNet.
    num_iterations : int
    lr : float
    input_channels : int
        Channels of the fixed random input z.
    ema_decay : float, optional
        Return an exponential moving average of the iterates instead of the
        last one (the DIP reference code uses 0.99). Smooths the occasional
        Adam spikes that make the last iterate unreliable.
    log_every : int, optional
    callback : callable(it, x_est, loss), optional
    """

    def __init__(self,
                 model_factory: Optional[Callable[[], nn.Module]] = None,
                 num_iterations: int = 2000,
                 lr: float = 1e-3,
                 input_channels: int = 32,
                 log_every: Optional[int] = None,
                 callback: Optional[Callable] = None,
                 ema_decay: Optional[float] = None):
        self.model_factory = model_factory
        self.num_iterations = num_iterations
        self.lr = lr
        self.input_channels = input_channels
        self.ema_decay = ema_decay
        self.log_every = log_every
        self.callback = callback

    def solve(self,
              measurement: torch.Tensor,
              operator: ForwardOperator,
              **kwargs) -> torch.Tensor:
        device = measurement.device
        B = measurement.shape[0]
        C, H, W = operator.domain_shape

        if self.model_factory is None:
            model = SkipUNet(in_channels=self.input_channels,
                             out_channels=C).to(device)
        else:
            model = self.model_factory().to(device)

        z = torch.randn(B, self.input_channels, H, W, device=device)
        optim = torch.optim.Adam(model.parameters(), lr=self.lr)
        x_avg = None

        for it in range(self.num_iterations):
            optim.zero_grad()
            x_est = model(z)
            y_est = operator.forward(x_est)
            loss = ((y_est - measurement) ** 2).mean()
            loss.backward()
            optim.step()
            if self.ema_decay is not None:
                x_avg = (x_est.detach() if x_avg is None else
                         self.ema_decay * x_avg + (1 - self.ema_decay) * x_est.detach())

            if self.log_every and (it % self.log_every == 0
                                   or it == self.num_iterations - 1):
                print(f"[DIP] iter {it}/{self.num_iterations} "
                      f"loss={loss.item():.4e}")
            if self.callback is not None:
                self.callback(it, x_est.detach(), loss.item())

        if x_avg is not None:
            return x_avg
        with torch.no_grad():
            return model(z)
