"""DDS sampler with in-the-loop LoRA adaptation (DDIP3D-aligned).

Ported from ``indus_CT/NaSn/recon/dds_sampler_lora.py``. Extends `DDSSolver`
with online out-of-distribution adaptation: at (a schedule of) reverse steps,
a few Adam steps update LoRA parameters of the prior to reduce the measurement
residual, optionally regularised toward an in-distribution batch. Aligned with
DDIP3D (Chung & Ye, 2407.10641):

  * dense per-step adaptation by default (every step in ``adapt_at_steps``);
  * per-timestep optimizer reset (fresh Adam moments each adapt step);
  * optional in-distribution regularisation (``reg_gamma > 0`` + ``id_sampler``);
  * optional SCD scale toggle (``use_scale_toggle``): predict the implicit eta
    from the base model (LoRA off) and x_hat from the adapted model (LoRA on).

The prior must expose a ``net`` submodule that has been LoRA-injected (see
`lora.adapt_pmfdit` / `lora.inject_lora_linear`) so `set_lora_scale` can toggle
it; ``adapt_params`` are the parameters the optimizer trains.
"""

from typing import Callable, Iterable, List, Optional

import torch

from ...operators.base import ForwardOperator, LinearOperator
from ..cg import cg_data_consistency, cg_data_consistency_grad
from .dds import DDSSolver
from .lora import set_lora_scale
from .velocity_prior import VelocityPrior


__all__ = ["LoRAAdaptiveDDSSolver"]


def _id_reg_loss(prior: VelocityPrior, x_id: torch.Tensor,
                 t: torch.Tensor) -> torch.Tensor:
    """DDIP3D-style in-distribution regularisation for a mean-flow prior.

    Noise x_id to level t, predict x_hat_id = predict_x0(z_id, t), return the
    reconstruction MSE ``||x_hat_id - x_id||^2``. Keeps adaptation from drifting
    off the prior's learned manifold.
    """
    t_b = t.view(-1, 1, 1, 1).to(x_id.dtype)
    eps = torch.randn_like(x_id) * prior.noise_scale
    z_id = (1.0 - t_b) * x_id + t_b * eps
    x_hat_id = prior.predict_x0(z_id, t)
    return ((x_hat_id - x_id) ** 2).mean()


class LoRAAdaptiveDDSSolver(DDSSolver):
    """DDS reverse sampler with optional in-the-loop LoRA adaptation.

    Parameters (in addition to `DDSSolver`)
    ----------
    adapt_params : list of nn.Parameter
        Parameters the inner optimizer trains (LoRA up/down + gates).
    adapt_optim_factory : callable(params) -> Optimizer
        Builds the inner optimizer; called once (if
        ``reset_optim_per_timestep`` is False) or per adapt step (if True).
    adapt_at_steps : list of int, optional
        Reverse-step indices at which to adapt. ``None`` -> every step;
        empty list -> no adaptation (falls back to plain DDS).
    adapt_K : int
        Inner Adam steps per adapt timestep.
    reset_optim_per_timestep : bool
        Rebuild the optimizer each adapt timestep (DDIP3D default).
    id_sampler : callable() -> (B, C, H, W), optional
        Returns one in-distribution batch per call; enables ID regularisation.
    reg_gamma : float
        Weight of the ID regularisation loss (0 disables it).
    use_scale_toggle : bool
        If True, predict the implicit eta from the base model (LoRA scale 0).
    """

    def __init__(self,
                 prior: VelocityPrior,
                 adapt_params: Optional[List[torch.nn.Parameter]] = None,
                 adapt_optim_factory: Optional[Callable] = None,
                 adapt_at_steps: Optional[List[int]] = None,
                 adapt_K: int = 4,
                 reset_optim_per_timestep: bool = True,
                 id_sampler: Optional[Callable[[], torch.Tensor]] = None,
                 reg_gamma: float = 0.0,
                 use_scale_toggle: bool = False,
                 num_steps: int = 16,
                 dc_lam: float = 0.05,
                 dc_iter: int = 10,
                 dc_eta_stoch: float = 0.0,
                 t_start: float = 1.0,
                 verbose: bool = False,
                 **dds_kwargs):
        super().__init__(prior, num_steps=num_steps, dc_mode="dds",
                         dc_lam=dc_lam, dc_iter=dc_iter,
                         dc_eta_stoch=dc_eta_stoch, t_start=t_start,
                         **dds_kwargs)
        self.adapt_params = adapt_params
        self.adapt_optim_factory = adapt_optim_factory
        self.adapt_at_steps = adapt_at_steps
        self.adapt_K = adapt_K
        self.reset_optim_per_timestep = reset_optim_per_timestep
        self.id_sampler = id_sampler
        self.reg_gamma = reg_gamma
        self.use_scale_toggle = use_scale_toggle
        self.verbose = verbose

    def _has_lora(self) -> bool:
        net = getattr(self.prior, "net", None)
        if net is None:
            return False
        return any(m.__class__.__name__ in ("LoraInjectedLinear",
                                            "LoraInjectedConv2d")
                   for m in net.modules())

    def solve(self,
              measurement: torch.Tensor,
              operator: ForwardOperator,
              x_init: Optional[torch.Tensor] = None,
              generator: Optional[torch.Generator] = None,
              **kwargs) -> torch.Tensor:
        if not isinstance(operator, LinearOperator):
            raise TypeError(
                "LoRAAdaptiveDDSSolver requires a LinearOperator (CG data "
                f"consistency needs an adjoint); got {type(operator).__name__}.")

        steps = self.adapt_at_steps
        if steps is None:
            steps = list(range(self.num_steps))
        adapt_set = set(int(i) for i in steps)
        if adapt_set:
            if self.adapt_params is None or self.adapt_optim_factory is None:
                raise ValueError("adapt_at_steps non-empty but adapt_params or "
                                 "adapt_optim_factory is None.")
            if not self._has_lora():
                raise RuntimeError("adapt requested but prior.net has no LoRA "
                                   "modules — inject LoRA first.")

        net = getattr(self.prior, "net", None)
        use_id = self.reg_gamma > 0.0 and self.id_sampler is not None
        device = self._device(measurement)
        measurement = measurement.to(device)
        B = measurement.shape[0]
        C, H, W = self.prior.img_channels, self.prior.img_size, self.prior.img_size

        def randn(shape):
            return torch.randn(shape, device=device, generator=generator) \
                * self.prior.noise_scale

        with torch.no_grad():
            eta0 = randn((B, C, H, W))
            if x_init is None:
                z = self.t_start * eta0
            else:
                z = (1.0 - self.t_start) * x_init.to(device) + self.t_start * eta0
        t_steps = torch.linspace(float(self.t_start), 0.0, self.num_steps + 1,
                                 device=device)

        optim = None
        if adapt_set and not self.reset_optim_per_timestep:
            optim = self.adapt_optim_factory(self.adapt_params)

        for i in range(self.num_steps):
            t = t_steps[i].expand(B)
            r = t_steps[i + 1].expand(B)
            t_b = t.view(-1, 1, 1, 1)
            r_b = r.view(-1, 1, 1, 1)

            # (1) implicit eta from the base model (optional scale toggle).
            if self.use_scale_toggle and net is not None:
                set_lora_scale(net, 0.0)
            with torch.no_grad():
                x_hat_eta = self.prior.predict_x0(z, t)
                eta_implicit = (z - (1.0 - t_b) * x_hat_eta) / t_b.clamp(min=1e-12)

            # (2) inner adapt loop.
            if i in adapt_set:
                if net is not None:
                    set_lora_scale(net, 1.0)
                if self.reset_optim_per_timestep:
                    optim = self.adapt_optim_factory(self.adapt_params)
                z_anchor = z.detach()
                for k in range(self.adapt_K):
                    optim.zero_grad(set_to_none=True)
                    if use_id:
                        x_id = self.id_sampler().to(device)
                        with torch.enable_grad():
                            L_reg = _id_reg_loss(self.prior, x_id, t) * self.reg_gamma
                        L_reg.backward()
                    with torch.enable_grad():
                        x_hat = self.prior.predict_x0(z_anchor, t)
                        x_hat_dc = cg_data_consistency_grad(
                            x_hat, measurement, operator,
                            lam=self.dc_lam, max_iter=self.dc_iter)
                        L_meas = ((operator.forward(x_hat_dc) - measurement) ** 2).mean()
                    L_meas.backward()
                    optim.step()
                    if self.verbose:
                        print(f"  step {i:02d} adapt {k+1}/{self.adapt_K} "
                              f"L_meas={L_meas.item():.4e}")

            # (3) standard DDS update with LoRA-on x_hat.
            if net is not None:
                set_lora_scale(net, 1.0)
            with torch.no_grad():
                x_hat = self.prior.predict_x0(z, t)
                x_hat_dc = cg_data_consistency(
                    x_hat, measurement, operator,
                    lam=self.dc_lam, max_iter=self.dc_iter)
                if self.dc_eta_stoch > 0.0:
                    eta_fresh = randn(z.shape)
                    c1 = (1.0 - self.dc_eta_stoch ** 2) ** 0.5
                    eta_eff = c1 * eta_implicit + self.dc_eta_stoch * eta_fresh
                else:
                    eta_eff = eta_implicit
                z = (1.0 - r_b) * x_hat_dc + r_b * eta_eff

        return z
