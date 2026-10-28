"""DDS sampler with in-the-loop LoRA adaptation (DDIP3D-aligned).

Ported from ``indus_CT/NaSn/recon/dds_sampler_lora.py``. Extends `DDSSolver`
with online out-of-distribution adaptation: at (a schedule of) reverse steps, a
few Adam steps update LoRA parameters of the velocity model to reduce the
measurement residual, optionally regularised toward an in-distribution batch.
Aligned with DDIP3D (Chung & Ye, 2407.10641):

  * dense per-step adaptation by default (every step in ``adapt_at_steps``);
  * per-timestep optimizer reset (fresh Adam moments each adapt step);
  * optional in-distribution regularisation (``reg_gamma > 0`` + ``id_sampler``);
  * optional SCD scale toggle (``use_scale_toggle``): predict the implicit eta
    from the base model (LoRA off) and x0 from the adapted model (LoRA on).

Uses the same model + `MeanFlowSchedule` abstractions as `DDSSolver` (velocity
``u = model(x_t, t)``; ``x0 = schedule.predict_x0(x_t, u, t)``). The model must
expose a ``net`` submodule that has been LoRA-injected (see
`lora.adapt_pmfdit` / `lora.inject_lora_linear`) so `set_lora_scale` can toggle
it; ``adapt_params`` are the parameters the optimizer trains.
"""

from typing import Callable, List, Optional

import torch

from ...operators.base import ForwardOperator, LinearOperator
from ..cg import cg_data_consistency, cg_data_consistency_grad
from .dds import DDSSolver
from .lora import set_lora_scale


__all__ = ["LoRAAdaptiveDDSSolver"]


def _id_reg_loss(model, schedule, x_id: torch.Tensor,
                 t: torch.Tensor) -> torch.Tensor:
    """DDIP3D-style in-distribution regularisation for a mean-flow model.

    Noise x_id to level t, predict x0_id = predict_x0(z_id, model(z_id, t), t),
    return the reconstruction MSE ``||x0_id - x_id||^2``. Keeps adaptation from
    drifting off the model's learned manifold.
    """
    noise_scale = float(getattr(model, "noise_scale", 1.0))
    t_b = t.view(-1, 1, 1, 1).to(x_id.dtype)
    eps = torch.randn_like(x_id) * noise_scale
    z_id = (1.0 - t_b) * x_id + t_b * eps
    x0_id = schedule.predict_x0(z_id, model(z_id, t), t)
    return ((x0_id - x_id) ** 2).mean()


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
                 model,
                 adapt_params: Optional[List[torch.nn.Parameter]] = None,
                 adapt_optim_factory: Optional[Callable] = None,
                 adapt_at_steps: Optional[List[int]] = None,
                 adapt_K: int = 4,
                 reset_optim_per_timestep: bool = True,
                 id_sampler: Optional[Callable[[], torch.Tensor]] = None,
                 reg_gamma: float = 0.0,
                 use_scale_toggle: bool = False,
                 num_inference_steps: int = 16,
                 eta: float = 0.0,
                 dc_lam: float = 0.05,
                 dc_iter: int = 10,
                 t_start: float = 1.0,
                 verbose: bool = False,
                 **dds_kwargs):
        super().__init__(model, num_inference_steps=num_inference_steps,
                         eta=eta, dc_mode="dds", dc_lam=dc_lam, dc_iter=dc_iter,
                         t_start=t_start, **dds_kwargs)
        self.adapt_params = adapt_params
        self.adapt_optim_factory = adapt_optim_factory
        self.adapt_at_steps = adapt_at_steps
        self.adapt_K = adapt_K
        self.reset_optim_per_timestep = reset_optim_per_timestep
        self.id_sampler = id_sampler
        self.reg_gamma = reg_gamma
        self.use_scale_toggle = use_scale_toggle
        self.verbose = verbose

    def _net(self):
        return getattr(self.model, "net", None)

    def _has_lora(self) -> bool:
        net = self._net()
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
            steps = list(range(self.num_inference_steps))
        adapt_set = set(int(i) for i in steps)
        if adapt_set:
            if self.adapt_params is None or self.adapt_optim_factory is None:
                raise ValueError("adapt_at_steps non-empty but adapt_params or "
                                 "adapt_optim_factory is None.")
            if not self._has_lora():
                raise RuntimeError("adapt requested but model.net has no LoRA "
                                   "modules — inject LoRA first.")

        net = self._net()
        model, schedule = self.model, self.schedule
        use_id = self.reg_gamma > 0.0 and self.id_sampler is not None
        device = self._device(measurement)
        measurement = measurement.to(device)
        B = measurement.shape[0]

        schedule.set_timesteps(self.num_inference_steps, device=device)

        def randn(shape):
            return torch.randn(shape, device=device,
                               generator=generator) * self.noise_scale

        with torch.no_grad():
            eta0 = randn((B, *operator.domain_shape))
            if x_init is None:
                x_t = self.t_start * eta0
            else:
                x_t = (1.0 - self.t_start) * x_init.to(device) + self.t_start * eta0

        optim = None
        if adapt_set and not self.reset_optim_per_timestep:
            optim = self.adapt_optim_factory(self.adapt_params)

        for i, t in enumerate(schedule.timesteps):
            t_b_scalar = float(t.item()) if torch.is_tensor(t) else float(t)
            t = torch.full((B,), t_b_scalar, device=device)
            t_b = t.view(-1, 1, 1, 1)
            r_b = schedule.next_t(t).view(-1, 1, 1, 1)

            # (1) implicit eta from the base model (optional scale toggle).
            if self.use_scale_toggle and net is not None:
                set_lora_scale(net, 0.0)
            with torch.no_grad():
                x0_eta = schedule.predict_x0(x_t, model(x_t, t), t)
                eta_impl = (x_t - (1.0 - t_b) * x0_eta) / t_b.clamp(min=1e-12)

            # (2) inner adapt loop.
            if i in adapt_set:
                if net is not None:
                    set_lora_scale(net, 1.0)
                if self.reset_optim_per_timestep:
                    optim = self.adapt_optim_factory(self.adapt_params)
                z_anchor = x_t.detach()
                for k in range(self.adapt_K):
                    optim.zero_grad(set_to_none=True)
                    if use_id:
                        x_id = self.id_sampler().to(device)
                        with torch.enable_grad():
                            L_reg = _id_reg_loss(model, schedule, x_id, t) * self.reg_gamma
                        L_reg.backward()
                    with torch.enable_grad():
                        x0 = schedule.predict_x0(z_anchor, model(z_anchor, t), t)
                        x0_dc = cg_data_consistency_grad(
                            x0, measurement, operator,
                            lam=self.dc_lam, max_iter=self.dc_iter)
                        L_meas = ((operator.forward(x0_dc) - measurement) ** 2).mean()
                    L_meas.backward()
                    optim.step()
                    if self.verbose:
                        print(f"  step {i:02d} adapt {k+1}/{self.adapt_K} "
                              f"L_meas={L_meas.item():.4e}")

            # (3) standard DDS update with LoRA-on x0.
            if net is not None:
                set_lora_scale(net, 1.0)
            with torch.no_grad():
                x0 = schedule.predict_x0(x_t, model(x_t, t), t)
                x0_dc = cg_data_consistency(
                    x0, measurement, operator,
                    lam=self.dc_lam, max_iter=self.dc_iter)
                if self.eta > 0.0:
                    eta_fresh = randn(x_t.shape)
                    c1 = (1.0 - self.eta ** 2) ** 0.5
                    eta_eff = c1 * eta_impl + self.eta * eta_fresh
                else:
                    eta_eff = eta_impl
                x_t = (1.0 - r_b) * x0_dc + r_b * eta_eff

        return x_t
