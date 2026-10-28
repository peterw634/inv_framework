"""Adapter that plugs a real pixelMeanFlow (pMF) model into the DDS/DDS-LoRA
solvers. Optional: this module only touches the pMF codebase when you actually
construct a `PMFVelocityPrior`, mirroring `diffusers_compat.py` — so importing
it never drags in the pMF DiT architecture, `pytorch_msssim`, or `lpips`.

Typical use (industrial CT, real prior)::

    import sys; sys.path.insert(0, '/home/xujb/diffusion/indus_CT/NaSn')
    from pmf import pixelMeanFlow

    pmf = pixelMeanFlow('pmfDiT_B_64', img_size=1024, img_channels=1,
                        num_classes=1, eval=True).to(device)
    pmf.net.load_state_dict(torch.load('ckpt_epoch_999.pt'))

    from inv_framework.solvers.diffusion.pmf_compat import PMFVelocityPrior
    from inv_framework.solvers.diffusion.dds import DDSSolver
    from inv_framework.operators.ct.truncated import TruncatedRadon2D

    model = PMFVelocityPrior(pmf, omega=1.0, t_min=0.0, t_max=1.0)
    A = TruncatedRadon2D(1024, num_angles=900, inner=205, device=device)
    x = DDSSolver(model, num_inference_steps=16, dc_lam=0.05, dc_iter=5).solve(y, A)

For OOD adaptation, LoRA-inject ``pmf.net`` with `lora.adapt_pmfdit` and use
`dds_lora.LoRAAdaptiveDDSSolver`.
"""

import torch


class PMFVelocityPrior:
    """Wrap a `pixelMeanFlow` instance as a mean-flow velocity model.

    Exposes the ``model(x_t, t) -> u`` signature `DDSSolver` expects (the
    mean-flow analog of `DiffusersUNetWrapper`), baking in the classifier-free
    guidance scalars and labels. The clean-estimate / step arithmetic lives in
    `MeanFlowSchedule`; this wrapper only returns the average velocity
    ``u = pmf.u_fn(x_t, t, t, omega, t_min, t_max, y)[0]``.

    Parameters
    ----------
    pmf : pixelMeanFlow
        A constructed pMF wrapper (exposes ``u_fn``, ``net``, ``noise_scale``,
        ``img_size``, ``img_channels``, ``dtype``).
    omega, t_min, t_max : float
        Classifier-free-guidance scalars forwarded to ``u_fn``.
    labels : torch.Tensor, optional
        Class labels (default: zeros).
    """

    def __init__(self, pmf, omega: float = 1.0, t_min: float = 0.0,
                 t_max: float = 1.0, labels=None):
        for attr in ("u_fn", "net", "noise_scale", "img_size", "img_channels"):
            if not hasattr(pmf, attr):
                raise TypeError(
                    f"PMFVelocityPrior expects a pixelMeanFlow-like object with "
                    f"a `{attr}` attribute; got {type(pmf).__name__}.")
        self.pmf = pmf
        self.net = pmf.net
        self.img_size = int(pmf.img_size)
        self.img_channels = int(pmf.img_channels)
        self.noise_scale = float(pmf.noise_scale)
        self.omega = omega
        self.t_min = t_min
        self.t_max = t_max
        self._labels = labels

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B = x_t.shape[0]
        dtype, device = x_t.dtype, x_t.device
        if not torch.is_tensor(t):
            t = torch.full((B,), float(t), device=device, dtype=dtype)
        elif t.ndim == 0:
            t = t.expand(B)
        labels = self._labels
        if labels is None:
            labels = torch.zeros(B, dtype=torch.int32, device=device)
        else:
            labels = labels.to(device)
        omega = torch.as_tensor(self.omega, dtype=dtype, device=device).expand(B)
        t_min = torch.as_tensor(self.t_min, dtype=dtype, device=device).expand(B)
        t_max = torch.as_tensor(self.t_max, dtype=dtype, device=device).expand(B)
        # mean-flow average velocity: h = t (one-step anchor).
        return self.pmf.u_fn(x_t, t, t, omega, t_min, t_max, y=labels)[0]

    def __call__(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.forward(x_t, t)

    def eval(self):
        self.net.eval()
        return self

    def parameters(self, recurse: bool = True):
        return self.net.parameters(recurse=recurse)
