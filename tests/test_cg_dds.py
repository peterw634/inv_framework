"""Tests for CG data-consistency, the DDS solver, and LoRA-adaptive DDS.

Uses the small `TinyVelocityPrior` so the full sparse-view / truncated CT
pipeline runs end-to-end on CPU without any pMF checkpoint.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inv_framework.operators.ct.truncated import TruncatedRadon2D
from inv_framework.operators.ct.radon_torch import ParallelBeamRadon2D
from inv_framework.solvers.cg import cg_data_consistency, cg_data_consistency_grad
from inv_framework.solvers.diffusion.velocity_prior import TinyVelocityPrior
from inv_framework.solvers.diffusion.dds import DDSSolver
from inv_framework.solvers.diffusion.dds_lora import LoRAAdaptiveDDSSolver
from inv_framework.solvers.diffusion.lora import inject_lora_linear


def _disk(N, radius):
    x = torch.zeros(1, 1, N, N)
    yy, xx = torch.meshgrid(torch.arange(N), torch.arange(N), indexing='ij')
    x[0, 0] = ((yy - N / 2) ** 2 + (xx - N / 2) ** 2 < radius ** 2).float()
    return x


def test_cg_reduces_measurement_residual():
    torch.manual_seed(0)
    N, n_ang, inner = 64, 40, 32
    A = TruncatedRadon2D(N, n_ang, inner=inner)
    x_true = _disk(N, inner / 2 - 3)
    y = A.forward(x_true)

    x0 = torch.zeros_like(x_true)
    x_dc, hist = cg_data_consistency(x0, y, A, lam=1e-3, max_iter=30,
                                     tol=1e-8, return_history=True)
    res0 = (A.forward(x0) - y).norm().item()
    res1 = (A.forward(x_dc) - y).norm().item()
    print(f"[cg] residual {res0:.3e} -> {res1:.3e} ({res0/max(res1,1e-30):.0f}x); "
          f"rel {hist[0]:.2e} -> {hist[-1]:.2e}")
    assert res1 < res0 / 100
    assert hist[-1] < hist[0]


def test_cg_grad_flows():
    torch.manual_seed(0)
    N, n_ang = 32, 20
    A = ParallelBeamRadon2D(N, n_ang)
    x = torch.randn(1, 1, N, N, requires_grad=True)
    y = torch.randn(1, 1, n_ang, N)
    x_dc = cg_data_consistency_grad(x, y, A, lam=0.1, max_iter=3)
    loss = (A.forward(x_dc) - y).pow(2).mean()
    g, = torch.autograd.grad(loss, x)
    print(f"[cg grad] grad norm = {g.norm().item():.3e}")
    assert torch.isfinite(g).all() and g.norm().item() > 0.0


def test_dds_solver_runs_and_is_data_consistent():
    torch.manual_seed(0)
    N, n_ang, inner = 32, 20, 24
    A = TruncatedRadon2D(N, n_ang, inner=inner)
    prior = TinyVelocityPrior(img_channels=1, img_size=N, base=16)
    x_true = _disk(N, inner / 2 - 2)
    y = A.forward(x_true)

    g = torch.Generator().manual_seed(0)
    solver = DDSSolver(prior, num_steps=8, dc_lam=0.02, dc_iter=5)
    x_rec = solver.solve(y, A, generator=g)
    assert x_rec.shape == (1, 1, N, N)
    assert torch.isfinite(x_rec).all()
    res = (A.forward(x_rec) - y).norm().item()
    print(f"[dds] final residual {res:.3e} vs ||y|| {y.norm().item():.3e}")
    assert res < y.norm().item()  # data consistency pulled it well below ||y||


def test_lora_adaptive_dds_trains_and_runs():
    torch.manual_seed(0)
    N, n_ang, inner = 32, 20, 24
    A = TruncatedRadon2D(N, n_ang, inner=inner)
    prior = TinyVelocityPrior(img_channels=1, img_size=N, base=16)
    groups = inject_lora_linear(prior.net, r=4)
    adapt_params = [p for gp in groups for p in gp]
    before = sum(p.detach().pow(2).sum().item() for p in adapt_params)

    x_true = _disk(N, inner / 2 - 2)
    y = A.forward(x_true)

    def make_optim(params):
        return torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-2)

    g = torch.Generator().manual_seed(0)
    solver = LoRAAdaptiveDDSSolver(
        prior, adapt_params=adapt_params, adapt_optim_factory=make_optim,
        adapt_at_steps=None, adapt_K=2, num_steps=6, dc_lam=0.02, dc_iter=4)
    x_rec = solver.solve(y, A, generator=g)
    after = sum(p.detach().pow(2).sum().item() for p in adapt_params)
    print(f"[lora-dds] adapt L2 {before:.2e} -> {after:.2e}; "
          f"residual {(A.forward(x_rec)-y).norm().item():.3e}")
    assert x_rec.shape == (1, 1, N, N) and torch.isfinite(x_rec).all()
    assert after > before  # LoRA params were actually trained in-loop


def test_lora_adaptive_dds_rejects_nonlinear_operator():
    from inv_framework.operators.base import ForwardOperator

    class _Sq(ForwardOperator):
        def __init__(self, N, n_ang):
            self.domain_shape = (1, N, N)
            self.range_shape = (1, n_ang, N)
            self._R = ParallelBeamRadon2D(N, n_ang)
        def forward(self, x):
            return self._R.forward(x) ** 2

    N, n_ang = 32, 20
    prior = TinyVelocityPrior(img_channels=1, img_size=N, base=16)
    inject_lora_linear(prior.net, r=4)
    op = _Sq(N, n_ang)
    y = torch.randn(1, 1, n_ang, N)
    solver = LoRAAdaptiveDDSSolver(
        prior, adapt_params=[torch.zeros(1, requires_grad=True)],
        adapt_optim_factory=lambda p: torch.optim.SGD(p, lr=0.0),
        adapt_at_steps=[], num_steps=2)
    try:
        solver.solve(y, op)
    except TypeError as e:
        print(f"[reject] {str(e)[:60]}...")
        return
    raise AssertionError("expected TypeError for non-linear operator")


if __name__ == "__main__":
    test_cg_reduces_measurement_residual()
    test_cg_grad_flows()
    test_dds_solver_runs_and_is_data_consistent()
    test_lora_adaptive_dds_trains_and_runs()
    test_lora_adaptive_dds_rejects_nonlinear_operator()
    print("\nAll CG/DDS/LoRA-DDS tests passed.")
