"""Regression tests for the correctness fixes of 2026-11-13 (CPU only)."""
import json

import pytest
import torch

from inv_framework.operators.base import LinearOperator, operator_norm_sq
from inv_framework.operators.ct.radon_torch import ParallelBeamRadon2D
from inv_framework.operators.ct.truncated import TruncatedRadon2D
from inv_framework.solvers.cg import cg_data_consistency
from inv_framework.solvers.classical import landweber


def _dot(a, b):
    return float((a.double() * b.double()).sum())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_torch_radon_adjoint_is_exact_transpose(dtype):
    torch.manual_seed(0)
    A = ParallelBeamRadon2D(48, 37)
    x = torch.randn(2, *A.domain_shape, dtype=dtype)
    y = torch.randn(2, *A.range_shape, dtype=dtype)
    lhs, rhs = _dot(A(x), y), _dot(x, A.adjoint(y))
    assert abs(lhs - rhs) / abs(lhs) < (1e-5 if dtype == torch.float32 else 1e-12)
    assert A(x).dtype == dtype


def test_torch_radon_autograd_matches_adjoint():
    torch.manual_seed(0)
    A = ParallelBeamRadon2D(32, 20)
    x = torch.randn(1, *A.domain_shape, requires_grad=True)
    y = torch.randn(1, *A.range_shape)
    (grad,) = torch.autograd.grad((A(x) * y).sum(), x)
    torch.testing.assert_close(grad, A.adjoint(y), rtol=1e-5, atol=1e-5)


def test_landweber_default_step_converges():
    torch.manual_seed(0)
    A = ParallelBeamRadon2D(64, 30)
    x = torch.rand(1, *A.domain_shape)
    y = A(x)
    residuals = [float((A(landweber(A, y, num_iterations=n)) - y).norm()) for n in (5, 20, 50)]
    assert all(r == r and r < float(y.norm()) for r in residuals)
    assert residuals[0] > residuals[1] > residuals[2]


def test_project_default_step_reduces_residual():
    torch.manual_seed(0)
    A = ParallelBeamRadon2D(64, 30)
    y = A(torch.rand(1, *A.domain_shape))
    x = torch.zeros(1, *A.domain_shape)
    assert (A(A.project(x, y)) - y).norm() < (A(x) - y).norm()


def test_operator_norm_is_cached_and_close_to_power_iteration():
    A = ParallelBeamRadon2D(32, 16)
    first = operator_norm_sq(A)
    assert operator_norm_sq(A) == first
    v = torch.randn(1, *A.domain_shape)
    for _ in range(100):
        v = A.adjoint(A(v))
        v = v / v.norm()
    assert float(A.adjoint(A(v)).norm()) <= first


def test_truncated_sirt_does_not_fit_unmeasured_columns_as_zero():
    torch.manual_seed(0)
    A = TruncatedRadon2D(64, 60, inner=32)
    yy, xx = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
    disk = (((yy - 31.5) ** 2 + (xx - 31.5) ** 2) < 26 ** 2).float()[None, None]
    y = A(disk)
    x = A.pseudo_inverse(y, method="sirt", num_iterations=100)
    roi = ((yy - 31.5) ** 2 + (xx - 31.5) ** 2) < 12 ** 2
    error = float((x[0, 0][roi] - 1).abs().mean())
    untruncated = A.radon.pseudo_inverse(y, method="sirt", num_iterations=100)
    assert error < float((untruncated[0, 0][roi] - 1).abs().mean())


class _Diagonal(LinearOperator):
    def __init__(self, shape):
        self.domain_shape = self.range_shape = shape
        self.weight = torch.linspace(0.5, 2.0, int(torch.tensor(shape).prod())).reshape(shape)

    def forward(self, x):
        return x * self.weight

    def adjoint(self, y):
        return y * self.weight


def test_cg_handles_volumes_and_batches():
    A = _Diagonal((3, 4, 5))
    x = torch.rand(2, 3, 4, 5)
    out = cg_data_consistency(torch.zeros_like(x), A(x), A, lam=0.0, max_iter=60, tol=1e-12)
    torch.testing.assert_close(out, x, rtol=1e-4, atol=1e-4)


def test_inr_fits_each_batch_item_separately():
    from inv_framework.solvers.inr import INRSolver
    torch.manual_seed(0)
    A = ParallelBeamRadon2D(16, 8)
    x = torch.stack([torch.zeros(1, 16, 16), torch.ones(1, 16, 16)])
    out = INRSolver(num_iterations=150, lr=1e-3, hidden_features=32, hidden_layers=2).solve(A(x), A)
    assert float(out[1].mean() - out[0].mean()) > 0.5


def test_dip_ema_returns_average_of_iterates():
    from inv_framework.solvers.dip import DIPSolver
    torch.manual_seed(0)
    A = ParallelBeamRadon2D(16, 8)
    y = A(torch.rand(1, 1, 16, 16))
    seen = []
    out = DIPSolver(num_iterations=3, ema_decay=0.5, input_channels=4,
                    callback=lambda it, x, loss: seen.append(x)).solve(y, A)
    expected = 0.5 * (0.5 * seen[0] + 0.5 * seen[1]) + 0.5 * seen[2]
    torch.testing.assert_close(out, expected)


def test_vp_schedule_last_step_reaches_clean_sample():
    from inv_framework.solvers.diffusion import VPSchedule
    schedule = VPSchedule(num_train_timesteps=1000)
    schedule.set_timesteps(10)
    x0 = torch.rand(1, 1, 4, 4)
    noise = torch.randn_like(x0)
    x = schedule.add_noise(x0, noise, int(schedule.timesteps[0]))
    for t in schedule.timesteps:
        a = schedule._alpha_bar_view(int(t))
        eps = (x - a.sqrt() * x0) / (1 - a).sqrt()      # oracle noise prediction
        x, _ = schedule.step(eps, t, x, eta=0.0)
    torch.testing.assert_close(x, x0, rtol=1e-4, atol=1e-4)


def test_meanflow_eta_noise_follows_generator():
    from inv_framework.solvers.diffusion import DDSSolver
    from inv_framework.solvers.diffusion.velocity_prior import TinyVelocityPrior
    A = ParallelBeamRadon2D(16, 8)
    y = A(torch.rand(1, 1, 16, 16))
    prior = TinyVelocityPrior(img_size=16, base=4)
    solver = DDSSolver(prior, num_inference_steps=3, eta=0.5)
    outs = []
    for global_seed in (1, 2):
        torch.manual_seed(global_seed)
        outs.append(solver.solve(y, A, generator=torch.Generator().manual_seed(0)))
    torch.testing.assert_close(outs[0], outs[1])


def test_lora_dds_uses_initial_noise():
    from inv_framework.solvers.diffusion import DDSSolver, LoRAAdaptiveDDSSolver
    from inv_framework.solvers.diffusion.velocity_prior import TinyVelocityPrior
    torch.manual_seed(0)
    A = ParallelBeamRadon2D(16, 8)
    y = A(torch.rand(1, 1, 16, 16))
    prior = TinyVelocityPrior(img_size=16, base=4)
    noise = torch.randn(1, 1, 16, 16)
    reference = DDSSolver(prior, num_inference_steps=3, dc_lam=0.05, dc_iter=5).solve(
        y, A, initial_noise=noise)
    torch.manual_seed(123)
    lora = LoRAAdaptiveDDSSolver(prior, adapt_at_steps=[], num_inference_steps=3,
                                 dc_lam=0.05, dc_iter=5).solve(y, A, initial_noise=noise)
    torch.testing.assert_close(lora, reference, rtol=1e-5, atol=1e-5)


def test_lora_groups_are_reusable_lists():
    from inv_framework.solvers.diffusion.lora import inject_lora_linear
    net = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
    groups = inject_lora_linear(net, r=2)
    assert all(isinstance(g, list) for g in groups)
    first = sum(p.numel() for g in groups for p in g)
    assert first > 0 and first == sum(p.numel() for g in groups for p in g)


def test_download_rejects_manifest_paths_outside_destination(tmp_path):
    from inv_framework.applications.nasn_ct.data.download import fetch
    manifest = {"schema_version": 1, "published": True, "profiles": {"p": ["a"]},
                "archives": {"a": {"filename": "../escaped.tar", "sha256": "0" * 64,
                                   "direct_url": "http://127.0.0.1:9/never"}}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsafe archive path"):
        fetch(path, tmp_path / "out", profile="p")
    assert not (tmp_path / "escaped.tar").exists()
