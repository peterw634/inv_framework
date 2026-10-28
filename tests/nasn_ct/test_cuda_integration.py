"""Native CUDA operator contracts, including differentiable CG for LoRA."""

import numpy as np
import pytest
import torch

from inv_framework.operators.ct import TomoPyCUDARadon2D
from inv_framework.solvers.cg import cg_data_consistency_grad

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")


def test_adjoint_identity_and_mask():
    torch.manual_seed(7)
    op = TomoPyCUDARadon2D(32, 25, 14, center=15.2)
    x = torch.randn(2, 1, 32, 32, device="cuda")
    y = torch.randn(2, 1, 25, 32, device="cuda")
    Ax, ATy = op(x), op.adjoint(y)
    left, right = (Ax * y).sum(), (x * ATy).sum()
    torch.testing.assert_close(left, right, rtol=1e-4, atol=2e-4)
    assert Ax[..., :9].count_nonzero() == 0
    assert Ax[..., 23:].count_nonzero() == 0


def test_forward_and_adjoint_backward_contracts():
    torch.manual_seed(8)
    op = TomoPyCUDARadon2D(32, 25, 14)
    x = torch.randn(1, 1, 32, 32, device="cuda", requires_grad=True)
    y = torch.randn(1, 1, 25, 32, device="cuda", requires_grad=True)
    saved = y.detach().clone()
    grad_x, = torch.autograd.grad((op(x) * y).sum(), x)
    torch.testing.assert_close(grad_x, op.adjoint(y), rtol=1e-5, atol=2e-5)
    grad_y, = torch.autograd.grad((op.adjoint(y) * x).sum(), y)
    torch.testing.assert_close(grad_y, op(x), rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(y.detach(), saved, rtol=0, atol=0)


def test_cg_gradient_matches_finite_difference():
    torch.manual_seed(9)
    op = TomoPyCUDARadon2D(16, 16, 8)
    x = torch.randn(1, 1, 16, 16, device="cuda", requires_grad=True)
    y = torch.randn(1, 1, 16, 16, device="cuda")
    direction = torch.randn_like(x)
    direction /= direction.norm()
    probe = torch.randn_like(x)

    def loss(value):
        corrected = cg_data_consistency_grad(value, y, op, lam=2., max_iter=3)
        return (corrected * probe).sum()

    gradient, = torch.autograd.grad(loss(x), x)
    analytical = (gradient * direction).sum()
    epsilon = .02
    with torch.no_grad():
        numerical = (loss(x + epsilon * direction) - loss(x - epsilon * direction)) / (2 * epsilon)
    torch.testing.assert_close(analytical, numerical, rtol=3e-2, atol=3e-3)
    assert torch.isfinite(gradient).all() and gradient.norm() > 0


def test_gridrec_matches_tomopy():
    tomopy = pytest.importorskip("tomopy")
    from inv_framework.operators.ct.radon_cuda import gridrec_cuda
    sinogram = np.random.default_rng(7).normal(size=(2, 45, 64)).astype(np.float32)
    theta = np.linspace(0, np.pi, 45, endpoint=False, dtype=np.float32)
    expected = tomopy.recon(sinogram.transpose(1, 0, 2), theta,
                            center=32., algorithm="gridrec", filter_name="shepp")
    actual = gridrec_cuda(torch.from_numpy(sinogram).cuda(), theta, 32.).cpu().numpy()
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) <= 1.5e-5
