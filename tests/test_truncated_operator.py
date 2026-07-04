"""Tests for the truncated (interior-tomography) Radon operator.

Verifies the design invariant that makes A = T . R a drop-in LinearOperator:
  1. shape sanity
  2. autograd backward == adjoint (the framework's exact-adjoint convention)
  3. detector truncation actually zeros the out-of-band columns
  4. inner == image_size reduces exactly to ParallelBeamRadon2D
  5. DetectorTruncation is idempotent and self-adjoint
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inv_framework.operators.ct.truncated import (
    TruncatedRadon2D, DetectorTruncation)
from inv_framework.operators.ct.radon_torch import ParallelBeamRadon2D


def test_shapes_and_autograd_equals_adjoint():
    torch.manual_seed(0)
    N, n_ang, inner = 64, 30, 24
    A = TruncatedRadon2D(N, n_ang, inner=inner)
    x = torch.randn(2, 1, N, N)
    y = torch.randn(2, 1, n_ang, N)

    Ax, ATy = A.forward(x), A.adjoint(y)
    assert Ax.shape == (2, 1, n_ang, N), Ax.shape
    assert ATy.shape == (2, 1, N, N), ATy.shape

    # Framework convention: autograd backward of forward IS the adjoint.
    x2 = x.clone().requires_grad_(True)
    loss = 0.5 * ((A.forward(x2) - y) ** 2).sum()
    g_auto = torch.autograd.grad(loss, x2)[0]
    g_manual = A.adjoint(A.forward(x) - y)
    rel = (g_auto - g_manual).norm().item() / g_manual.norm().item()
    print(f"[autograd==adjoint] rel err = {rel:.3e}")
    assert rel < 1e-5, rel


def test_truncation_zeros_out_of_band():
    torch.manual_seed(0)
    N, n_ang, inner = 64, 30, 24
    A = TruncatedRadon2D(N, n_ang, inner=inner)
    x = torch.randn(1, 1, N, N)
    sino = A.forward(x)[0, 0]
    lo = (N - inner) // 2
    left = sino[:, :lo].abs().max().item()
    right = sino[:, lo + inner:].abs().max().item()
    print(f"[mask] out-of-band max = {left:.2e}, {right:.2e}")
    assert left == 0.0 and right == 0.0
    # in-band should be nonzero
    assert sino[:, lo:lo + inner].abs().max().item() > 0.0


def test_degenerate_reduces_to_radon():
    torch.manual_seed(0)
    N, n_ang = 64, 30
    A_full = TruncatedRadon2D(N, n_ang, inner=N)
    R = ParallelBeamRadon2D(N, n_ang)
    x = torch.randn(2, 1, N, N)
    d = (A_full.forward(x) - R.forward(x)).abs().max().item()
    print(f"[degenerate] inner==N vs Radon max|diff| = {d:.2e}")
    assert d == 0.0


def test_detector_truncation_self_adjoint_idempotent():
    torch.manual_seed(0)
    N, n_ang, inner = 48, 20, 16
    T = DetectorTruncation(n_ang, N, inner)
    y = torch.randn(2, 1, n_ang, N)
    # self-adjoint: <T y, w> == <y, T w>
    w = torch.randn_like(y)
    lhs = (T.forward(y) * w).sum().item()
    rhs = (y * T.adjoint(w)).sum().item()
    print(f"[T self-adjoint] {lhs:.4e} vs {rhs:.4e}")
    assert abs(lhs - rhs) / abs(rhs) < 1e-6
    # idempotent: T(T y) == T y
    assert (T.forward(T.forward(y)) - T.forward(y)).abs().max().item() == 0.0


if __name__ == "__main__":
    test_shapes_and_autograd_equals_adjoint()
    test_truncation_zeros_out_of_band()
    test_degenerate_reduces_to_radon()
    test_detector_truncation_self_adjoint_idempotent()
    print("\nAll truncated-operator tests passed.")
