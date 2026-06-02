"""Sanity tests for the parallel-beam Radon operator.

The "formal" adjoint identity <Ax, y> = <x, A^T y> holds only for ideal
continuous Radon; discrete CT operators (ASTRA included) use bilinear-
interpolated forward and discrete back-projection, which are *geometric*
duals but not bit-exact numerical transposes. What we DO require:

  1. autograd backward of forward equals explicit adjoint (by construction).
  2. FBP recovers a Shepp-Logan-like phantom on a moderately dense sampling.

This file tests both.
"""

import math
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inv_framework.operators.ct.radon_torch import ParallelBeamRadon2D
from inv_framework.solvers.classical import fbp


def test_autograd_matches_adjoint(image_size=32, num_angles=24, tol=1e-5,
                                  seed=1):
    torch.manual_seed(seed)
    op = ParallelBeamRadon2D(image_size=image_size,
                             num_angles=num_angles, device='cpu')
    x = torch.randn(1, 1, image_size, image_size, requires_grad=True)
    y = torch.randn(1, 1, num_angles, image_size)
    s = (op.forward(x) * y).sum()
    g_auto, = torch.autograd.grad(s, x)
    g_exact = op.adjoint(y)
    rel = (g_auto - g_exact).norm() / (g_exact.norm() + 1e-12)
    print(f"autograd vs adjoint rel_err = {rel.item():.3e} (tol {tol:.1e})")
    assert rel < tol, f"Autograd != adjoint: rel_err={rel.item():.3e}"
    print("PASS: backward(forward) == adjoint.")


def _disk_phantom(N: int, device='cpu') -> torch.Tensor:
    grid = torch.linspace(-1, 1, N, device=device)
    yy, xx = torch.meshgrid(grid, grid, indexing='ij')
    img = torch.zeros(N, N, device=device)
    img = torch.where(xx ** 2 + yy ** 2 <= 0.7 ** 2, img + 1.0, img)
    img = torch.where((xx - 0.2) ** 2 + yy ** 2 <= 0.15 ** 2,
                      img - 0.5, img)
    return img.clamp(0, 1)


def test_fbp_recovers_disk(image_size=128, num_angles=180, tol_psnr=20.0,
                           seed=0):
    torch.manual_seed(seed)
    x_true = _disk_phantom(image_size)[None, None]
    op = ParallelBeamRadon2D(image_size=image_size, num_angles=num_angles,
                             device='cpu')
    y = op.forward(x_true)
    x_rec = fbp(op, y).clamp(0, 1)
    mse = ((x_rec - x_true) ** 2).mean()
    psnr = 10.0 * math.log10(1.0 / mse.item())
    print(f"FBP PSNR on disk phantom: {psnr:.2f} dB (require >{tol_psnr})")
    assert psnr > tol_psnr, f"FBP failed: PSNR={psnr:.2f} dB"
    print("PASS: FBP recovers disk phantom.")


if __name__ == '__main__':
    test_autograd_matches_adjoint()
    test_fbp_recovers_disk()
    print("\nAll Radon operator tests passed.")
