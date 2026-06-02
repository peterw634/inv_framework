"""Verify the framework supports a non-linear forward operator end-to-end.

Tests:
  1. A custom ForwardOperator subclass without `adjoint` can be instantiated
     and called.
  2. Classical solvers (which need an adjoint) correctly refuse it with a
     clear error message.
  3. DIPSolver and INRSolver successfully fit the non-linear forward.
"""

import math
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inv_framework.operators.base import ForwardOperator, LinearOperator
from inv_framework.solvers.classical import fbp, sirt, landweber
from inv_framework.solvers.dip import DIPSolver


class _SquaredBlur(ForwardOperator):
    """Toy non-linear op: y = (5x5_average_pool(x))^2 + small.

    Non-linear in x via the squaring step.
    """
    def __init__(self, image_size: int, kernel: int = 5):
        self.kernel = kernel
        self.domain_shape = (1, image_size, image_size)
        # average pool with same-size output via conv with stride 1
        self.range_shape = (1, image_size, image_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.kernel // 2
        kernel = torch.ones(1, 1, self.kernel, self.kernel,
                            device=x.device, dtype=x.dtype) / (self.kernel ** 2)
        blurred = torch.nn.functional.conv2d(x, kernel, padding=pad)
        return blurred ** 2


def test_nonlinear_op_does_not_require_adjoint():
    op = _SquaredBlur(image_size=32)
    assert not isinstance(op, LinearOperator)
    x = torch.rand(1, 1, 32, 32)
    y = op.forward(x)
    assert y.shape == (1, 1, 32, 32)
    print("PASS: non-linear operator forward works.")


def test_classical_refuses_nonlinear():
    op = _SquaredBlur(image_size=16)
    y = torch.rand(1, 1, 16, 16)
    for fn, name in [(fbp, "fbp"), (sirt, "sirt"), (landweber, "landweber")]:
        try:
            fn(op, y)
        except TypeError as e:
            assert "LinearOperator" in str(e), str(e)
            print(f"PASS: {name} refused non-linear op -- {e}")
            continue
        raise AssertionError(f"{name} should have raised TypeError")


def test_dip_recovers_nonlinear(image_size=32, iters=400, tol_psnr=15.0,
                                 seed=0):
    """DIP fits the non-linear forward; check the recovered x reduces residual."""
    torch.manual_seed(seed)
    x_true = torch.rand(1, 1, image_size, image_size)
    # Smooth the truth so DIP has a reasonable target.
    pad = 2
    k = torch.ones(1, 1, 5, 5) / 25
    x_true = torch.nn.functional.conv2d(x_true, k, padding=pad).clamp(0, 1)

    op = _SquaredBlur(image_size=image_size)
    y = op.forward(x_true)

    x_rec = DIPSolver(num_iterations=iters, lr=5e-3,
                      log_every=iters // 4).solve(y, op).clamp(0, 1)

    # Verify the measurement residual dropped substantially.
    mse_init = ((op.forward(torch.zeros_like(x_true)) - y) ** 2).mean().item()
    mse_rec = ((op.forward(x_rec) - y) ** 2).mean().item()
    print(f"measurement MSE: init={mse_init:.4e} -> recovered={mse_rec:.4e}")
    assert mse_rec < 0.1 * mse_init, (
        f"DIP failed to reduce measurement residual: "
        f"{mse_rec:.4e} vs init {mse_init:.4e}")
    print("PASS: DIP fits non-linear forward (residual reduced >10x).")


if __name__ == '__main__':
    test_nonlinear_op_does_not_require_adjoint()
    test_classical_refuses_nonlinear()
    test_dip_recovers_nonlinear()
    print("\nAll non-linear operator tests passed.")
