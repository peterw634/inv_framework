"""Non-linear inverse-problem demo: saturated-detector CT.

Models a CT setup where the detector saturates smoothly:

    y = tanh(alpha * A_radon(x)) / tanh(alpha * mu),

so high-attenuation rays clip toward 1. This is non-linear in x, which means
FBP/SIRT do not apply -- but DIP and INR still work because they only need
the autograd-traceable `forward`.

Showcases how to plug a custom non-linear operator into the framework: just
subclass ForwardOperator and implement `forward`. Solvers do not change.

Run:
    python examples/nonlinear_demo.py --alpha 2.0 --num-angles 60
"""

import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inv_framework.operators.base import ForwardOperator
from inv_framework.operators.ct.radon_torch import ParallelBeamRadon2D
from inv_framework.operators.noise import GaussianNoise, NoNoise
from inv_framework.solvers.dip import DIPSolver
from inv_framework.solvers.inr import INRSolver
from inv_framework.utils.metrics import psnr, ssim

from examples.ct_demo import shepp_logan


class SaturatedRadon2D(ForwardOperator):
    """Non-linear forward: y = tanh(alpha * Radon(x) / image_size).

    Composes a linear Radon transform with a smooth saturation. The 1/N
    normalisation keeps the tanh argument in O(1) regardless of image size,
    so `alpha` controls saturation strength rather than trivially clipping.

    Larger alpha -> sharper, more non-linear detector response.
    Smaller alpha -> response approaches linear in Radon(x).
    """

    def __init__(self, image_size: int, num_angles: int,
                 alpha: float = 2.0, device: str = 'cpu'):
        self.linear = ParallelBeamRadon2D(image_size, num_angles, device=device)
        self.alpha = float(alpha)
        self.image_size = int(image_size)
        self.domain_shape = self.linear.domain_shape
        self.range_shape = self.linear.range_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sino = self.linear.forward(x) / self.image_size
        return torch.tanh(self.alpha * sino)

    def invert_saturation(self, y: torch.Tensor) -> torch.Tensor:
        """Pointwise inverse of the saturation -- helpful for sanity checks.

        Returns the underlying linear sinogram given a clean (noiseless) y.
        Not used by solvers; only for visualisation.
        """
        y_clip = y.clamp(-0.999, 0.999)
        return torch.atanh(y_clip) / self.alpha * self.image_size


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image-size', type=int, default=128)
    p.add_argument('--num-angles', type=int, default=60)
    p.add_argument('--alpha', type=float, default=2.0,
                   help='Saturation sharpness; small alpha approaches linear.')
    p.add_argument('--noise-sigma', type=float, default=0.02)
    p.add_argument('--dip-iters', type=int, default=1500)
    p.add_argument('--inr-iters', type=int, default=1500)
    p.add_argument('--out', type=str,
                   default='examples/nonlinear_demo_out.png')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    print(f"Device: {device}")

    x_true = shepp_logan(args.image_size, device=device)[None, None]
    op = SaturatedRadon2D(args.image_size, args.num_angles,
                          alpha=args.alpha, device=device)
    noiser = (GaussianNoise(args.noise_sigma)
              if args.noise_sigma > 0 else NoNoise())

    print(f"Forward op: SaturatedRadon2D (alpha={args.alpha})")
    with torch.no_grad():
        y_clean = op.forward(x_true)
        y = noiser(y_clean)

    # Show that classical solvers correctly refuse a non-linear operator.
    print("\nTrying FBP on the non-linear operator (should raise)...")
    try:
        from inv_framework.solvers.classical import fbp
        fbp(op, y)
    except TypeError as e:
        print(f"  caught TypeError: {e}")

    results = {'Ground truth': x_true, 'Measurement (mid-slice)': None}

    print(f"\nSolving with DIP ({args.dip_iters} iters)...")
    results['DIP'] = DIPSolver(
        num_iterations=args.dip_iters, lr=1e-3,
        log_every=max(args.dip_iters // 5, 1)
    ).solve(y, op).clamp(0, 1)

    print(f"Solving with INR ({args.inr_iters} iters)...")
    results['INR'] = INRSolver(
        num_iterations=args.inr_iters, lr=1e-4,
        log_every=max(args.inr_iters // 5, 1)
    ).solve(y, op).clamp(0, 1)

    print("\nResults vs ground truth:")
    print(f"  {'Method':18s} {'PSNR (dB)':>10s} {'SSIM':>8s}")
    for name, rec in results.items():
        if name in ('Ground truth', 'Measurement (mid-slice)'):
            continue
        pp = psnr(rec, x_true, data_range=1.0).item()
        ss = ssim(rec, x_true, data_range=1.0).item()
        print(f"  {name:18s} {pp:10.2f} {ss:8.4f}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
        axes[0].imshow(x_true[0, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('Ground truth')
        axes[1].imshow(y[0, 0].cpu().numpy(), cmap='gray', aspect='auto')
        axes[1].set_title('Sinogram (saturated)')
        axes[2].imshow(results['DIP'][0, 0].cpu().numpy(),
                       cmap='gray', vmin=0, vmax=1)
        axes[2].set_title('DIP')
        axes[3].imshow(results['INR'][0, 0].cpu().numpy(),
                       cmap='gray', vmin=0, vmax=1)
        axes[3].set_title('INR')
        for ax in axes:
            ax.axis('off')
        fig.tight_layout()
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        fig.savefig(args.out, dpi=120, bbox_inches='tight')
        print(f"Saved comparison figure to {args.out}")
    except ImportError:
        print("matplotlib not installed; skipping figure.")


if __name__ == '__main__':
    main()
