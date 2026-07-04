"""End-to-end interior-tomography (truncated CT) demo.

Simulates a *truncated* parallel-beam sinogram — the detector sees only a
central band of width ``--inner``, so the object extends beyond the field of
view (interior tomography) — then reconstructs with:

  1) Truncated FBP           (shows the classic interior-tomography cupping)
  2) SIRT                     (iterative, still data-limited outside the band)
  3) DDS + CG data consistency (mean-flow prior + per-step CG correction)

The DDS prior here is an *untrained* TinyVelocityPrior, so this demo is about
the pipeline mechanics (truncated operator + CG data consistency + mean-flow
sampler), not reconstruction quality — swap in a real pMF prior via
`PMFVelocityPrior` for actual industrial-CT reconstructions.

Run from the project root:
    python examples/truncated_ct_demo.py --image-size 128 --num-angles 90 --inner 48
"""

import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inv_framework.operators.ct.truncated import TruncatedRadon2D
from inv_framework.operators.noise import GaussianNoise, NoNoise
from inv_framework.solvers.classical import FBPSolver, SIRTSolver
from inv_framework.solvers.diffusion.velocity_prior import TinyVelocityPrior
from inv_framework.solvers.diffusion.dds import DDSSolver
from inv_framework.utils.metrics import psnr, ssim


def shepp_logan(N: int, device='cpu') -> torch.Tensor:
    """Modified Shepp-Logan phantom in [0, 1], shape (N, N)."""
    grid = torch.linspace(-1, 1, N, device=device)
    yy, xx = torch.meshgrid(grid, grid, indexing='ij')
    phantom = torch.zeros(N, N, device=device)
    ellipses = [
        (1.00, 0.6900, 0.9200, 0.0, 0.0, 0),
        (-0.80, 0.6624, 0.8740, 0.0, -0.0184, 0),
        (-0.20, 0.1100, 0.3100, 0.22, 0.0, -18),
        (-0.20, 0.1600, 0.4100, -0.22, 0.0, 18),
        (0.10, 0.2100, 0.2500, 0.0, 0.35, 0),
        (0.10, 0.0460, 0.0460, 0.0, 0.10, 0),
        (0.10, 0.0460, 0.0460, 0.0, -0.10, 0),
    ]
    for inten, a, b, cx, cy, ang in ellipses:
        cosA, sinA = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        x, y = xx - cx, yy - cy
        xr = cosA * x + sinA * y
        yr = -sinA * x + cosA * y
        mask = (xr / a) ** 2 + (yr / b) ** 2 <= 1
        phantom = torch.where(mask, phantom + inten, phantom)
    return phantom.clamp(0, 1)


def _roi_metrics(rec, x_true, inner):
    """PSNR/SSIM over the central data band (radius ~ inner/2)."""
    N = x_true.shape[-1]
    yy, xx = torch.meshgrid(torch.arange(N), torch.arange(N), indexing='ij')
    rr = ((yy - N / 2) ** 2 + (xx - N / 2) ** 2).sqrt().to(x_true.device)
    m = (rr < inner / 2).float()[None, None]
    p = psnr(rec * m, x_true * m, data_range=1.0).item()
    s = ssim(rec * m, x_true * m, data_range=1.0).item()
    return p, s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--num-angles', type=int, default=90)
    parser.add_argument('--inner', type=int, default=48,
                        help='Measured central detector-band width (interior '
                             'tomography). < image-size truncates the FOV.')
    parser.add_argument('--noise-sigma', type=float, default=0.0)
    parser.add_argument('--dds-steps', type=int, default=16)
    parser.add_argument('--dds-lam', type=float, default=0.05)
    parser.add_argument('--dds-iter', type=int, default=5)
    parser.add_argument('--out', type=str,
                        default='examples/truncated_ct_demo_out.png')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    N, inner = args.image_size, args.inner
    print(f"Device: {device}")
    print(f"Interior tomography: {N}x{N} frame, measured band = {inner} "
          f"({100*inner/N:.0f}% of FOV), {args.num_angles} angles")

    x_true = shepp_logan(N, device=device)[None, None]
    A = TruncatedRadon2D(N, args.num_angles, inner=inner, device=device)
    noiser = GaussianNoise(args.noise_sigma) if args.noise_sigma > 0 else NoNoise()

    with torch.no_grad():
        y = noiser(A.forward(x_true))

    results = {'Ground truth': x_true}

    print("Truncated FBP...")
    results['Trunc-FBP'] = FBPSolver().solve(y, A).clamp(0, 1)

    print("SIRT...")
    results['SIRT'] = SIRTSolver(num_iterations=100, min_value=0.0,
                                 max_value=1.0).solve(y, A).clamp(0, 1)

    print(f"DDS + CG ({args.dds_steps} steps, untrained TinyVelocityPrior)...")
    prior = TinyVelocityPrior(img_channels=1, img_size=N, base=32).to(device)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    results['DDS+CG'] = DDSSolver(
        prior, num_steps=args.dds_steps, dc_lam=args.dds_lam,
        dc_iter=args.dds_iter, log_every=max(args.dds_steps // 4, 1),
    ).solve(y, A, generator=gen).clamp(0, 1)

    print("\nCentral-band (ROI) reconstruction quality:")
    print(f"  {'Method':14s} {'PSNR (dB)':>10s} {'SSIM':>8s}")
    for name, rec in results.items():
        if name == 'Ground truth':
            continue
        p, s = _roi_metrics(rec, x_true, inner)
        print(f"  {name:14s} {p:10.2f} {s:8.4f}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        n = len(results)
        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.4))
        for ax, (name, img) in zip(axes, results.items()):
            ax.imshow(img[0, 0].detach().cpu().numpy(), cmap='gray',
                      vmin=0, vmax=1)
            ax.set_title(name)
            ax.axis('off')
        fig.tight_layout()
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        fig.savefig(args.out, dpi=120, bbox_inches='tight')
        print(f"Saved comparison figure to {args.out}")
    except ImportError:
        print("matplotlib not installed; skipping figure.")


if __name__ == '__main__':
    main()
