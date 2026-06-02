"""End-to-end CT inverse-problem demo using inv_framework.

Builds a Shepp-Logan phantom, simulates a sparse-view parallel-beam sinogram
with optional Gaussian noise, then reconstructs with:
  1) FBP
  2) SIRT
  3) Deep Image Prior (DIP)
  4) SIREN INR
  5) Diffusion Posterior Sampling (if a pretrained checkpoint is given)

Run from the project root:
    python examples/ct_demo.py --image-size 128 --num-angles 30
    python examples/ct_demo.py --dps-checkpoint trained_tiny_unet.pt
"""

import argparse
import math
import os
import sys
import torch

# Make the package importable when run from the project root without install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inv_framework.operators.ct.radon_torch import ParallelBeamRadon2D
from inv_framework.operators.noise import GaussianNoise, NoNoise
from inv_framework.solvers.classical import FBPSolver, SIRTSolver
from inv_framework.solvers.dip import DIPSolver
from inv_framework.solvers.inr import INRSolver
from inv_framework.utils.metrics import psnr, ssim


def shepp_logan(N: int, device='cpu') -> torch.Tensor:
    """Modified Shepp-Logan phantom in [0, 1], shape (N, N)."""
    grid = torch.linspace(-1, 1, N, device=device)
    yy, xx = torch.meshgrid(grid, grid, indexing='ij')
    phantom = torch.zeros(N, N, device=device)
    # (intensity, a, b, cx, cy, angle_deg)
    ellipses = [
        ( 1.00, 0.6900, 0.9200,  0.0000,  0.0000,   0),
        (-0.80, 0.6624, 0.8740,  0.0000, -0.0184,   0),
        (-0.20, 0.1100, 0.3100,  0.2200,  0.0000, -18),
        (-0.20, 0.1600, 0.4100, -0.2200,  0.0000,  18),
        ( 0.10, 0.2100, 0.2500,  0.0000,  0.3500,   0),
        ( 0.10, 0.0460, 0.0460,  0.0000,  0.1000,   0),
        ( 0.10, 0.0460, 0.0460,  0.0000, -0.1000,   0),
        ( 0.10, 0.0460, 0.0230, -0.0800, -0.6050,   0),
        ( 0.10, 0.0230, 0.0230,  0.0000, -0.6060,   0),
        ( 0.10, 0.0230, 0.0460,  0.0600, -0.6050,   0),
    ]
    for inten, a, b, cx, cy, ang in ellipses:
        cosA = math.cos(math.radians(ang))
        sinA = math.sin(math.radians(ang))
        x = xx - cx
        y = yy - cy
        xr = cosA * x + sinA * y
        yr = -sinA * x + cosA * y
        mask = (xr / a) ** 2 + (yr / b) ** 2 <= 1
        phantom = torch.where(mask, phantom + inten, phantom)
    return phantom.clamp(0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--num-angles', type=int, default=30)
    parser.add_argument('--noise-sigma', type=float, default=0.5)
    parser.add_argument('--dip-iters', type=int, default=1000)
    parser.add_argument('--inr-iters', type=int, default=1000)
    parser.add_argument('--dps-checkpoint', type=str, default=None,
                        help='Path to TinyUNet .pt or directory saved by '
                             'diffusers UNet2DModel.save_pretrained(). If '
                             'absent, DPS is skipped.')
    parser.add_argument('--dps-use-diffusers', action='store_true',
                        help='Load DPS checkpoint as a diffusers UNet2DModel '
                             '+ DDIMScheduler instead of TinyUNet + VPSchedule.')
    parser.add_argument('--dps-steps', type=int, default=200)
    parser.add_argument('--dps-scale', type=float, default=1.0)
    parser.add_argument('--out', type=str, default='examples/ct_demo_out.png')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    print(f"Device: {device}")

    print(f"Building Shepp-Logan phantom {args.image_size}x{args.image_size}")
    x_true = shepp_logan(args.image_size, device=device)[None, None]

    operator = ParallelBeamRadon2D(args.image_size, args.num_angles,
                                   device=device)
    noiser = (GaussianNoise(args.noise_sigma)
              if args.noise_sigma > 0 else NoNoise())

    print(f"Simulating sinogram: {args.num_angles} angles, "
          f"noise sigma={args.noise_sigma}")
    with torch.no_grad():
        y_clean = operator.forward(x_true)
        y = noiser(y_clean)

    results = {'Ground truth': x_true}

    print("Solving with FBP...")
    results['FBP'] = FBPSolver().solve(y, operator).clamp(0, 1)

    print("Solving with SIRT...")
    results['SIRT'] = SIRTSolver(
        num_iterations=100, min_value=0.0, max_value=1.0
    ).solve(y, operator).clamp(0, 1)

    print(f"Solving with DIP ({args.dip_iters} iters)...")
    results['DIP'] = DIPSolver(
        num_iterations=args.dip_iters, lr=1e-3,
        log_every=max(args.dip_iters // 5, 1)
    ).solve(y, operator).clamp(0, 1)

    print(f"Solving with INR ({args.inr_iters} iters)...")
    results['INR'] = INRSolver(
        num_iterations=args.inr_iters, lr=1e-4,
        log_every=max(args.inr_iters // 5, 1)
    ).solve(y, operator).clamp(0, 1)

    if args.dps_checkpoint and os.path.exists(args.dps_checkpoint):
        print(f"Solving with DPS using {args.dps_checkpoint}...")
        from inv_framework.solvers.diffusion.dps import DPSSolver

        if args.dps_use_diffusers:
            from diffusers import UNet2DModel, DDIMScheduler
            from inv_framework.solvers.diffusion.diffusers_compat import (
                DiffusersScheduleAdapter, DiffusersUNetWrapper,
            )
            unet = UNet2DModel.from_pretrained(args.dps_checkpoint).to(device)
            model = DiffusersUNetWrapper(unet)
            schedule = DiffusersScheduleAdapter(
                DDIMScheduler(num_train_timesteps=1000))
        else:
            from inv_framework.models.tiny_unet import TinyUNet
            from inv_framework.solvers.diffusion.scheduler import VPSchedule
            model = TinyUNet(channels=1).to(device)
            model.load_state_dict(
                torch.load(args.dps_checkpoint, map_location=device))
            schedule = VPSchedule(device=device)

        results['DPS'] = DPSSolver(
            model=model,
            schedule=schedule,
            num_inference_steps=args.dps_steps,
            scale=args.dps_scale,
            log_every=max(args.dps_steps // 5, 1)
        ).solve(y, operator).clamp(0, 1)
    else:
        print("No DPS checkpoint provided -- skipping DPS. "
              "Train one with examples/train_diffusion_for_dps.py")

    print("\nResults vs ground truth:")
    print(f"  {'Method':18s} {'PSNR (dB)':>10s} {'SSIM':>8s}")
    for name, rec in results.items():
        if name == 'Ground truth':
            continue
        p = psnr(rec, x_true, data_range=1.0).item()
        s = ssim(rec, x_true, data_range=1.0).item()
        print(f"  {name:18s} {p:10.2f} {s:8.4f}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        n = len(results)
        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.4))
        if n == 1:
            axes = [axes]
        for ax, (name, img) in zip(axes, results.items()):
            ax.imshow(img[0, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
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
