"""Train a noise predictor on random ellipse phantoms.

Produces a small unconditional prior over Shepp-Logan-like images, suitable
as the eps_theta model for DPSSolver in ct_demo.py.

Backbone choice:
    default              : bundled TinyUNet (no extra deps)
    --use-diffusers      : HuggingFace diffusers.UNet2DModel + DDPMScheduler

Run:
    python examples/train_diffusion_for_dps.py --steps 5000 \
        --out trained_tiny_unet.pt
    python examples/train_diffusion_for_dps.py --use-diffusers --steps 5000 \
        --out trained_diffusers_unet
"""

import argparse
import math
import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inv_framework.models.tiny_unet import TinyUNet
from inv_framework.solvers.diffusion.scheduler import VPSchedule


def random_ellipse_phantom(N: int, num_ellipses: int = 6,
                           device='cpu') -> torch.Tensor:
    grid = torch.linspace(-1, 1, N, device=device)
    yy, xx = torch.meshgrid(grid, grid, indexing='ij')
    img = torch.zeros(N, N, device=device)
    for _ in range(num_ellipses):
        ab = torch.rand(2, device=device) * 0.4 + 0.05
        a, b = ab[0], ab[1]
        cxcy = torch.rand(2, device=device) * 1.2 - 0.6
        cx, cy = cxcy[0], cxcy[1]
        ang = torch.rand((), device=device) * math.pi
        inten = torch.rand((), device=device) * 0.8 + 0.2
        cosA, sinA = torch.cos(ang), torch.sin(ang)
        x = xx - cx
        y = yy - cy
        xr = cosA * x + sinA * y
        yr = -sinA * x + cosA * y
        mask = (xr / a) ** 2 + (yr / b) ** 2 <= 1
        img = torch.where(mask, img + inten, img)
    return img.clamp(0, 1)


def sample_batch(batch_size: int, N: int, device) -> torch.Tensor:
    return torch.stack([random_ellipse_phantom(N, device=device)
                        for _ in range(batch_size)])[:, None]


def build_tiny(args, device):
    model = TinyUNet(channels=1).to(device)
    sched = VPSchedule(device=device)
    return model, sched, lambda m, x, t: m(x, t)


def build_diffusers(args, device):
    from diffusers import UNet2DModel, DDPMScheduler
    from inv_framework.solvers.diffusion.diffusers_compat import (
        DiffusersScheduleAdapter,
    )
    model = UNet2DModel(
        sample_size=args.image_size,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 128),
        down_block_types=('DownBlock2D', 'DownBlock2D', 'AttnDownBlock2D'),
        up_block_types=('AttnUpBlock2D', 'UpBlock2D', 'UpBlock2D'),
    ).to(device)
    raw_sched = DDPMScheduler(num_train_timesteps=1000)
    sched = DiffusersScheduleAdapter(raw_sched)
    return model, sched, lambda m, x, t: m(x, t).sample


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image-size', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--steps', type=int, default=5000)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--out', type=str, default='trained_tiny_unet.pt',
                   help='State-dict file for TinyUNet, or directory for '
                        'diffusers UNet2DModel.save_pretrained().')
    p.add_argument('--use-diffusers', action='store_true',
                   help='Use diffusers.UNet2DModel + DDPMScheduler instead '
                        'of the bundled TinyUNet + VPSchedule.')
    p.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    device = args.device
    print(f"Device: {device} | backbone: "
          f"{'diffusers UNet2DModel' if args.use_diffusers else 'TinyUNet'}")

    builder = build_diffusers if args.use_diffusers else build_tiny
    model, sched, call_model = builder(args, device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"model: {num_params/1e6:.2f} M params")

    for step in range(args.steps):
        x0 = sample_batch(args.batch_size, args.image_size, device)
        t = torch.randint(0, sched.num_train_timesteps, (args.batch_size,),
                          device=device)
        noise = torch.randn_like(x0)
        x_t = sched.add_noise(x0, noise, t)
        eps_pred = call_model(model, x_t, t)
        loss = F.mse_loss(eps_pred, noise)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if step % 100 == 0:
            print(f"step {step}/{args.steps}  loss={loss.item():.4f}")

    if args.use_diffusers:
        model.save_pretrained(args.out)
        print(f"Saved diffusers UNet to {args.out}/")
    else:
        torch.save(model.state_dict(), args.out)
        print(f"Saved TinyUNet checkpoint to {args.out}")


if __name__ == '__main__':
    main()
