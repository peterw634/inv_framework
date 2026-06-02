"""PSNR and a simple SSIM implementation."""

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor,
         data_range: float = 1.0) -> torch.Tensor:
    mse = ((pred - target) ** 2).mean(dim=(-1, -2, -3))
    return 10.0 * torch.log10(data_range ** 2 / mse.clamp(min=1e-12))


def ssim(pred: torch.Tensor, target: torch.Tensor,
         data_range: float = 1.0, window_size: int = 11) -> torch.Tensor:
    pred = pred.clamp(0, data_range)
    target = target.clamp(0, data_range)
    sigma = 1.5
    coords = (torch.arange(window_size, dtype=pred.dtype, device=pred.device)
              - window_size // 2)
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())
    window_2d = g[:, None] * g[None, :]
    C = pred.shape[-3]
    window = window_2d.expand(C, 1, window_size, window_size).contiguous()
    pad = window_size // 2
    mu_x = F.conv2d(pred, window, padding=pad, groups=C)
    mu_y = F.conv2d(target, window, padding=pad, groups=C)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x = F.conv2d(pred * pred, window, padding=pad, groups=C) - mu_x2
    sigma_y = F.conv2d(target * target, window, padding=pad, groups=C) - mu_y2
    sigma_xy = F.conv2d(pred * target, window, padding=pad, groups=C) - mu_xy
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    ssim_map = (((2 * mu_xy + C1) * (2 * sigma_xy + C2))
                / ((mu_x2 + mu_y2 + C1) * (sigma_x + sigma_y + C2)))
    return ssim_map.mean(dim=(-1, -2, -3))
