"""Pure-torch 2D parallel-beam Radon transform.

Uses `grid_sample` to rotate the image, then sums along one axis. Wrapped in
a custom autograd Function so that the backward pass equals the explicit
back-projection (exact adjoint), regardless of grid_sample's internal grad.
"""

import math
import torch
import torch.nn.functional as F

from ..base import LinearOperator


def _rotation_grid(theta: torch.Tensor, shape):
    """Build a grid_sample grid that rotates an image of `shape` by theta rad.

    `shape` is (B, C, H, W). theta is a scalar tensor on the same device.
    """
    B = shape[0]
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    zero = torch.zeros_like(cos_t)
    rot = torch.stack([
        torch.stack([cos_t, -sin_t, zero]),
        torch.stack([sin_t,  cos_t, zero]),
    ])
    rot = rot.to(dtype=torch.float32).unsqueeze(0).expand(B, -1, -1)
    return F.affine_grid(rot, list(shape), align_corners=False)


def _radon_forward_impl(image: torch.Tensor,
                        angles: torch.Tensor) -> torch.Tensor:
    """Radon transform without autograd wiring.

    image: (B, C, H, W) with H == W.
    angles: (n_angles,) radians.
    returns: (B, C, n_angles, W).
    """
    H, W = image.shape[-2], image.shape[-1]
    assert H == W, "ParallelBeamRadon2D currently requires square images."
    sinos = []
    for theta in angles:
        grid = _rotation_grid(theta, image.shape)
        rotated = F.grid_sample(image, grid, align_corners=False,
                                padding_mode='zeros')
        sinos.append(rotated.sum(dim=-2))  # sum along H
    return torch.stack(sinos, dim=-2)  # (B, C, n_angles, W)


def _radon_adjoint_impl(sinogram: torch.Tensor,
                        angles: torch.Tensor,
                        image_size: int) -> torch.Tensor:
    """Unfiltered back-projection (adjoint of Radon).

    sinogram: (B, C, n_angles, W)
    returns: (B, C, image_size, image_size)
    """
    B, C, n_angles, W = sinogram.shape
    H = image_size
    out = torch.zeros(B, C, H, W, device=sinogram.device, dtype=sinogram.dtype)
    target_shape = (B, C, H, W)
    for i in range(n_angles):
        theta = angles[i]
        proj = sinogram[:, :, i:i + 1, :].expand(B, C, H, W)
        grid = _rotation_grid(-theta, target_shape)
        smeared = F.grid_sample(proj, grid, align_corners=False,
                                padding_mode='zeros')
        out = out + smeared
    return out


class _RadonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, angles, image_size):
        ctx.save_for_backward(angles)
        ctx.image_size = image_size
        return _radon_forward_impl(image, angles)

    @staticmethod
    def backward(ctx, grad_output):
        (angles,) = ctx.saved_tensors
        grad_image = _radon_adjoint_impl(grad_output, angles, ctx.image_size)
        return grad_image, None, None


class _RadonAdjointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sinogram, angles, image_size):
        ctx.save_for_backward(angles)
        ctx.image_size = image_size
        return _radon_adjoint_impl(sinogram, angles, image_size)

    @staticmethod
    def backward(ctx, grad_output):
        (angles,) = ctx.saved_tensors
        grad_sino = _radon_forward_impl(grad_output, angles)
        return grad_sino, None, None


class ParallelBeamRadon2D(LinearOperator):
    """2D parallel-beam Radon transform on square images, pure torch.

    Parameters
    ----------
    image_size : int
        H == W of the image.
    num_angles : int
        Number of equally spaced angles in [0, pi). Ignored if `angles` is set.
    angles : torch.Tensor, optional
        Explicit projection angles (radians).
    device : str or torch.device
    """

    def __init__(self,
                 image_size: int,
                 num_angles: int = 60,
                 angles: torch.Tensor = None,
                 device: str = 'cpu',
                 in_channels: int = 1):
        if angles is None:
            angles = torch.linspace(0.0, math.pi, num_angles + 1,
                                    device=device)[:-1]
        self.angles = angles.to(device=device, dtype=torch.float32)
        self.image_size = int(image_size)
        self.num_angles = int(self.angles.numel())
        self.device = device
        self.domain_shape = (in_channels, image_size, image_size)
        self.range_shape = (in_channels, self.num_angles, image_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _RadonFunction.apply(x, self.angles, self.image_size)

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        return _RadonAdjointFunction.apply(y, self.angles, self.image_size)

    def pseudo_inverse(self, y: torch.Tensor, method: str = 'fbp',
                       **kwargs) -> torch.Tensor:
        from inv_framework.solvers.classical import fbp, sirt, landweber
        if method == 'fbp':
            return fbp(self, y)
        if method == 'sirt':
            return sirt(self, y, **kwargs)
        if method == 'landweber':
            return landweber(self, y, **kwargs)
        raise ValueError(f"Unknown pseudo_inverse method: {method!r}")
