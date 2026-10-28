"""TomoPy-compatible CUDA CT as a framework ``LinearOperator``.

The numerical kernels are preserved from nasn-truncated-ct. Both A and A^T
have autograd rules, so the same operator works with DDS, DIP and LoRA/CG.
CUDA compilation is deferred until an operator is evaluated.
"""

import math

import torch

from ..base import LinearOperator


class _Project(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, theta, center, mask):
        from .radon_cuda import project
        ctx.save_for_backward(theta, mask)
        ctx.center = center
        ctx.input_dtype = image.dtype
        result = project(image[:, 0].float().contiguous(), theta, center)
        return (result * mask).unsqueeze(1).to(image.dtype)

    @staticmethod
    def backward(ctx, gradient):
        theta, mask = ctx.saved_tensors
        return _Adjoint.apply(gradient, theta, ctx.center, mask).to(ctx.input_dtype), None, None, None


class _Adjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sinogram, theta, center, mask):
        from .radon_cuda import bp
        ctx.save_for_backward(theta, mask)
        ctx.center = center
        ctx.input_dtype = sinogram.dtype
        # Multiply out-of-place: autograd gradients and caller data must not
        # be modified by the native kernels.
        measured = (sinogram[:, 0].float() * mask).contiguous()
        result = bp(measured, theta, center, mask.numel(), mask.numel())
        return result.unsqueeze(1).to(sinogram.dtype)

    @staticmethod
    def backward(ctx, gradient):
        theta, mask = ctx.saved_tensors
        return _Project.apply(gradient, theta, ctx.center, mask).to(ctx.input_dtype), None, None, None


class TomoPyCUDARadon2D(LinearOperator):
    """Parallel-beam ``A = T R`` with the NaSn CUDA projector and exact BP.

    ``inner=None`` measures the full detector. ``center`` is expressed in
    detector pixels, including any padding. Only one-channel square images
    are supported by this backend. Gridrec is exposed as ``pseudo_inverse``;
    the explicit ``adjoint`` always uses unfiltered back-projection.
    """

    def __init__(self, image_size=2048, num_angles=900, inner=None,
                 center=None, angles=None, device="cuda"):
        self.image_size = int(image_size)
        self.inner = self.image_size if inner is None else int(inner)
        if not 0 < self.inner <= self.image_size:
            raise ValueError("expected 0 < inner <= image_size")
        self.center = self.image_size / 2 if center is None else float(center)
        if not math.isfinite(self.center):
            raise ValueError("center must be finite")
        if angles is None:
            # Keep the published NumPy float32 angle grid byte-for-byte.
            import numpy as np
            if int(num_angles) <= 0:
                raise ValueError("num_angles must be positive")
            angles = np.linspace(0, np.pi, int(num_angles), endpoint=False, dtype=np.float32)
        self.angles = torch.as_tensor(angles, dtype=torch.float32).detach().clone()
        if self.angles.ndim != 1 or not self.angles.numel() or not torch.isfinite(self.angles).all():
            raise ValueError("angles must be a nonempty finite vector in radians")
        self.num_angles = self.angles.numel()
        self.domain_shape = (1, self.image_size, self.image_size)
        self.range_shape = (1, self.num_angles, self.image_size)
        self._mask = torch.zeros(self.image_size)
        left = (self.image_size - self.inner) // 2
        self._mask[left:left + self.inner] = 1
        self.to(device)

    def to(self, device):
        self.device = torch.device(device)
        self.angles = self.angles.to(self.device).contiguous()
        self._mask = self._mask.to(self.device)
        return self

    def _check(self, value, shape):
        if value.ndim != 4 or tuple(value.shape[1:]) != shape:
            raise ValueError(f"expected (B, {shape}), got {tuple(value.shape)}")
        if not value.is_cuda or value.device != self.angles.device:
            raise RuntimeError("input and operator must be on the same CUDA device; use operator.to(device)")
        if not value.is_floating_point():
            raise TypeError("CT inputs must be floating-point tensors")

    def forward(self, image):
        self._check(image, self.domain_shape)
        return _Project.apply(image, self.angles, self.center, self._mask)

    def adjoint(self, sinogram):
        self._check(sinogram, self.range_shape)
        return _Adjoint.apply(sinogram, self.angles, self.center, self._mask)

    def pseudo_inverse(self, sinogram, method="gridrec", **kwargs):
        self._check(sinogram, self.range_shape)
        if method not in {"gridrec", "fbp"}:
            raise ValueError("this backend exposes method='gridrec' (alias 'fbp')")
        from .radon_cuda import TomoPyGridrecCUDA
        inverse = TomoPyGridrecCUDA(
            self.image_size, self.angles.cpu(), self.center, **kwargs,
        ).to(self.angles.device)
        with torch.no_grad():
            measured = sinogram[:, 0].float() * self._mask
            return inverse(measured).unsqueeze(1).to(sinogram.dtype)
