"""Truncated (interior-tomography) parallel-beam Radon.

Industrial and lab micro-CT frequently image an object *larger* than the
detector: at every angle only a central band of the sinogram is measured, and
the region outside that band must be filled in by a prior. This is *interior
tomography* / *region-of-interest CT*, and its forward model is

    A = T . R

where ``R`` is the ordinary parallel-beam Radon transform and ``T`` is a
detector-truncation mask that zeros every detector column outside a central
``inner``-wide band.

We do **not** re-implement projection here. ``T`` is a diagonal (hence
self-adjoint) mask on the last (detector) axis, and this framework already
ships an autograd-exact Radon (`ParallelBeamRadon2D`, whose backward equals its
adjoint). Composing the two gives, for free,

    forward(x)  = T(R x)
    adjoint(y)  = R^T(T y)
    backward(forward)  ==  adjoint            (T self-adjoint, R exact)

so no custom ``torch.autograd.Function`` is needed — the composition inherits
the exact-adjoint property. Pure torch, no tomopy / CUDA dependency.

The geometry mirrors ``indus_CT/NaSn/ct_operators.py`` (``_detector_mask_np``):
the measured band is centred with ``lo = (W - inner) // 2``. With
``inner == image_size`` the mask is all-ones and ``TruncatedRadon2D`` reduces
exactly to ``ParallelBeamRadon2D``.
"""

import torch

from ..base import LinearOperator
from .radon_torch import ParallelBeamRadon2D


def _detector_mask(width: int, inner: int, device, dtype=torch.float32):
    """1-D detector mask: 1.0 on the central ``inner`` columns, else 0.0.

    Centring matches ``_detector_mask_np`` in indus_CT: ``lo = (W-inner)//2``.
    """
    inner = min(int(inner), int(width))
    m = torch.zeros(width, device=device, dtype=dtype)
    lo = (width - inner) // 2
    m[lo:lo + inner] = 1.0
    return m


class DetectorTruncation(LinearOperator):
    """Self-adjoint detector-truncation mask ``T`` on the last (detector) axis.

    Domain and range are the same sinogram shape ``(C, n_angles, W)``; the
    operator zeros every detector column outside the central ``inner`` band.
    Because ``T`` is diagonal it is its own adjoint, and its autograd backward
    (a plain elementwise multiply) already equals that adjoint.
    """

    def __init__(self,
                 num_angles: int,
                 width: int,
                 inner: int,
                 device: str = 'cpu',
                 in_channels: int = 1):
        self.num_angles = int(num_angles)
        self.width = int(width)
        self.inner = int(inner)
        self.device = device
        self.domain_shape = (in_channels, self.num_angles, self.width)
        self.range_shape = (in_channels, self.num_angles, self.width)
        self._mask = _detector_mask(self.width, self.inner, device)

    def _apply_mask(self, y: torch.Tensor) -> torch.Tensor:
        return y * self._mask.to(device=y.device, dtype=y.dtype)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self._apply_mask(y)

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:  # T is self-adjoint
        return self._apply_mask(y)


class TruncatedRadon2D(LinearOperator):
    """Interior-tomography forward model ``A = T . R`` (pure torch).

    Parameters
    ----------
    image_size : int
        H == W of the reconstruction frame.
    num_angles : int
        Number of equally spaced angles in [0, pi). Ignored if ``angles`` set.
    inner : int
        Width (in detector columns) of the measured central band. ``inner ==
        image_size`` disables truncation and reduces this to a plain Radon.
    angles : torch.Tensor, optional
        Explicit projection angles (radians).
    device : str or torch.device
    in_channels : int

    Notes
    -----
    ``forward`` is autograd-traceable and its backward equals ``adjoint`` (the
    inner `ParallelBeamRadon2D` is exact and the truncation mask is
    self-adjoint), so gradient-based solvers (DIP/INR/DPS/DDS) and
    explicit-adjoint solvers (SIRT/Landweber/CG) agree on ``A^T``.
    """

    def __init__(self,
                 image_size: int,
                 num_angles: int = 60,
                 inner: int = None,
                 angles: torch.Tensor = None,
                 device: str = 'cpu',
                 in_channels: int = 1):
        self.radon = ParallelBeamRadon2D(
            image_size=image_size, num_angles=num_angles, angles=angles,
            device=device, in_channels=in_channels)
        self.image_size = self.radon.image_size
        self.num_angles = self.radon.num_angles
        self.device = device
        self.inner = int(self.image_size if inner is None else inner)
        self.trunc = DetectorTruncation(
            num_angles=self.num_angles, width=self.image_size,
            inner=self.inner, device=device, in_channels=in_channels)
        self.domain_shape = self.radon.domain_shape
        self.range_shape = self.radon.range_shape  # detector width unchanged

    @property
    def angles(self) -> torch.Tensor:
        return self.radon.angles

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunc.forward(self.radon.forward(x))

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        return self.radon.adjoint(self.trunc.adjoint(y))

    def pseudo_inverse(self, y: torch.Tensor, method: str = 'fbp',
                       **kwargs) -> torch.Tensor:
        """Warm-start / display reconstruction via the underlying Radon.

        Truncated FBP exhibits the expected interior-tomography cupping; it is
        meant only for initialisation or visualisation, with the outer field
        recovered by a prior + data consistency.
        """
        return self.radon.pseudo_inverse(y, method=method, **kwargs)
