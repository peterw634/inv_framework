"""ASTRA-toolbox backend for 3D cone- or parallel-beam CT.

Wraps astra's CUDA3D projector behind the LinearOperator interface, so any
solver in this framework can run against ASTRA without changes. Mirrors the
operator from DM4CT (forward_operators_ct.py).

Requires astra-toolbox + a CUDA GPU. Import lazily so the rest of the package
can be used CPU-only.
"""

import torch

try:
    import astra
    import astra.experimental  # noqa: F401
    _HAS_ASTRA = True
except ImportError as _e:
    _HAS_ASTRA = False
    _IMPORT_ERROR = _e

from ..base import LinearOperator


def _require_astra():
    if not _HAS_ASTRA:
        raise ImportError(
            "ASTRAOperator3D requires astra-toolbox. "
            "Install with: conda install -c astra-toolbox astra-toolbox"
        ) from _IMPORT_ERROR


class _ASTRAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, volume, projector, projection_shape, volume_shape):
        if volume.ndim == 4:
            B = volume.shape[0]
            proj = torch.zeros((B, *projection_shape), dtype=torch.float32,
                               device='cuda')
            for i in range(B):
                astra.experimental.direct_FP3D(
                    projector, vol=volume[i].contiguous().detach(),
                    proj=proj[i])
        elif volume.ndim == 3:
            proj = torch.zeros(projection_shape, dtype=torch.float32,
                               device='cuda')
            astra.experimental.direct_FP3D(
                projector, vol=volume.detach(), proj=proj)
        else:
            raise ValueError(f"Unsupported volume.ndim={volume.ndim}")
        ctx.projector = projector
        ctx.volume_shape = volume_shape
        return proj

    @staticmethod
    def backward(ctx, grad_output):
        projector = ctx.projector
        volume_shape = ctx.volume_shape
        if grad_output.ndim == 4:
            B = grad_output.shape[0]
            grad_vol = torch.zeros((B, *volume_shape), dtype=torch.float32,
                                   device='cuda')
            for i in range(B):
                astra.experimental.direct_BP3D(
                    projector, vol=grad_vol[i],
                    proj=grad_output[i].contiguous().detach())
        else:
            grad_vol = torch.zeros(volume_shape, dtype=torch.float32,
                                   device='cuda')
            astra.experimental.direct_BP3D(
                projector, vol=grad_vol, proj=grad_output.detach())
        return grad_vol, None, None, None


class ASTRAOperator3D(LinearOperator):
    """ASTRA CUDA3D projector wrapped as a LinearOperator.

    Parameters
    ----------
    volume_geometry : dict
        astra volume geometry (e.g. astra.create_vol_geom(...)).
    projection_geometry : dict
        astra projection geometry (parallel or cone).
    """

    def __init__(self, volume_geometry, projection_geometry):
        _require_astra()
        self.volume_geometry = volume_geometry
        self.projection_geometry = projection_geometry
        self.projector = astra.create_projector(
            'cuda3d', projection_geometry, volume_geometry)
        self.projection_shape = tuple(astra.geom_size(projection_geometry))
        self.volume_shape = tuple(astra.geom_size(volume_geometry))
        self.domain_shape = self.volume_shape
        self.range_shape = self.projection_shape
        if 'ProjectionAngles' in projection_geometry:
            self.num_angles = len(projection_geometry['ProjectionAngles'])
        else:
            self.num_angles = len(projection_geometry['Vectors'])

    def forward(self, x):
        return _ASTRAFunction.apply(
            x, self.projector, self.projection_shape, self.volume_shape)

    def adjoint(self, y):
        if y.ndim == 4:
            B = y.shape[0]
            vol = torch.zeros((B, *self.volume_shape), dtype=torch.float32,
                              device='cuda')
            for i in range(B):
                astra.experimental.direct_BP3D(
                    self.projector, vol=vol[i],
                    proj=y[i].contiguous().detach())
        elif y.ndim == 3:
            vol = torch.zeros(self.volume_shape, dtype=torch.float32,
                              device='cuda')
            astra.experimental.direct_BP3D(
                self.projector, vol=vol, proj=y.detach())
        else:
            raise ValueError(f"Unsupported y.ndim={y.ndim}")
        return vol
