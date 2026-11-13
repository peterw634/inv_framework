"""ASTRA operators (lab cone/fan beam, synchrotron 3D parallel beam): adjoint, units, analytic recon."""
import numpy as np
import pytest
import torch

astra = pytest.importorskip("astra")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")

from inv_framework.operators.ct import (ConeBeamASTRA, FanBeamASTRA2D,  # noqa: E402
                                        circular_cone_geometry, fan_beam_geometry)


def _dot(a, b):
    return float((a.double() * b.double()).sum())


def _relative_adjoint_error(A):
    gen = torch.Generator(device="cuda").manual_seed(0)
    x = torch.rand((2, *A.domain_shape), device="cuda", generator=gen)
    y = torch.rand((2, *A.range_shape), device="cuda", generator=gen)
    lhs = _dot(A(x), y)
    return abs(lhs - _dot(x, A.adjoint(y))) / abs(lhs)


def _cylinder(shape, radius):
    ny, nx = shape[-2:]
    yy, xx = np.meshgrid(np.arange(ny) - (ny - 1) / 2, np.arange(nx) - (nx - 1) / 2, indexing="ij")
    disk = ((yy ** 2 + xx ** 2) < radius ** 2).astype(np.float32)
    return torch.from_numpy(np.broadcast_to(disk, shape).copy()).cuda()[None]


def test_cone_beam_obliquity_weight_matches_adjoint():
    geometry = circular_cone_geometry((32, 64, 64), (48, 96), sod=200, sdd=300, num_angles=60)
    weighted, plain = ConeBeamASTRA(geometry), ConeBeamASTRA(geometry, obliquity=False)
    assert _relative_adjoint_error(weighted) < 1e-4
    assert _relative_adjoint_error(weighted) < _relative_adjoint_error(plain)


def test_cone_beam_units_and_fdk_scale():
    geometry = circular_cone_geometry((32, 64, 64), (48, 96), sod=200, sdd=300, num_angles=120)
    A = ConeBeamASTRA(geometry)
    volume = _cylinder(A.domain_shape, 20)
    projection = A(volume)
    rows, _, cols = A.range_shape
    central = float(projection[0, rows // 2, 0, cols // 2 - 1:cols // 2 + 1].mean())
    assert abs(central - 40) < 2                       # chord through the axis = diameter
    reconstruction = A.fdk(projection)
    assert abs(float(reconstruction[0, 16, 28:36, 28:36].mean()) - 1) < 0.02


def test_fan_beam_adjoint_fbp_and_autograd():
    geometry = fan_beam_geometry(128, 144, sod=256, sdd=384, num_angles=90)
    A = FanBeamASTRA2D(geometry)
    assert _relative_adjoint_error(A) < 1e-4
    image = _cylinder(A.domain_shape, 40)
    reconstruction = A.fbp(A(image))
    assert abs(float(reconstruction[0, 0, 60:68, 60:68].mean()) - 1) < 0.02
    x = image.clone().requires_grad_()
    y = torch.rand_like(A(image))
    (grad,) = torch.autograd.grad((A(x) * y).sum(), x)
    torch.testing.assert_close(grad, A.adjoint(y))


def test_parallel_3d_adjoint_units_and_fbp_scale():
    from inv_framework.operators.ct import ParallelBeamASTRA3D, parallel_beam_3d_geometry
    for voxel, pitch in [(1.0, 1.0), (0.5, 0.75), (0.07, 0.07)]:
        cols = int(64 * voxel / pitch) + 8
        A = ParallelBeamASTRA3D(parallel_beam_3d_geometry((8, 64, 64), (8, cols), voxel_size=voxel,
                                                          det_pixel=(voxel, pitch), num_angles=90))
        assert _relative_adjoint_error(A) < 1e-4
        volume = _cylinder(A.domain_shape, 20)
        projection = A(volume)
        central = float(projection[0, 4, 0, cols // 2 - 1:cols // 2 + 1].mean())
        assert abs(central - 40 * voxel) < 0.05 * 40 * voxel
        assert abs(float(A.fbp(projection)[0, 4, 28:36, 28:36].mean()) - 1) < 0.02
