"""ASTRA CUDA operators: lab cone/fan beam and synchrotron 3D parallel beam.

``ConeBeamASTRA`` follows ``bmc/operators.py:AstraCone`` of the battery
multi-cell benchmark:

* explicit ``cone_vec`` geometry (the vectors ``astra.geom_2vec`` builds for a
  ``'cone'`` geometry, plus an optional detector shift along u);
* FP3D_CUDA / BP3D_CUDA on persistent torch buffers linked into ASTRA, so no
  host round trip;
* an optional ray-obliquity weight on the back-projection. ASTRA's cone BP is
  not the transpose of its FP (relative dot-test error ~1e-2); weighting each
  detector pixel by 1/cos(ray, detector normal) brings it to ~1e-6.

``ConeBeamASTRA`` and ``ParallelBeamASTRA3D`` (3D): domain ``(nz, ny, nx)``,
range ``(det_rows, num_angles, det_cols)`` in ASTRA order. The parallel-beam
operator is the synchrotron volume geometry; its rays are normal to the
detector, so no obliquity weight is needed, and every detector row maps to one
slice (the same problem as stacking 2D parallel-beam slices).

``FanBeamASTRA2D`` is the mid-plane of the same scanner on ASTRA's 2D kernels,
with the framework's 2D shapes ``(1, H, W)`` and ``(1, n_angles, W)``, so every
2D solver and prior runs unchanged. A one-row cone geometry is not used for
this: the 3D kernels' FP/BP mismatch grows near the volume and detector edges
(dot-test error 1e-2 with one slice and one row, 4e-5 from eight). The 2D BP
with the in-plane 1/cos weight matches the 2D FP to ~1e-5.

Lengths (``sod``, ``sdd``, ``voxel_size``, detector pitch) share one unit, and
projections are line integrals in that unit: with attenuation per unit length
in the volume, the sinogram is dimensionless (-log I/I0).
"""

import math

import numpy as np
import torch

from ..base import LinearOperator


def circular_cone_geometry(volume_shape, det_shape, sod, sdd, voxel_size=1.0,
                           det_pixel=None, num_angles=360, angles=None,
                           det_offset_u=0.0):
    """Geometry dict for a circular cone-beam scan around the z axis.

    volume_shape : (nz, ny, nx) voxels, centred on the rotation axis.
    det_shape    : (det_rows, det_cols).
    sod, sdd     : source-object and source-detector distances.
    voxel_size   : isotropic voxel pitch.
    det_pixel    : (dv, du) detector pitch; default ``voxel_size * sdd / sod``
                   (one voxel at the axis maps to one detector pixel).
    angles       : source angles in radians; default ``num_angles`` uniform
                   angles over a full turn [0, 2 pi).
    det_offset_u : detector-centre shift along u, in detector pixels.
    """
    nz, ny, nx = (int(n) for n in volume_shape)
    rows, cols = (int(n) for n in det_shape)
    if not 0 < sod < sdd:
        raise ValueError("expected 0 < sod < sdd")
    magnification = sdd / sod
    if det_pixel is None:
        det_pixel = (voxel_size * magnification,) * 2
    dv, du = (float(p) for p in det_pixel)
    if angles is None:
        angles = np.linspace(0, 2 * np.pi, int(num_angles), endpoint=False)
    angles = np.asarray(angles, dtype=np.float64)
    sin, cos, zero = np.sin(angles), np.cos(angles), np.zeros_like(angles)
    src = np.stack([sin * sod, -cos * sod, zero], 1)
    det = np.stack([-sin * (sdd - sod), cos * (sdd - sod), zero], 1)
    u = np.stack([cos * du, sin * du, zero], 1)
    v = np.stack([zero, zero, np.full_like(angles, dv)], 1)
    det = det + det_offset_u * u
    half = np.array([nx, ny, nz], dtype=np.float64) * voxel_size / 2
    return dict(kind="cone", volume_shape=(nz, ny, nx), det_shape=(rows, cols), sod=float(sod),
                sdd=float(sdd), voxel_size=float(voxel_size), det_pixel=(dv, du),
                angles=angles.astype(np.float32), det_offset_u=float(det_offset_u),
                vectors=np.concatenate([src, det, u, v], 1),
                window=(-half[0], half[0], -half[1], half[1], -half[2], half[2]))


def parallel_beam_3d_geometry(volume_shape, det_shape, voxel_size=1.0, det_pixel=None,
                              num_angles=180, angles=None, det_offset_u=0.0):
    """Geometry dict for a synchrotron-style parallel-beam scan around the z axis.

    volume_shape : (nz, ny, nx) voxels, centred on the rotation axis.
    det_shape    : (det_rows, det_cols).
    det_pixel    : (dv, du) detector pitch; default ``voxel_size``.
    angles       : projection angles in radians; default ``num_angles``
                   uniform angles over [0, pi).
    det_offset_u : rotation-axis offset on the detector along u, in pixels.
    """
    nz, ny, nx = (int(n) for n in volume_shape)
    rows, cols = (int(n) for n in det_shape)
    if det_pixel is None:
        det_pixel = (voxel_size, voxel_size)
    dv, du = (float(p) for p in det_pixel)
    if angles is None:
        angles = np.linspace(0, np.pi, int(num_angles), endpoint=False)
    angles = np.asarray(angles, dtype=np.float64)
    sin, cos, zero = np.sin(angles), np.cos(angles), np.zeros_like(angles)
    ray = np.stack([sin, -cos, zero], 1)          # astra.geom_2vec('parallel3d') convention
    u = np.stack([cos * du, sin * du, zero], 1)
    v = np.stack([zero, zero, np.full_like(angles, dv)], 1)
    det = det_offset_u * u
    half = np.array([nx, ny, nz], dtype=np.float64) * voxel_size / 2
    return dict(kind="parallel3d", volume_shape=(nz, ny, nx), det_shape=(rows, cols),
                voxel_size=float(voxel_size), det_pixel=(dv, du),
                angles=angles.astype(np.float32), det_offset_u=float(det_offset_u),
                vectors=np.concatenate([ray, det, u, v], 1),
                window=(-half[0], half[0], -half[1], half[1], -half[2], half[2]))


def fan_beam_geometry(image_size, det_cols, sod, sdd, voxel_size=1.0, det_pixel=None,
                      num_angles=360, angles=None, det_offset_u=0.0):
    """Geometry dict for the fan-beam mid-plane of ``circular_cone_geometry``.

    Same conventions (source at angle 0 on -y, detector u along +x); the
    detector pitch defaults to ``voxel_size * sdd / sod``.
    """
    if not 0 < sod < sdd:
        raise ValueError("expected 0 < sod < sdd")
    du = voxel_size * sdd / sod if det_pixel is None else float(det_pixel)
    if angles is None:
        angles = np.linspace(0, 2 * np.pi, int(num_angles), endpoint=False)
    angles = np.asarray(angles, dtype=np.float64)
    sin, cos = np.sin(angles), np.cos(angles)
    src = np.stack([sin * sod, -cos * sod], 1)
    det = np.stack([-sin * (sdd - sod), cos * (sdd - sod)], 1)
    u = np.stack([cos * du, sin * du], 1)
    det = det + det_offset_u * u
    half = int(image_size) * voxel_size / 2
    return dict(image_size=int(image_size), det_cols=int(det_cols), sod=float(sod),
                sdd=float(sdd), voxel_size=float(voxel_size), det_pixel=du,
                angles=angles.astype(np.float32), det_offset_u=float(det_offset_u),
                vectors=np.concatenate([src, det, u], 1), window=(-half, half, -half, half))


def obliquity_weights(geometry):
    """1/cos(ray, detector normal) per detector pixel.

    Shape (rows, 1, cols) for a cone geometry and (1, 1, cols) for a fan
    geometry. For a circular orbit the detector moves rigidly with the
    source, so the weight is the same for every view.
    """
    vec = np.asarray(geometry["vectors"], np.float64)
    if vec.shape[1] == 6:
        src, det, u = vec[0, 0:2], vec[0, 2:4], vec[0, 4:6]
        normal = np.array([-u[1], u[0]]) / np.linalg.norm(u)
        c = np.arange(geometry["det_cols"]) - (geometry["det_cols"] - 1) / 2
        ray = det + c[:, None] * u - src
        cos = np.abs(ray @ normal) / np.linalg.norm(ray, axis=-1)
        return (1.0 / cos)[None, None, :].astype(np.float32)
    rows, cols = geometry["det_shape"]
    src, det, u, v = vec[0, 0:3], vec[0, 3:6], vec[0, 6:9], vec[0, 9:12]
    normal = np.cross(u, v)
    normal /= np.linalg.norm(normal)
    r = np.arange(rows) - (rows - 1) / 2
    c = np.arange(cols) - (cols - 1) / 2
    pixels = det + r[:, None, None] * v + c[None, :, None] * u
    ray = pixels - src
    cos = np.abs(ray @ normal) / np.linalg.norm(ray, axis=-1)
    return (1.0 / cos)[:, None, :].astype(np.float32)


class _Project(torch.autograd.Function):
    @staticmethod
    def forward(ctx, volume, op):
        ctx.op = op
        return op._apply(volume, back=False)

    @staticmethod
    def backward(ctx, grad):
        return _BackProject.apply(grad, ctx.op), None


class _BackProject(torch.autograd.Function):
    @staticmethod
    def forward(ctx, projection, op):
        ctx.op = op
        return op._apply(projection, back=True)

    @staticmethod
    def backward(ctx, grad):
        return _Project.apply(grad, ctx.op), None


class _ASTRA3D(LinearOperator):
    """FP3D_CUDA / BP3D_CUDA on GPU-linked buffers for a ``*_vec`` geometry."""

    _projection_type = None

    def __init__(self, geometry, obliquity=False, device=None):
        try:
            import astra
        except ImportError as error:
            raise ImportError(f"{type(self).__name__} requires astra-toolbox "
                              "(conda install -c astra-toolbox astra-toolbox)") from error
        if not torch.cuda.is_available():
            raise RuntimeError(f"{type(self).__name__} requires a CUDA GPU")
        self.astra = astra
        self.device = (torch.device("cuda", torch.cuda.current_device()) if device is None
                       else torch.device(device))
        if self.device.type != "cuda":
            raise ValueError(f"{type(self).__name__} runs on CUDA only")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.geometry = geometry
        nz, ny, nx = geometry["volume_shape"]
        rows, cols = geometry["det_shape"]
        self.num_angles = len(geometry["angles"])
        self.angles = torch.as_tensor(geometry["angles"], dtype=torch.float32)
        self.domain_shape = (nz, ny, nx)
        self.range_shape = (rows, self.num_angles, cols)
        self._pg = astra.create_proj_geom(self._projection_type, rows, cols, geometry["vectors"])
        x0, x1, y0, y1, z0, z1 = geometry["window"]
        self._vg = astra.create_vol_geom(ny, nx, nz, x0, x1, y0, y1, z0, z1)
        with torch.cuda.device(self.device):
            self._volume = torch.zeros(self.domain_shape, dtype=torch.float32, device=self.device)
            self._projection = torch.zeros(self.range_shape, dtype=torch.float32, device=self.device)
        self._vid = astra.data3d.link("-vol", self._vg, self._volume)
        self._pid = astra.data3d.link("-sino", self._pg, self._projection)
        option = {"GPUindex": self.device.index}
        fp = astra.astra_dict("FP3D_CUDA")
        fp.update(VolumeDataId=self._vid, ProjectionDataId=self._pid, option=option)
        bp = astra.astra_dict("BP3D_CUDA")
        bp.update(ReconstructionDataId=self._vid, ProjectionDataId=self._pid, option=option)
        self._fp = astra.algorithm.create(fp)
        self._bp = astra.algorithm.create(bp)
        self._weight = (torch.from_numpy(obliquity_weights(geometry)).to(self.device)
                        if obliquity else None)

    def _run(self, algorithm):
        torch.cuda.synchronize(self.device)
        self.astra.algorithm.run(algorithm)
        torch.cuda.synchronize(self.device)

    def _apply(self, value, back):
        """Batched FP (back=False) or weighted BP (back=True) without autograd."""
        in_shape, out_shape = ((self.range_shape, self.domain_shape) if back
                               else (self.domain_shape, self.range_shape))
        if tuple(value.shape[1:]) != in_shape:
            raise ValueError(f"expected (B, {in_shape}), got {tuple(value.shape)}")
        source, target = ((self._projection, self._volume) if back
                          else (self._volume, self._projection))
        out = torch.empty((value.shape[0], *out_shape), dtype=torch.float32, device=self.device)
        for i in range(value.shape[0]):
            item = value[i].detach().to(device=self.device, dtype=torch.float32)
            if back and self._weight is not None:
                item = item * self._weight
            source.copy_(item)
            target.zero_()
            self._run(self._bp if back else self._fp)
            out[i].copy_(target)
        return out.to(value.dtype) if value.is_floating_point() else out

    def forward(self, x):
        return _Project.apply(x, self)

    def adjoint(self, y):
        return _BackProject.apply(y, self)

    def close(self):
        """Release the ASTRA algorithms and linked data objects."""
        if getattr(self, "_fp", None) is not None:
            self.astra.algorithm.delete([self._fp, self._bp])
            self.astra.data3d.delete([self._vid, self._pid])
            self._fp = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class ConeBeamASTRA(_ASTRA3D):
    """``A`` = ASTRA cone-beam FP; ``adjoint`` = (obliquity-weighted) BP.

    Parameters
    ----------
    geometry : dict from ``circular_cone_geometry``.
    obliquity : weight the BP by 1/cos so it matches the FP's transpose.
    device : CUDA device; defaults to the current one.
    """

    _projection_type = "cone_vec"

    def __init__(self, geometry, obliquity=True, device=None):
        if geometry.get("kind", "cone") != "cone":
            raise ValueError("ConeBeamASTRA needs a circular_cone_geometry")
        super().__init__(geometry, obliquity=obliquity, device=device)

    def fdk(self, y, filter_type="ram-lak", short_scan=False):
        """ASTRA FDK_CUDA reconstruction of a batch of projections."""
        astra = self.astra
        config = astra.astra_dict("FDK_CUDA")
        config.update(ProjectionDataId=self._pid, ReconstructionDataId=self._vid,
                      option={"GPUindex": self.device.index, "FilterType": filter_type,
                              "ShortScan": bool(short_scan)})
        algorithm = astra.algorithm.create(config)
        try:
            out = torch.empty((y.shape[0], *self.domain_shape), dtype=torch.float32,
                              device=self.device)
            for i in range(y.shape[0]):
                self._projection.copy_(y[i].detach().to(self.device, torch.float32))
                self._volume.zero_()
                self._run(algorithm)
                out[i].copy_(self._volume)
            return out
        finally:
            astra.algorithm.delete(algorithm)

    def pseudo_inverse(self, y, method="fdk", **kwargs):
        if method == "fdk":
            return self.fdk(y, **kwargs)
        from inv_framework.solvers.classical import landweber, sirt
        if method == "sirt":
            return sirt(self, y, **kwargs)
        if method == "landweber":
            return landweber(self, y, **kwargs)
        raise ValueError(f"Unknown pseudo_inverse method: {method!r}")



class ParallelBeamASTRA3D(_ASTRA3D):
    """Synchrotron 3D parallel beam: ``A`` = ASTRA FP3D, ``adjoint`` = BP3D.

    Parameters
    ----------
    geometry : dict from ``parallel_beam_3d_geometry``.
    device : CUDA device; defaults to the current one.
    """

    _projection_type = "parallel3d_vec"

    def __init__(self, geometry, device=None):
        if geometry.get("kind") != "parallel3d":
            raise ValueError("ParallelBeamASTRA3D needs a parallel_beam_3d_geometry")
        super().__init__(geometry, obliquity=False, device=device)

    def fbp(self, y, filter_type="ram-lak"):
        """Ram-Lak filtering along u, then BP: exact FBP for parallel beam.

        Assumes angles uniformly covering [0, pi) or [0, 2 pi).
        """
        if filter_type != "ram-lak":
            raise ValueError("ParallelBeamASTRA3D.fbp implements only filter_type='ram-lak'")
        from inv_framework.solvers.classical import _filter_sino
        filtered = _filter_sino(y.to(self.device, torch.float32))
        # The world-unit ramp filter carries 1/du; ASTRA's discrete BP3D carries
        # du/voxel^2 relative to the continuous back-projection (measured over
        # voxel and pitch sweeps), leaving 1/voxel^2.
        voxel = self.geometry["voxel_size"]
        return self.adjoint(filtered) * (math.pi / self.num_angles / voxel ** 2)

    def pseudo_inverse(self, y, method="fbp", **kwargs):
        if method == "fbp":
            return self.fbp(y, **kwargs)
        from inv_framework.solvers.classical import landweber, sirt
        if method == "sirt":
            return sirt(self, y, **kwargs)
        if method == "landweber":
            return landweber(self, y, **kwargs)
        raise ValueError(f"Unknown pseudo_inverse method: {method!r}")


class FanBeamASTRA2D(LinearOperator):
    """2D fan-beam ``A`` = ASTRA FP_CUDA; ``adjoint`` = 1/cos-weighted BP_CUDA.

    Parameters
    ----------
    geometry : dict from ``fan_beam_geometry``.
    obliquity : weight the BP by 1/cos so it matches the FP's transpose.
    device : CUDA device for ASTRA and for the returned tensors.

    ASTRA's 2D data live in host memory, so each call stages one image or
    sinogram through linked NumPy buffers.
    """

    def __init__(self, geometry, obliquity=True, device=None):
        try:
            import astra
        except ImportError as error:
            raise ImportError("FanBeamASTRA2D requires astra-toolbox "
                              "(conda install -c astra-toolbox astra-toolbox)") from error
        if not torch.cuda.is_available():
            raise RuntimeError("FanBeamASTRA2D requires a CUDA GPU")
        self.astra = astra
        self.device = torch.device("cuda", torch.cuda.current_device() if device is None
                                   else torch.device(device).index or 0)
        self.geometry = geometry
        n, cols = geometry["image_size"], geometry["det_cols"]
        self.num_angles = len(geometry["angles"])
        self.angles = torch.as_tensor(geometry["angles"], dtype=torch.float32)
        self.image_size = n
        self.domain_shape = (1, n, n)
        self.range_shape = (1, self.num_angles, cols)
        self._pg = astra.create_proj_geom("fanflat_vec", cols, geometry["vectors"])
        x0, x1, y0, y1 = geometry["window"]
        self._vg = astra.create_vol_geom(n, n, x0, x1, y0, y1)
        self._volume = np.zeros((n, n), np.float32)
        self._projection = np.zeros((self.num_angles, cols), np.float32)
        self._vid = astra.data2d.link("-vol", self._vg, self._volume)
        self._pid = astra.data2d.link("-sino", self._pg, self._projection)
        option = {"GPUindex": self.device.index}
        fp = astra.astra_dict("FP_CUDA")
        fp.update(VolumeDataId=self._vid, ProjectionDataId=self._pid, option=option)
        bp = astra.astra_dict("BP_CUDA")
        bp.update(ReconstructionDataId=self._vid, ProjectionDataId=self._pid, option=option)
        self._fp = astra.algorithm.create(fp)
        self._bp = astra.algorithm.create(bp)
        self._weight = (torch.from_numpy(obliquity_weights(geometry)[0]) if obliquity else None)

    def _apply(self, value, back):
        in_shape, out_shape = ((self.range_shape, self.domain_shape) if back
                               else (self.domain_shape, self.range_shape))
        if tuple(value.shape[1:]) != in_shape:
            raise ValueError(f"expected (B, {in_shape}), got {tuple(value.shape)}")
        source, target = ((self._projection, self._volume) if back
                          else (self._volume, self._projection))
        out = torch.empty((value.shape[0], *out_shape), dtype=torch.float32)
        for i in range(value.shape[0]):
            item = value[i, 0].detach().float().cpu()
            if back and self._weight is not None:
                item = item * self._weight
            source[...] = item.numpy()
            target[...] = 0
            self.astra.algorithm.run(self._bp if back else self._fp)
            out[i, 0] = torch.from_numpy(target)
        return out.to(device=value.device, dtype=value.dtype)

    def forward(self, x):
        return _Project.apply(x, self)

    def adjoint(self, y):
        return _BackProject.apply(y, self)

    def fbp(self, y, filter_type="ram-lak", short_scan=False):
        """ASTRA FBP_CUDA (fan-beam weighting included) of a batch of sinograms."""
        astra = self.astra
        config = astra.astra_dict("FBP_CUDA")
        config.update(ProjectionDataId=self._pid, ReconstructionDataId=self._vid,
                      option={"GPUindex": self.device.index, "FilterType": filter_type,
                              "ShortScan": bool(short_scan)})
        algorithm = astra.algorithm.create(config)
        try:
            out = torch.empty((y.shape[0], *self.domain_shape), dtype=torch.float32)
            for i in range(y.shape[0]):
                self._projection[...] = y[i, 0].detach().float().cpu().numpy()
                self._volume[...] = 0
                astra.algorithm.run(algorithm)
                out[i, 0] = torch.from_numpy(self._volume)
            return out.to(y.device)
        finally:
            astra.algorithm.delete(algorithm)

    def pseudo_inverse(self, y, method="fbp", **kwargs):
        if method == "fbp":
            return self.fbp(y, **kwargs)
        from inv_framework.solvers.classical import landweber, sirt
        if method == "sirt":
            return sirt(self, y, **kwargs)
        if method == "landweber":
            return landweber(self, y, **kwargs)
        raise ValueError(f"Unknown pseudo_inverse method: {method!r}")

    def close(self):
        if getattr(self, "_fp", None) is not None:
            self.astra.algorithm.delete([self._fp, self._bp])
            self.astra.data2d.delete([self._vid, self._pid])
            self._fp = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
