"""JIT-loaded TomoPy-compatible CUDA tomography operators.

The public surface deliberately contains only ``project``, its strict
unfiltered adjoint ``bp``, and the independently compiled Gridrec port.
Importing this module on a CPU machine does not trigger compilation.
"""
from __future__ import annotations

import hashlib
import os
import sysconfig
import weakref
from pathlib import Path
from threading import Lock

_HERE = Path(__file__).resolve().parent
_EXT = None
_LOCK = Lock()
_CACHE_LOCK = Lock()
_THETA_REF = None
_THETA_VERSION = None
_GEOMETRY = None


def _cuda_includes():
    root = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    if not root.is_dir():
        return []
    return [str(path / "include") for path in root.iterdir() if (path / "include").is_dir()]


def _math_mode():
    mode = os.getenv("NASN_RADON_MATH_MODE", "strict").strip().lower()
    if mode not in {"strict", "precise", "fast"}:
        raise ValueError("NASN_RADON_MATH_MODE must be strict, precise, or fast")
    return mode


def _source_key(paths):
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def ext():
    """Build once in PyTorch's extension cache and return the CUDA module."""
    global _EXT
    if _EXT is not None:
        return _EXT
    with _LOCK:
        if _EXT is not None:
            return _EXT
        from torch.utils.cpp_extension import load

        sources = [_HERE / "radon_cuda.cpp", _HERE / "radon_cuda_kernel.cu"]
        mode = _math_mode()
        includes = [f"-I{path}" for path in _cuda_includes()]
        cuda_flags = ["-O3", "-lineinfo", *includes]
        if mode == "strict":
            cuda_flags.append("--fmad=false")
        elif mode == "fast":
            cuda_flags.append("--use_fast_math")
        _EXT = load(
            name=f"nasn_radon_{_source_key(sources)}_{mode}",
            sources=[str(path) for path in sources],
            extra_cflags=["-O3", *includes],
            extra_cuda_cflags=cuda_flags,
            verbose=os.getenv("NASN_CUDA_VERBOSE", "0") == "1",
        )
        return _EXT


def _validate_theta(theta):
    import torch
    if not isinstance(theta, torch.Tensor) or not theta.is_cuda:
        raise RuntimeError("theta must be a CUDA torch.Tensor")
    if theta.dtype != torch.float32 or theta.ndim != 1 or not theta.is_contiguous():
        raise ValueError("theta must be contiguous float32 with shape (angles,)")


def prepare_geometry(theta):
    _validate_theta(theta)
    import torch
    cpu = theta.detach().to(device="cpu", dtype=torch.float32).contiguous()
    sine, cosine, quadrant = ext().prepare_geometry(cpu)
    return sine.to(theta.device), cosine.to(theta.device), quadrant.to(theta.device)


def _cached_geometry(theta):
    global _THETA_REF, _THETA_VERSION, _GEOMETRY
    _validate_theta(theta)
    with _CACHE_LOCK:
        cached = _THETA_REF() if _THETA_REF is not None else None
        if cached is theta and _THETA_VERSION == int(theta._version):
            return _GEOMETRY
        _GEOMETRY = prepare_geometry(theta)
        _THETA_REF = weakref.ref(theta)
        _THETA_VERSION = int(theta._version)
        return _GEOMETRY


def _check_geometry(geometry, theta):
    import torch
    if not isinstance(geometry, (tuple, list)) or len(geometry) != 3:
        raise TypeError("geometry must be (sin, cos, quadrant)")
    for value, dtype in zip(geometry, (torch.float32, torch.float32, torch.int32)):
        if value.device != theta.device or value.dtype != dtype or value.shape != theta.shape:
            raise ValueError("geometry tensors do not match theta")
    return tuple(value.contiguous() for value in geometry)


def _centers(center, batch, device):
    import torch
    value = torch.as_tensor(center, device=device, dtype=torch.float32)
    if value.numel() == 1:
        value = value.reshape(1).expand(batch).clone()
    elif value.numel() == batch:
        value = value.reshape(batch)
    else:
        raise ValueError("center must be scalar or one value per batch item")
    return value.contiguous()


def project(image, theta, center, *, geometry=None):
    """TomoPy-compatible parallel-beam projection, layout ``(B,A,D)``."""
    tables = _cached_geometry(theta) if geometry is None else _check_geometry(geometry, theta)
    return ext().project(image, *tables, _centers(center, image.shape[0], image.device))


def bp(sinogram, theta, center, ngridx, ngridy, *, init_recon=None, geometry=None):
    """Strict unfiltered adjoint/back-projection, not filtered FBP."""
    import torch
    shape = (sinogram.shape[0], int(ngridx), int(ngridy))
    if init_recon is None:
        recon = torch.zeros(shape, device=sinogram.device, dtype=torch.float32)
    else:
        if tuple(init_recon.shape) != shape:
            raise ValueError(f"init_recon must have shape {shape}")
        recon = init_recon.to(device=sinogram.device, dtype=torch.float32).contiguous().clone()
    tables = _cached_geometry(theta) if geometry is None else _check_geometry(geometry, theta)
    return ext().bp(
        sinogram, *tables,
        _centers(center, sinogram.shape[0], sinogram.device), recon,
    )


from .gridrec import GridrecCUDA, TomoPyGridrecCUDA, gridrec_cuda, gridrec_ext

__all__ = [
    "GridrecCUDA", "TomoPyGridrecCUDA", "bp", "ext", "gridrec_cuda",
    "gridrec_ext", "prepare_geometry", "project",
]
