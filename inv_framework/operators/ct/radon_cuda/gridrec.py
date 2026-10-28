"""CUDA port of TomoPy 1.15.x ``algorithm='gridrec'``.

The implementation follows ``source/libtomo/gridrec/gridrec.c`` rather than
the separately parameterized PITRE Gridrec code in this repository.  It keeps
TomoPy's fixed PSWF constants, filtering, center phase, FFT directions,
Fourier-grid dimensions, correction factors, crop, and output orientation.

No TomoPy or ASTRA routine is called by the reconstruction path.  TomoPy is
used only by the external regression test as the numerical reference.
"""
from __future__ import annotations

import math
import hashlib
import os
from pathlib import Path
from threading import Lock

import numpy as np
import torch
import torch.nn as nn


_THIS_DIR = Path(__file__).resolve().parent
_EXTENSION = None
_EXTENSION_LOCK = Lock()

_FILTER_NAMES = {
    'none',
    'shepp',
    'cosine',
    'hann',
    'hamming',
    'ramlak',
    'parzen',
    'butterworth',
    'custom',
    'custom2d',
}


def _find_cuda_includes():
    import sysconfig

    site_packages = Path(sysconfig.get_paths()['purelib'])
    nvidia_root = site_packages / 'nvidia'
    if not nvidia_root.is_dir():
        return []
    includes = []
    for child in nvidia_root.iterdir():
        include = child / 'include'
        if include.is_dir():
            includes.append(str(include))
    return includes


def gridrec_ext():
    """Build once and return the local C++/CUDA extension."""
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    with _EXTENSION_LOCK:
        if _EXTENSION is not None:
            return _EXTENSION
        from torch.utils.cpp_extension import load

        sources = [
            _THIS_DIR / 'gridrec_cuda.cpp',
            _THIS_DIR / 'gridrec_cuda_kernel.cu',
        ]
        digest = hashlib.sha256()
        for source in sources:
            digest.update(source.read_bytes())
        includes = _find_cuda_includes()
        include_flags = [f'-I{path}' for path in includes]
        _EXTENSION = load(
            name=f'nasn_gridrec_{digest.hexdigest()[:12]}',
            sources=[str(source) for source in sources],
            extra_cflags=['-O3', *include_flags],
            extra_cuda_cflags=[
                '-O3',
                '--fmad=false',
                '-lineinfo',
                *include_flags,
            ],
            verbose=os.environ.get('NASN_CUDA_VERBOSE', '0') == '1',
        )
        return _EXTENSION


def _next_power_of_two_at_least_detector(detectors: int) -> int:
    pdim = 16
    while pdim < int(detectors):
        pdim *= 2
    return pdim


def _as_cpu_float32(value, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device='cpu')
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    if tensor.numel() == 0:
        raise ValueError(f'{name} must not be empty')
    return tensor


def _canonical_filter_name(filter_name: str) -> str:
    name = str(filter_name).strip().lower().replace('_', '-')
    aliases = {
        'ramp': 'ramlak',
        'ram-lak': 'ramlak',
        'shepp-logan': 'shepp',
        'hamm': 'hamming',
    }
    return aliases.get(name, name)


class TomoPyGridrecCUDA(nn.Module):
    """TomoPy-compatible Gridrec reconstruction on CUDA.

    Args:
        n_det: Detector width ``dx``.
        theta: Explicit projection angles in radians, matching TomoPy's
            float32 ``theta`` argument.
        center: Scalar center, one center per reconstructed slice, or ``None``
            for TomoPy's ``n_det / 2`` default. A scalar (or an equal-valued
            vector) uses TomoPy's two-slice complex packing. Slices with
            different centers are reconstructed independently so the result
            follows the public ``tomopy.recon`` per-slice-center contract and
            does not depend on TomoPy's ``ncore`` chunking.
        ngridx, ngridy: Output dimensions; default to ``n_det``.
        filter_name: Any filter supported by TomoPy Gridrec.
        filter_par: TomoPy filter parameters. Defaults to ``[0.5, 8]``.
        pair_chunk: Maximum number of packed two-slice grids processed at
            once. ``1`` bounds a 4096-square Fourier grid to about 128 MiB.
    """

    def __init__(
        self,
        n_det: int,
        theta,
        center,
        *,
        ngridx: int | None = None,
        ngridy: int | None = None,
        filter_name: str = 'shepp',
        filter_par=None,
        pair_chunk: int = 1,
    ):
        super().__init__()
        self.n_det = int(n_det)
        if self.n_det <= 0:
            raise ValueError('n_det must be positive')
        self.pdim = _next_power_of_two_at_least_detector(self.n_det)
        self.ngridx = self.n_det if ngridx is None else int(ngridx)
        self.ngridy = self.n_det if ngridy is None else int(ngridy)
        if not (0 < self.ngridx <= self.pdim):
            raise ValueError('ngridx must be in (0, pdim]')
        if not (0 < self.ngridy <= self.pdim):
            raise ValueError('ngridy must be in (0, pdim]')
        self.pair_chunk = int(pair_chunk)
        if self.pair_chunk <= 0:
            raise ValueError('pair_chunk must be positive')

        theta_cpu = _as_cpu_float32(theta, name='theta').flatten()
        self.n_ang = int(theta_cpu.numel())
        if center is None:
            center = self.n_det / 2.0
        center_cpu = _as_cpu_float32(center, name='center').flatten()
        self._center_count = int(center_cpu.numel())
        self._shared_center = bool(
            self._center_count == 1
            or torch.equal(
                center_cpu,
                center_cpu[0].expand_as(center_cpu),
            )
        )

        name = _canonical_filter_name(filter_name)
        if name not in _FILTER_NAMES:
            # The TomoPy C get_filter() silently falls back to Shepp. The
            # Python CUDA API is explicit to avoid accidental misspellings.
            raise ValueError(
                f'unknown Gridrec filter {filter_name!r}; '
                f'expected one of {sorted(_FILTER_NAMES)}'
            )
        self.filter_name = name
        if filter_par is None:
            filter_par = np.array([0.5, 8.0], dtype=np.float32)
        filter_par_cpu = _as_cpu_float32(
            filter_par, name='filter_par'
        ).flatten()

        extension = gridrec_ext()
        sine_cpu, cose_cpu = extension.prepare_trig_tables(theta_cpu)
        wtbl_cpu, winv_cpu = extension.prepare_pswf_tables(self.pdim)

        # The C routine packs two real slices into one complex FFT and uses
        # center[s] for that pair.  TomoPy's Python dispatcher usually splits
        # a small batch into one-slice chunks, however, so different centers
        # are supposed to remain per-slice at the public API.  Preserve the
        # fast packing only when the phase is genuinely shared.
        phase_centers = (
            center_cpu[:1] if self._shared_center else center_cpu
        ).contiguous()
        phase_float = extension.prepare_filter_phase(
            phase_centers,
            self.n_ang,
            self.pdim,
            self.filter_name,
            filter_par_cpu,
        )
        phase_complex = torch.view_as_complex(phase_float.contiguous())

        self.register_buffer('theta', theta_cpu)
        self.register_buffer('sine', sine_cpu)
        self.register_buffer('cose', cose_cpu)
        self.register_buffer('wtbl', wtbl_cpu)
        self.register_buffer('winv', winv_cpu)
        self.register_buffer('filter_phase', phase_complex)

        pdim2 = self.pdim // 2
        padx = (self.pdim - self.ngridx) // 2
        pady = (self.pdim - self.ngridy) // 2
        offsetx = pdim2 - padx
        offsety = pdim2 - pady

        output_rows = torch.arange(self.ngridx, dtype=torch.int64)
        output_cols = torch.arange(self.ngridy, dtype=torch.int64)
        k_for_row = self.ngridx - 1 - output_rows
        source_v = torch.remainder(
            k_for_row + self.pdim - offsetx, self.pdim
        )
        source_u = torch.remainder(
            output_cols + self.pdim - offsety, self.pdim
        )
        # gridrec.c allocates pdim-1 correction entries but its public API can
        # request pdim output pixels. That edge case reads one element past
        # ``winv`` and may produce zero or allocator-dependent garbage. Give
        # it a deterministic zero boundary instead of reproducing undefined
        # memory access.
        winv_lookup = torch.cat(
            [winv_cpu, torch.zeros(1, dtype=torch.float32)]
        )
        correction_rows = winv_lookup[k_for_row + padx]
        correction_cols = winv_lookup[output_cols + pady]
        correction = correction_rows[:, None] * correction_cols[None, :]
        self.register_buffer('source_u', source_u)
        self.register_buffer('source_v', source_v)
        self.register_buffer('correction', correction.contiguous())

    def _phase_for_pairs(
        self,
        pair_start: int,
        pair_stop: int,
    ) -> torch.Tensor:
        pair_count = pair_stop - pair_start
        return self.filter_phase.expand(
            pair_count, -1, -1
        ).contiguous()

    def _reconstruct_pair_chunk(
        self,
        real_slices: torch.Tensor,
        imaginary_slices: torch.Tensor,
        phase: torch.Tensor,
    ) -> torch.Tensor:
        import torch.nn.functional as functional

        pad_right = self.pdim - self.n_det
        real_padded = functional.pad(real_slices, (0, pad_right))
        imaginary_padded = functional.pad(imaginary_slices, (0, pad_right))
        projections = torch.complex(real_padded, imaginary_padded)
        # FFTW_BACKWARD is unnormalized; torch.ifft includes 1/pdim.
        projections = torch.fft.ifft(projections, dim=-1) * self.pdim

        pair_count = projections.shape[0]
        fourier_grid = torch.zeros(
            (pair_count, self.pdim, self.pdim),
            dtype=torch.complex64,
            device=projections.device,
        )
        gridrec_ext().gridding(
            projections.contiguous(),
            fourier_grid,
            phase,
            self.sine,
            self.cose,
            self.wtbl,
        )
        # FFTW_FORWARD 2-D transform, without normalization.
        spatial_grid = torch.fft.fft2(fourier_grid)

        selected = spatial_grid[
            :,
            self.source_u[None, :],
            self.source_v[:, None],
        ]
        corrected = selected * self.correction[None, :, :]
        return torch.stack(
            (corrected.real, corrected.imag), dim=1
        ).reshape(-1, self.ngridx, self.ngridy)

    @torch.no_grad()
    def forward(self, sinograms: torch.Tensor) -> torch.Tensor:
        """Reconstruct ``(batch, n_ang, n_det)`` float32 CUDA sinograms."""
        if not isinstance(sinograms, torch.Tensor):
            raise TypeError('sinograms must be a torch.Tensor')
        if not sinograms.is_cuda:
            raise RuntimeError('sinograms must be CUDA')
        if sinograms.dtype != torch.float32:
            raise TypeError('sinograms must be float32')
        if sinograms.ndim != 3:
            raise ValueError('sinograms must have shape (batch, n_ang, n_det)')
        if tuple(sinograms.shape[1:]) != (self.n_ang, self.n_det):
            raise ValueError(
                f'expected (*, {self.n_ang}, {self.n_det}); '
                f'got {tuple(sinograms.shape)}'
            )
        if not sinograms.is_contiguous():
            sinograms = sinograms.contiguous()
        if self.theta.device != sinograms.device:
            raise RuntimeError(
                'module buffers and sinograms must be on the same CUDA device; '
                'call model.to(sinograms.device) first'
            )

        batch_size = int(sinograms.shape[0])
        if self._center_count not in (1, batch_size):
            raise ValueError(
                f'center contains {self._center_count} values but input '
                f'batch contains {batch_size} slices'
            )

        # A shared phase permits TomoPy's real/imaginary two-slice packing.
        # With genuinely different per-slice centers, using one phase for a
        # pair would silently make the answer depend on the first slice.  Run
        # those slices as independent real components instead.
        if not self._shared_center:
            chunks = []
            for slice_start in range(0, batch_size, self.pair_chunk):
                slice_stop = min(
                    slice_start + self.pair_chunk, batch_size
                )
                real_slices = sinograms[slice_start:slice_stop]
                imaginary_slices = torch.zeros_like(real_slices)
                phase = self.filter_phase[
                    slice_start:slice_stop
                ].contiguous()
                reconstructed = self._reconstruct_pair_chunk(
                    real_slices,
                    imaginary_slices,
                    phase,
                )
                chunks.append(reconstructed[0::2])
            return torch.cat(chunks, dim=0)

        pair_count = (batch_size + 1) // 2
        chunks = []
        for pair_start in range(0, pair_count, self.pair_chunk):
            pair_stop = min(pair_start + self.pair_chunk, pair_count)
            slice_start = 2 * pair_start
            slice_stop = min(2 * pair_stop, batch_size)
            chunk = sinograms[slice_start:slice_stop]
            real_slices = chunk[0::2]
            imaginary_slices = torch.zeros_like(real_slices)
            available_imaginary = chunk[1::2]
            if available_imaginary.numel() != 0:
                imaginary_slices[:available_imaginary.shape[0]].copy_(
                    available_imaginary
                )
            phase = self._phase_for_pairs(
                pair_start, pair_stop
            )
            reconstructed = self._reconstruct_pair_chunk(
                real_slices,
                imaginary_slices,
                phase,
            )
            chunks.append(reconstructed[:slice_stop - slice_start])
        return torch.cat(chunks, dim=0)


GridrecCUDA = TomoPyGridrecCUDA


@torch.no_grad()
def gridrec_cuda(
    sinograms: torch.Tensor,
    theta,
    center,
    *,
    ngridx: int | None = None,
    ngridy: int | None = None,
    filter_name: str = 'shepp',
    filter_par=None,
    pair_chunk: int = 1,
) -> torch.Tensor:
    """One-shot functional interface equivalent to TomoPy Gridrec."""
    if sinograms.ndim != 3:
        raise ValueError('sinograms must have shape (batch, n_ang, n_det)')
    model = TomoPyGridrecCUDA(
        int(sinograms.shape[-1]),
        theta,
        center,
        ngridx=ngridx,
        ngridy=ngridy,
        filter_name=filter_name,
        filter_par=filter_par,
        pair_chunk=pair_chunk,
    ).to(sinograms.device)
    return model(sinograms)


__all__ = [
    'GridrecCUDA',
    'TomoPyGridrecCUDA',
    'gridrec_ext',
    'gridrec_cuda',
]
