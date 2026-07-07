"""Torch straight-ray USCT line-integral operator.

The operator is linear in slowness perturbation

    delta_s = 1 / c - 1 / c0

and maps `(B, C, H, W)` images to `(B, C, n_tx, n_rx)` travel-time delays.
Coordinates use `(y, x)` order in meters to match image arrays `[row=y, col=x]`.
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np
import torch

from ..base import LinearOperator


def centered_grid_origin(
    grid_shape: Tuple[int, int],
    spacing_m: Tuple[float, float],
) -> Tuple[float, float]:
    """Return an origin whose pixel-edge grid is centered around `(0, 0)`."""

    ny, nx = int(grid_shape[0]), int(grid_shape[1])
    dy, dx = float(spacing_m[0]), float(spacing_m[1])
    return (-0.5 * ny * dy, -0.5 * nx * dx)


def make_ring_geometry(
    n_transducers: int,
    radius_m: float,
    *,
    center_yx: Tuple[float, float] = (0.0, 0.0),
    start_angle_rad: float = 0.0,
) -> np.ndarray:
    """Return `(n, 2)` transducer positions on a circular ring in `(y, x)`."""

    if n_transducers <= 0:
        raise ValueError("n_transducers must be positive")
    if radius_m <= 0.0:
        raise ValueError("radius_m must be positive")
    angles = start_angle_rad + np.linspace(
        0.0, 2.0 * math.pi, int(n_transducers), endpoint=False
    )
    cy, cx = float(center_yx[0]), float(center_yx[1])
    return np.stack(
        [
            cy + float(radius_m) * np.sin(angles),
            cx + float(radius_m) * np.cos(angles),
        ],
        axis=1,
    )


class USCTStraightRayOperator(LinearOperator):
    """Sparse straight-ray projector with explicit adjoint.

    Parameters
    ----------
    grid_shape:
        Image grid `(ny, nx)`.
    spacing_m:
        Pixel spacing `(dy, dx)` in meters.
    tx_positions_yx, rx_positions_yx:
        Source and receiver coordinates in `(y, x)` meters.  If receivers are
        omitted, the transmitter ring is reused.
    origin_m:
        Pixel-edge origin `(y0, x0)`. Defaults to a centered grid.
    """

    def __init__(
        self,
        grid_shape: Tuple[int, int],
        spacing_m: Tuple[float, float],
        tx_positions_yx: Iterable[Iterable[float]],
        rx_positions_yx: Iterable[Iterable[float]] | None = None,
        *,
        origin_m: Tuple[float, float] | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        in_channels: int = 1,
    ) -> None:
        self.grid_shape = _validate_grid_shape(grid_shape)
        self.spacing_m = _validate_spacing(spacing_m)
        self.origin_m = (
            centered_grid_origin(self.grid_shape, self.spacing_m)
            if origin_m is None
            else (float(origin_m[0]), float(origin_m[1]))
        )
        self.tx_positions_yx = _validate_positions(tx_positions_yx, "tx_positions_yx")
        self.rx_positions_yx = (
            self.tx_positions_yx.copy()
            if rx_positions_yx is None
            else _validate_positions(rx_positions_yx, "rx_positions_yx")
        )
        self.ray_shape = (self.tx_positions_yx.shape[0], self.rx_positions_yx.shape[0])
        self.n_rays = int(self.ray_shape[0] * self.ray_shape[1])
        self.n_pixels = int(self.grid_shape[0] * self.grid_shape[1])
        self.domain_shape = (int(in_channels), *self.grid_shape)
        self.range_shape = (int(in_channels), *self.ray_shape)
        self.device = torch.device(device)
        self.dtype = dtype
        self.matrix = _build_sparse_matrix(
            self.grid_shape,
            self.spacing_m,
            self.origin_m,
            self.tx_positions_yx,
            self.rx_positions_yx,
            device=self.device,
            dtype=self.dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values, batch_shape, squeeze = self._flatten_image_batch(x)
        projected = torch.sparse.mm(self.matrix, values.T).T
        out = projected.reshape(*batch_shape, *self.ray_shape)
        return out[0, 0] if squeeze else out

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        values, batch_shape, squeeze = self._flatten_ray_batch(y)
        backprojected = torch.sparse.mm(self.matrix.transpose(0, 1), values.T).T
        out = backprojected.reshape(*batch_shape, *self.grid_shape)
        return out[0, 0] if squeeze else out

    def forward_flat(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the projector and flatten the ray dimensions."""

        return self.forward(x).reshape(*torch.as_tensor(x).shape[:-2], self.n_rays)

    def adjoint_flat(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the adjoint to flattened ray values."""

        return self.adjoint(y)

    def row_norms(self, power: int = 2) -> torch.Tensor:
        matrix = self._matrix_with_power(power)
        ones = torch.ones(self.n_pixels, dtype=self.dtype, device=self.device)
        return torch.sparse.mm(matrix, ones[:, None]).reshape(self.ray_shape)

    def col_norms(self, power: int = 2) -> torch.Tensor:
        matrix = self._matrix_with_power(power)
        ones = torch.ones(self.n_rays, dtype=self.dtype, device=self.device)
        return torch.sparse.mm(matrix.transpose(0, 1), ones[:, None]).reshape(
            self.grid_shape
        )

    def _matrix_with_power(self, power: int) -> torch.Tensor:
        matrix = self.matrix.coalesce()
        return torch.sparse_coo_tensor(
            matrix.indices(),
            matrix.values().pow(int(power)),
            size=matrix.shape,
            dtype=self.dtype,
            device=self.device,
        ).coalesce()

    def _flatten_image_batch(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, ...], bool]:
        tensor = torch.as_tensor(image, dtype=self.dtype, device=self.device)
        if tuple(tensor.shape[-2:]) != self.grid_shape:
            raise ValueError(
                f"image trailing shape must be {self.grid_shape}, got {tensor.shape}"
            )
        squeeze = tensor.ndim == 2
        if squeeze:
            tensor = tensor.reshape(1, 1, *self.grid_shape)
        batch_shape = tuple(tensor.shape[:-2])
        return tensor.reshape(-1, self.n_pixels), batch_shape, squeeze

    def _flatten_ray_batch(
        self, ray_values: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, ...], bool]:
        tensor = torch.as_tensor(ray_values, dtype=self.dtype, device=self.device)
        squeeze = False
        if tuple(tensor.shape[-2:]) == self.ray_shape:
            squeeze = tensor.ndim == 2
            if squeeze:
                tensor = tensor.reshape(1, 1, *self.ray_shape)
            batch_shape = tuple(tensor.shape[:-2])
            return tensor.reshape(-1, self.n_rays), batch_shape, squeeze
        if tensor.shape[-1] != self.n_rays:
            raise ValueError(
                f"ray_values trailing shape must be {self.ray_shape} or "
                f"length {self.n_rays}, got {tensor.shape}"
            )
        squeeze = tensor.ndim == 1
        if squeeze:
            tensor = tensor.reshape(1, 1, self.n_rays)
        batch_shape = tuple(tensor.shape[:-1])
        return tensor.reshape(-1, self.n_rays), batch_shape, squeeze


def _build_sparse_matrix(
    grid_shape: Tuple[int, int],
    spacing_m: Tuple[float, float],
    origin_m: Tuple[float, float],
    tx_positions_yx: np.ndarray,
    rx_positions_yx: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    values: list[np.ndarray] = []
    ray_id = 0
    for tx in tx_positions_yx:
        for rx in rx_positions_yx:
            indices, lengths = _trace_ray(grid_shape, spacing_m, origin_m, tx, rx)
            if indices.size:
                rows.append(np.full(indices.size, ray_id, dtype=np.int64))
                cols.append(indices.astype(np.int64, copy=False))
                values.append(lengths.astype(float, copy=False))
            ray_id += 1
    if rows:
        index_array = np.stack([np.concatenate(rows), np.concatenate(cols)], axis=0)
        value_array = np.concatenate(values)
    else:
        index_array = np.empty((2, 0), dtype=np.int64)
        value_array = np.empty(0, dtype=float)
    indices_tensor = torch.as_tensor(index_array, dtype=torch.long, device=device)
    values_tensor = torch.as_tensor(value_array, dtype=dtype, device=device)
    return torch.sparse_coo_tensor(
        indices_tensor,
        values_tensor,
        size=(ray_id, grid_shape[0] * grid_shape[1]),
        dtype=dtype,
        device=device,
    ).coalesce()


def _trace_ray(
    grid_shape: Tuple[int, int],
    spacing_m: Tuple[float, float],
    origin_m: Tuple[float, float],
    tx_yx: np.ndarray,
    rx_yx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y0, x0 = float(tx_yx[0]), float(tx_yx[1])
    y1, x1 = float(rx_yx[0]), float(rx_yx[1])
    dy_total = y1 - y0
    dx_total = x1 - x0
    segment_length = float(np.hypot(dy_total, dx_total))
    if segment_length == 0.0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=float)

    ny, nx = int(grid_shape[0]), int(grid_shape[1])
    dy, dx = float(spacing_m[0]), float(spacing_m[1])
    origin_y, origin_x = float(origin_m[0]), float(origin_m[1])
    y_edges = origin_y + np.arange(ny + 1) * dy
    x_edges = origin_x + np.arange(nx + 1) * dx

    t_values = [0.0, 1.0]
    if dy_total != 0.0:
        t_values.extend(float((edge - y0) / dy_total) for edge in y_edges)
    if dx_total != 0.0:
        t_values.extend(float((edge - x0) / dx_total) for edge in x_edges)
    t = np.array([value for value in t_values if 0.0 <= value <= 1.0], dtype=float)
    if t.size < 2:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=float)
    t = np.unique(np.round(t, decimals=15))

    indices: list[int] = []
    lengths: list[float] = []
    for start, end in zip(t[:-1], t[1:]):
        if end <= start:
            continue
        mid = 0.5 * (start + end)
        y_mid = y0 + mid * dy_total
        x_mid = x0 + mid * dx_total
        iy = int(np.floor((y_mid - origin_y) / dy))
        ix = int(np.floor((x_mid - origin_x) / dx))
        if 0 <= iy < ny and 0 <= ix < nx:
            indices.append(iy * nx + ix)
            lengths.append(segment_length * (end - start))
    if not indices:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=float)
    return np.asarray(indices, dtype=np.int64), np.asarray(lengths, dtype=float)


def _validate_grid_shape(grid_shape: Tuple[int, int]) -> Tuple[int, int]:
    ny, nx = int(grid_shape[0]), int(grid_shape[1])
    if ny <= 0 or nx <= 0:
        raise ValueError("grid_shape must contain positive integers")
    return ny, nx


def _validate_spacing(spacing_m: Tuple[float, float]) -> Tuple[float, float]:
    dy, dx = float(spacing_m[0]), float(spacing_m[1])
    if dy <= 0.0 or dx <= 0.0:
        raise ValueError("spacing_m must contain positive values")
    return dy, dx


def _validate_positions(values: Iterable[Iterable[float]], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] == 0:
        raise ValueError(f"{name} must have shape (n, 2)")
    return array
