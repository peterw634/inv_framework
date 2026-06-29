"""Source-subset operators for budgeted inverse-problem acquisition.

Many computational inverse problems in this repo have a source/shot budget:
USCT chooses transmitters on an array, and FTTT chooses seismic sources.
The physical forward model is still a ``ForwardOperator``, but solvers need a
standard way to ask for "the same operator restricted to these sources".
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Callable, Optional, Sequence, Tuple

import torch

from .base import ForwardOperator


class SubsettableForwardOperator(ForwardOperator):
    """Forward operator with a selectable source/shot axis.

    ``source_axis`` is the axis in the *batched* measurement tensor, so the
    default ``1`` means measurements are shaped like ``(B, sources, ...)``.
    """

    num_sources: int
    source_axis: int = 1

    @abstractmethod
    def subset(self, source_indices: Sequence[int]) -> ForwardOperator:
        """Return an operator that evaluates only the selected sources."""

    def subset_measurement(self, measurement: torch.Tensor,
                           source_indices: Sequence[int]) -> torch.Tensor:
        """Slice a batched measurement tensor to match ``subset(indices)``."""
        axis = self.source_axis
        if axis < 0:
            axis = measurement.ndim + axis
        index = torch.as_tensor(
            list(source_indices), dtype=torch.long, device=measurement.device)
        return torch.index_select(measurement, axis, index)


class CallableSubsetForwardOperator(SubsettableForwardOperator):
    """Small adapter for externally implemented source-subset physics.

    Parameters
    ----------
    forward_fn:
        Callable with signature ``forward_fn(x, source_indices)``. It can wrap a
        differentiable Torch implementation, or a heavier domain adapter such as
        CBS/Helmholtz and eikonal backends.
    domain_shape:
        Per-sample shape of the unknown, e.g. ``(1, 240, 240)``.
    range_shape:
        Per-sample full-measurement shape. For the default ``source_axis=1`` in
        batched tensors, the source dimension is ``range_shape[0]``.
    """

    def __init__(
        self,
        forward_fn: Callable[[torch.Tensor, Tuple[int, ...]], torch.Tensor],
        domain_shape: Tuple[int, ...],
        range_shape: Tuple[int, ...],
        num_sources: int,
        source_axis: int = 1,
        source_indices: Optional[Sequence[int]] = None,
        name: str = "callable_subset_operator",
    ):
        self._forward_fn = forward_fn
        self.domain_shape = tuple(domain_shape)
        self._full_range_shape = tuple(range_shape)
        self.num_sources = int(num_sources)
        self.source_axis = int(source_axis)
        if source_indices is None:
            self.source_indices = tuple(range(self.num_sources))
        else:
            self.source_indices = tuple(int(i) for i in source_indices)
        self.name = name
        self.range_shape = self._range_shape_for(self.source_indices)

    def _range_shape_for(self, indices: Sequence[int]) -> Tuple[int, ...]:
        shape = list(self._full_range_shape)
        # Convert batched axis to per-sample range axis.
        range_axis = self.source_axis - 1 if self.source_axis > 0 else self.source_axis
        if range_axis < 0:
            range_axis = len(shape) + range_axis
        if 0 <= range_axis < len(shape):
            shape[range_axis] = len(indices)
        return tuple(shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_fn(x, self.source_indices)

    def subset(self, source_indices: Sequence[int]) -> "CallableSubsetForwardOperator":
        return CallableSubsetForwardOperator(
            forward_fn=self._forward_fn,
            domain_shape=self.domain_shape,
            range_shape=self._full_range_shape,
            num_sources=self.num_sources,
            source_axis=self.source_axis,
            source_indices=source_indices,
            name=self.name,
        )
