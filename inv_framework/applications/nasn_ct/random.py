"""Per-sample deterministic random-number generator used by pMF sampling."""
from __future__ import annotations

import torch


class BatchGenerator:
    def __init__(self, device, seeds):
        self.device = torch.device(device)
        self.generators = [
            torch.Generator("cpu").manual_seed(int(seed) % (1 << 32))
            for seed in seeds
        ]

    def randn(self, shape, **kwargs):
        if shape[0] != len(self.generators):
            raise ValueError("one seed is required per batch item")
        kwargs.pop("device", None)
        return torch.stack([
            torch.randn(shape[1:], generator=generator, **kwargs).to(self.device)
            for generator in self.generators
        ])

    def randn_like(self, value):
        return self.randn(value.shape, dtype=value.dtype, layout=value.layout)
