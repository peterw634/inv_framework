"""TIFF dataset for the linearly scaled FullCT training corpus."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset


class CTSliceDataset(Dataset):
    def __init__(self, root, split="train", augment=False):
        self.directory = Path(root) / split
        self.files = sorted(self.directory.glob("*.tif"))
        if not self.files:
            raise FileNotFoundError(f"no TIFF slices in {self.directory}")
        self.augment = bool(augment)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = torch.from_numpy(
            tifffile.imread(self.files[index]).astype(np.float32)
        ).unsqueeze(0)
        if self.augment:
            if random.random() < 0.5:
                image = image.flip(-1)
            if random.random() < 0.5:
                image = image.flip(-2)
            image = torch.rot90(image, random.randint(0, 3), (-2, -1))
        return image, 0
