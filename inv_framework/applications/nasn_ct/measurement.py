"""Published NaSn measurement geometry and its training-scale conversion."""

from pathlib import Path
import math

import cv2
import numpy as np
import tifffile
import torch

from ...operators.ct import TomoPyCUDARadon2D

SOURCE_DETECTORS = 2560
MEASURED_DETECTORS = 410
IMAGE_SIZE = 2048
SOURCE_CENTER = 1172.0
PAD_LEFT = (IMAGE_SIZE - MEASURED_DETECTORS) // 2
PADDED_CENTER = PAD_LEFT + (SOURCE_CENTER + 0.5) * MEASURED_DETECTORS / SOURCE_DETECTORS - 0.5


def downsample_and_pad(sinogram, scale=1.0):
    sinogram = np.asarray(sinogram, dtype=np.float32)
    if sinogram.ndim != 2 or sinogram.shape[1] != SOURCE_DETECTORS:
        raise ValueError(f"expected (angles,{SOURCE_DETECTORS}); got {sinogram.shape}")
    if not math.isfinite(float(scale)) or float(scale) <= 0:
        raise ValueError("scale must be finite and positive")
    small = cv2.resize(sinogram, (MEASURED_DETECTORS, sinogram.shape[0]), interpolation=cv2.INTER_AREA)
    result = np.zeros((sinogram.shape[0], IMAGE_SIZE), dtype=np.float32)
    result[:, PAD_LEFT:PAD_LEFT + MEASURED_DETECTORS] = small * float(scale)
    return result


class TruncatedMeasurement:
    def __init__(self, sinogram_dir, scale, center=PADDED_CENTER):
        self.sinogram_dir = Path(sinogram_dir)
        self.scale = float(scale)
        self.center = float(center)

    def load(self, slice_label, device="cuda", dtype=torch.float32):
        path = self.sinogram_dir / f"sino_{int(slice_label):06d}.tif"
        value = downsample_and_pad(tifffile.imread(path), self.scale)
        if value.shape[0] != 900:
            raise ValueError(f"NaSn release geometry requires 900 views, got {value.shape[0]}")
        return torch.from_numpy(value)[None, None].to(device=device, dtype=dtype)

    def operator(self, device="cuda"):
        return TomoPyCUDARadon2D(IMAGE_SIZE, 900, MEASURED_DETECTORS, self.center, device=device)


def load_scale_stats(path):
    """Reject the legacy scale computed at a different rotation center."""
    import json
    value = json.loads(Path(path).read_text())
    if float(value.get("center_2560", float("nan"))) != SOURCE_CENTER:
        raise ValueError("scale statistics must be calibrated at source center 1172.0")
    scale = float(value["final_scale"])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("final_scale must be finite and positive")
    return scale
