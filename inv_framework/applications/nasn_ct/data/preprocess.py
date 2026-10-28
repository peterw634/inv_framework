"""Portable implementation of the legacy PAD-PA + Rivers data pipeline."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import tifffile

from ..config import inclusive_range


def discover(directory: Path, prefix: str, extension=".tif") -> list[Path]:
    return [
        path for path in sorted(directory.iterdir())
        if path.suffix.lower() == extension.lower()
        and path.name.lower().startswith(prefix.lower())
        and ".rec" not in path.name.lower()
    ]


def mean_images(paths: list[Path]) -> np.ndarray | None:
    total = None
    for path in paths:
        image = tifffile.imread(path).astype(np.float32)
        if total is None:
            total = np.zeros_like(image, dtype=np.float64)
        total += image
    return None if total is None else (total / len(paths)).astype(np.float32)


def flat_correct(projection, flat, dark=None, dark_value=0.0):
    projection = np.asarray(projection, dtype=np.float32)
    flat = np.asarray(flat, dtype=np.float32)
    if dark is None:
        dark = np.float32(dark_value)
    else:
        dark = np.asarray(dark, dtype=np.float32)
    numerator = projection - dark
    denominator = flat - dark
    numerator_fill = float((projection.mean(dtype=np.float64) - np.mean(dark)) / 1000)
    denominator_fill = float((flat.mean(dtype=np.float64) - np.mean(dark)) / 1000)
    numerator = np.where(numerator <= 0, numerator_fill, numerator)
    denominator = np.where(denominator <= 0, denominator_fill, denominator)
    return (numerator / denominator).astype(np.float32)


def pad_pa_batch(corrected, pixel_size_m, distance_m, energy_kev, epsilon,
                 device="cpu", clip_min=1e-6):
    """Apply the historical propagation-based phase retrieval formula."""
    import torch
    import torch.nn.functional as functional

    corrected = np.asarray(corrected, dtype=np.float64)
    _, height, width = corrected.shape
    padded_h = 2 ** int(np.ceil(np.log2(height)))
    padded_w = 2 ** int(np.ceil(np.log2(width)))
    top = (padded_h - height) // 2
    bottom = padded_h - height - top
    left = (padded_w - width) // 2
    right = padded_w - width - left
    with torch.no_grad():
        batch = torch.as_tensor(corrected, dtype=torch.float64, device=device)
        batch = batch.clamp_min(clip_min)
        fy = torch.fft.fftfreq(padded_h, d=pixel_size_m, device=device, dtype=torch.float64)
        fx = torch.fft.fftfreq(padded_w, d=pixel_size_m, device=device, dtype=torch.float64)
        frequency2 = (2 * torch.pi * fy)[:, None] ** 2 + (2 * torch.pi * fx)[None, :] ** 2
        wavelength = 12.39841984e-10 / float(energy_kev)
        denominator = 1 + epsilon * wavelength * distance_m / (4 * torch.pi) * frequency2
        padded = functional.pad(batch, (left, right, top, bottom), value=0)
        retrieved = torch.abs(torch.fft.ifft2(torch.fft.fft2(padded) / denominator))
        retrieved = retrieved[:, top:top + height, left:left + width]
        retrieved = -0.5 * torch.log(retrieved)
        retrieved *= 2 * wavelength / (4 * np.pi * pixel_size_m) * epsilon
        return retrieved.cpu().numpy()


def smooth_edge_mean(values, width):
    width = int(width)
    left, right = width // 2, width - 1 - width // 2
    padded = np.pad(values, (left, right), mode="edge").astype(np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(padded)))
    return ((cumulative[width:] - cumulative[:-width]) / width).astype(np.float32)


def rivers_1d(sinogram, width=5):
    sinogram = np.asarray(sinogram, dtype=np.float32)
    detector_mean = sinogram.mean(axis=0)
    stripe = detector_mean - smooth_edge_mean(detector_mean, int(width))
    return (sinogram - stripe[None, :]).astype(np.float32)


def _paths(raw_root: Path, config: dict):
    projection_dir = raw_root / config.get("projection_dir", "Projection")
    if not projection_dir.is_dir():
        raise FileNotFoundError(f"projection directory not found: {projection_dir}")
    prefixes = config["prefixes"]
    tomos = discover(projection_dir, prefixes["tomo"])
    flats = discover(projection_dir, prefixes["flat"])
    darks = discover(projection_dir, prefixes.get("dark", "dark_"))
    selected = config.get("tomo_range_1based", [1, len(tomos)])
    tomos = tomos[selected[0] - 1:selected[1]]
    if not tomos or not flats:
        raise FileNotFoundError("both tomography projections and flats are required")
    return tomos, flats, darks


def prepare_sinograms(raw_root, output, config, slice_range=None,
                      device=None, batch_projections=2):
    """Create per-slice Rivers-corrected sinograms using bounded RAM.

    A resumable float32 memmap is filled projection-by-projection. Slice labels
    are 1-based and map to detector rows with ``row = label - 1``.
    """
    import torch

    raw_root, output = Path(raw_root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sino_dir = output / "sinograms_rivers"
    work_dir = output / ".nasn-work"
    sino_dir.mkdir(exist_ok=True)
    work_dir.mkdir(exist_ok=True)
    tomos, flats, darks = _paths(raw_root, config)
    labels = list(inclusive_range(slice_range or config["slice_range_1based"]))
    columns = config["detector_columns_1based"]
    col_start, col_stop = int(columns[0]) - 1, int(columns[1])
    detector_count = col_stop - col_start
    state_path = work_dir / "sinogram-state.json"
    memmap_path = work_dir / "sinograms.float32.dat"
    expected = {
        "shape": [len(labels), len(tomos), detector_count],
        "labels": labels,
        "completed_views": 0,
    }
    if state_path.exists() and memmap_path.exists():
        state = json.loads(state_path.read_text())
        if state["shape"] != expected["shape"] or state["labels"] != labels:
            raise RuntimeError(
                f"stale work files in {work_dir}; move them aside before changing the range"
            )
        mode = "r+"
    else:
        state = expected
        mode = "w+"
    buffer = np.memmap(memmap_path, mode=mode, dtype=np.float32, shape=expected["shape"])
    flat = mean_images(flats)
    dark = mean_images(darks)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    physics = config["phase_retrieval"]
    start = int(state["completed_views"])
    for view_start in range(start, len(tomos), int(batch_projections)):
        paths = tomos[view_start:view_start + int(batch_projections)]
        corrected = np.stack([
            flat_correct(tifffile.imread(path), flat, dark) for path in paths
        ])
        retrieved = pad_pa_batch(
            corrected,
            float(physics["pixel_size_m"]), float(physics["distance_m"]),
            float(physics["energy_kev"]), float(physics["epsilon"]), device=device,
        )
        rows = np.asarray(labels, dtype=np.int64) - 1
        if rows.min() < 0 or rows.max() >= retrieved.shape[1]:
            raise ValueError("slice range is outside the projection height")
        stop = view_start + len(paths)
        buffer[:, view_start:stop, :] = retrieved[:, rows, col_start:col_stop].transpose(1, 0, 2)
        buffer.flush()
        state["completed_views"] = stop
        state_path.write_text(json.dumps(state, indent=2))
    for index, label in enumerate(labels):
        destination = sino_dir / f"sino_{label:06d}.tif"
        if not destination.exists():
            tifffile.imwrite(destination, rivers_1d(buffer[index], config.get("rivers_width", 5)))
    provenance = {
        "raw_root": str(raw_root.resolve()),
        "configuration": {key: value for key, value in config.items() if key != "_config_path"},
        "sinograms": len(labels),
        "shape": expected["shape"],
        "device": device,
    }
    (output / "preprocessing.json").write_text(json.dumps(provenance, indent=2))
    return sino_dir


def circular_mask(images, ratio=0.95):
    height, width = images.shape[-2:]
    yy, xx = np.ogrid[:height, :width]
    center_y, center_x = (height - 1) / 2, (width - 1) / 2
    radius = min(height, width) * float(ratio) / 2
    mask = (yy - center_y) ** 2 + (xx - center_x) ** 2 <= radius ** 2
    return images * mask


def reconstruct_fullct(sino_dir, output, config, device="cuda"):
    """Reconstruct prepared FullCT sinograms using the public CUDA Gridrec."""
    import torch
    from ....operators.ct.radon_cuda import TomoPyGridrecCUDA

    if not torch.cuda.is_available():
        raise RuntimeError("FullCT Gridrec reconstruction requires a CUDA device")
    sino_dir, output = Path(sino_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files = sorted(sino_dir.glob("sino_*.tif"))
    if not files:
        raise FileNotFoundError(f"no sinograms in {sino_dir}")
    first = tifffile.imread(files[0])
    theta = np.linspace(0, np.pi, first.shape[0], endpoint=False, dtype=np.float32)
    model = TomoPyGridrecCUDA(
        first.shape[1], theta, float(config["center"]),
        filter_name=config.get("filter_name", "shepp"), pair_chunk=1,
    ).to(device)
    for path in files:
        destination = output / path.name.replace("sino_", "slice_")
        if destination.exists():
            continue
        sinogram = torch.from_numpy(tifffile.imread(path).astype(np.float32))[None].to(device)
        image = model(sinogram)[0].cpu().numpy()
        image = circular_mask(image, config.get("circular_mask_ratio", 0.95))
        tifffile.imwrite(destination, image.astype(np.float32))
    return output


def estimate_scale(reconstruction_dirs, samples_per_scan=80, multiplier=1.5):
    lows, highs = [], []
    for directory in map(Path, reconstruction_dirs):
        files = sorted(directory.glob("slice_*.tif"))
        if not files:
            raise FileNotFoundError(f"no reconstructions in {directory}")
        indexes = np.linspace(0, len(files) - 1, min(samples_per_scan, len(files)), dtype=int)
        for index in indexes:
            low, high = np.percentile(tifffile.imread(files[index]), [0.5, 99.5])
            lows.append(float(low))
            highs.append(float(high))
    low, high = float(np.median(lows)), float(np.median(highs))
    return {"lo_p005": low, "hi_p995": high, "scale_mul": multiplier,
            "final_scale": 2.0 / (high - low) * multiplier,
            "normalization": "image_norm = image_physical * final_scale (no centering)"}


def calibrate_target_scale(sino_dir, output, center=1172.0, samples=80,
                           multiplier=1.5, device="cuda"):
    """Recompute target scaling using public Gridrec at the corrected center."""
    import torch
    from ....operators.ct.radon_cuda import TomoPyGridrecCUDA

    if not torch.cuda.is_available():
        raise RuntimeError("target scale calibration requires a CUDA device")
    files = sorted(Path(sino_dir).glob("sino_*.tif"))
    if not files:
        raise FileNotFoundError(f"no sinograms in {sino_dir}")
    indexes = np.linspace(0, len(files) - 1, min(int(samples), len(files)), dtype=int)
    first = tifffile.imread(files[indexes[0]])
    theta = np.linspace(0, np.pi, first.shape[0], endpoint=False, dtype=np.float32)
    model = TomoPyGridrecCUDA(
        first.shape[1], theta, float(center), filter_name="shepp", pair_chunk=1
    ).to(device)
    lows, highs = [], []
    for index in indexes:
        sinogram = torch.from_numpy(tifffile.imread(files[index]).astype(np.float32))[None].to(device)
        image = model(sinogram)[0].cpu().numpy()
        image = circular_mask(image, 0.95)
        low, high = np.percentile(image, [0.5, 99.5])
        lows.append(float(low)); highs.append(float(high))
    low, high = float(np.median(lows)), float(np.median(highs))
    result = {
        "source": "prepared target sinograms", "center_2560": float(center),
        "n_samples": len(indexes), "lo_p005": low, "hi_p995": high,
        "scale_mul": float(multiplier),
        "final_scale": 2 / (high - low) * float(multiplier),
        "operator": "nasn_ct CUDA TomoPy-compatible Gridrec; shepp; circular mask 0.95",
    }
    Path(output).write_text(json.dumps(result, indent=2))
    return result


def build_corpus(train_recon_dirs, val_recon_dirs, output, size=2048,
                 samples_per_scan=80, scale_multiplier=1.5):
    """Build the exact train/val split and its leakage-free linear scale."""
    output = Path(output)
    (output / "train").mkdir(parents=True, exist_ok=True)
    (output / "val").mkdir(parents=True, exist_ok=True)
    stats = estimate_scale(train_recon_dirs, samples_per_scan, scale_multiplier)
    (output / "scale_stats.json").write_text(json.dumps(stats, indent=2))
    rows = []
    for split, directories in (("train", train_recon_dirs), ("val", val_recon_dirs)):
        global_id = 0
        for directory in map(Path, directories):
            for source in sorted(directory.glob("slice_*.tif")):
                image = tifffile.imread(source).astype(np.float32) * stats["final_scale"]
                image = cv2.resize(image, (int(size), int(size)), interpolation=cv2.INTER_CUBIC)
                destination = output / split / f"{global_id:06d}.tif"
                if not destination.exists():
                    tifffile.imwrite(destination, image.astype(np.float32))
                rows.append({"global_id": global_id, "split": split,
                             "source_scan": source.parent.parent.name,
                             "source_slice": source.name})
                global_id += 1
    with (output / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return stats
