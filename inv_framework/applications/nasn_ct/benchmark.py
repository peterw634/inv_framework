"""Held-out synthetic interior-CT benchmark for the public CUDA operator."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import tifffile
import torch
from skimage.metrics import structural_similarity

from ...operators.ct import TomoPyCUDARadon2D
from ...solvers.diffusion import DDSSolver, PMFVelocityPrior, CGDataConsistency
from .random import BatchGenerator
from .weights import load_pmf


def _roi_metrics(reference, estimate, radius=256):
    reference = reference.astype(np.float32)
    estimate = estimate.astype(np.float32)
    height, width = reference.shape
    yy, xx = np.ogrid[:height, :width]
    mask = (yy - (height - 1) / 2) ** 2 + (xx - (width - 1) / 2) ** 2 <= radius ** 2
    data_range = float(reference.max() - reference.min())
    mse = float(np.mean((reference[mask] - estimate[mask]) ** 2))
    psnr = float("inf") if mse == 0 else 10 * np.log10(data_range ** 2 / mse)
    cy, cx, radius = height // 2, width // 2, int(radius)
    ssim = structural_similarity(
        reference[cy-radius:cy+radius, cx-radius:cx+radius],
        estimate[cy-radius:cy+radius, cx-radius:cx+radius],
        data_range=data_range,
    )
    return float(psnr), float(ssim)


def benchmark_synthetic(weights, val_root, output, model_config=None,
                        slice_ids=range(991, 1001), steps=(16, 32),
                        dc_lam=5.0, dc_iter=10, seed=0, device="cuda"):
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("synthetic 2048 benchmark requires CUDA")
    val_root, output = Path(val_root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    slice_ids, steps = tuple(map(int, slice_ids)), tuple(map(int, steps))
    model = load_pmf(weights, model_config, device)
    forward = TomoPyCUDARadon2D(2048, 2048, 512, 1024.0, device=device)
    report = {
        "status": "inv_framework migration; synthetic CUDA benchmark", "checkpoint": "epoch 449 EMA",
        "configuration": {"image_size": 2048, "angles": 2048, "detectors": 2048,
                          "measured_detectors": 512, "roi_radius": 256,
                          "dc_lambda": dc_lam, "cg_iterations": dc_iter,
                          "seed": seed, "operator": "inv_framework TomoPyCUDARadon2D"},
        "per_slice": {}, "aggregate": {},
    }
    ground_truth = np.stack([
        tifffile.imread(val_root / f"{slice_id:06d}.tif").astype(np.float32)
        for slice_id in slice_ids
    ])
    x = torch.from_numpy(ground_truth)[:, None].to(device)
    target = forward((x + 1) / 2)
    for slice_id in slice_ids:
        report["per_slice"][str(slice_id)] = {}
    for count in steps:
        started = time.time()
        conditioning = CGDataConsistency(
            forward, lam=dc_lam, cg_iter=dc_iter,
            image_scale=0.5, image_offset=0.5,
        )
        solver = DDSSolver(
            PMFVelocityPrior(model), num_inference_steps=count,
            conditioning=conditioning,
        )
        noise = BatchGenerator(device, [seed] * len(slice_ids)).randn(tuple(x.shape))
        result = solver.solve(target, forward, initial_noise=noise)
        torch.cuda.synchronize(device)
        elapsed = time.time() - started
        images = result[:, 0].float().cpu().numpy()
        projected = forward((result + 1) / 2)
        residuals = torch.linalg.vector_norm(
            (projected - target).float().flatten(1), dim=1
        ) / torch.linalg.vector_norm(target.float().flatten(1), dim=1).clamp_min(1e-30)
        for index, slice_id in enumerate(slice_ids):
            psnr, ssim = _roi_metrics(ground_truth[index], images[index])
            report["per_slice"][str(slice_id)][str(count)] = {
                "roi_psnr": psnr, "roi_ssim": ssim,
                "relative_measurement_l2": float(residuals[index]),
                "seconds_per_batch": elapsed,
            }
        (output / "progress.json").write_text(json.dumps(report, indent=2))
    for count in steps:
        rows = [report["per_slice"][str(index)][str(count)] for index in slice_ids]
        report["aggregate"][str(count)] = {
            "roi_psnr_mean": float(np.mean([row["roi_psnr"] for row in rows])),
            "roi_psnr_std": float(np.std([row["roi_psnr"] for row in rows])),
            "roi_ssim_mean": float(np.mean([row["roi_ssim"] for row in rows])),
            "relative_measurement_l2_mean": float(np.mean([
                row["relative_measurement_l2"] for row in rows
            ])),
        }
    gates = {16: (35, 0.80), 32: (37, 0.84)}
    report["acceptance"] = {
        str(count): (report["aggregate"][str(count)]["roi_psnr_mean"] >= gates[count][0]
                     and report["aggregate"][str(count)]["roi_ssim_mean"] >= gates[count][1])
        for count in steps if count in gates
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2))
    return report
