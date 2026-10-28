"""Real NaSn reconstruction using the framework's DDS/CG abstractions."""

import json
import time
from pathlib import Path

import numpy as np
import tifffile
import torch

from ...solvers.diffusion import DDSSolver, PMFVelocityPrior
from .measurement import PAD_LEFT, MEASURED_DETECTORS, IMAGE_SIZE, TruncatedMeasurement
from .random import BatchGenerator
from .weights import load_pmf


def reconstruct_slices(weights, sinogram_dir, scale, output, slices,
                       model_config=None, num_steps=16, dc_lam=0.05,
                       dc_iter=10, seed=0, omega=1.0, t_start=1.0,
                       warm_start=False, device="cuda"):
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the released 2048 DDS path requires a CUDA GPU")
    if warm_start and not 0 < t_start < 1:
        raise ValueError("warm_start requires 0 < t_start < 1")
    model = load_pmf(weights, model_config, device=device)
    if model.img_size != IMAGE_SIZE or model.img_channels != 1:
        raise ValueError("the NaSn release requires a single-channel 2048 model")
    measurement = TruncatedMeasurement(sinogram_dir, scale)
    operator = measurement.operator(device)
    prior = PMFVelocityPrior(model, omega=omega)
    solver = DDSSolver(prior, num_inference_steps=num_steps,
                       dc_lam=dc_lam, dc_iter=dc_iter, t_start=t_start)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "release_status": "research preview; real target data have no ground truth",
        "configuration": {
            "weights": str(Path(weights).resolve()), "scale": float(scale),
            "num_steps": int(num_steps), "dc_lam": float(dc_lam),
            "dc_iter": int(dc_iter), "seed": int(seed), "omega": float(omega),
            "t_start": float(t_start), "warm_start": bool(warm_start),
            "back_operator": "strict adjoint", "solver": "inv_framework.DDSSolver",
        }, "slices": {},
    }
    for label in map(int, slices):
        target = measurement.load(label, device=device, dtype=model.dtype)
        initial = operator.pseudo_inverse(target) if warm_start else None
        # The reference draws one independent CPU RNG stream per sample.
        noise = BatchGenerator(device, [seed]).randn((1, *operator.domain_shape))
        torch.cuda.synchronize(device)
        started = time.time()
        result = solver.solve(target, operator, x_init=initial, initial_noise=noise)
        torch.cuda.synchronize(device)
        elapsed = time.time() - started
        with torch.no_grad():
            projected = operator(result)
        band = slice(PAD_LEFT, PAD_LEFT + MEASURED_DETECTORS)
        residual = (
            torch.linalg.vector_norm((projected[..., band] - target[..., band]).float())
            / torch.linalg.vector_norm(target[..., band].float()).clamp_min(1e-30)
        ).item()
        image = result[0, 0].float().cpu().numpy()
        tifffile.imwrite(output / f"recon_slice_{label:06d}.tif", image)
        report["slices"][str(label)] = {
            "measured_band_relative_l2": residual, "seconds": elapsed,
            "p005": float(np.percentile(image, 0.5)),
            "p995": float(np.percentile(image, 99.5)),
            "minimum": float(image.min()), "maximum": float(image.max()),
        }
        (output / "summary.json").write_text(json.dumps(report, indent=2))
    return report
