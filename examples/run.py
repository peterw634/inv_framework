"""Run one registered algorithm on an .npz measurement pack.

    python examples/run.py --list
    python examples/run.py --algo sirt --input examples/data/toy_parallel_ct_128.npz --output outputs/sirt
    python examples/run.py --algo ddim_dds --input examples/data/toy_lodoind_512.npz \
        --weights weights/lodoind_pixel_diffusion --device cuda --output outputs/ddim_dds
    python examples/run.py --algo pmf_dds --input examples/data/toy_nasn_pmf_2048.npz \
        --weights weights/nasn-pmf-2048-epoch449-ema.safetensors --device cuda --output outputs/pmf_dds
    python examples/run.py --algo sirt --input examples/data/toy_parallel_ct_3d.npz --device cuda --output outputs/sirt_par3d
    python examples/run.py --algo sirt --input examples/data/toy_cone_ct_3d.npz --device cuda --output outputs/sirt_cone

Parameters default to the recommended values in inv_framework/registry.py;
override them with ``--set name=value`` (values are parsed as JSON, so
``--set min_value=null`` disables a clamp). The pack layout is described in
examples/data/README.md. Outputs: reconstruction.npy (H x W float32, same
units as the pack's ``image``; Z x Y x X for 3D packs), preview.png
and summary.json.
"""
import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inv_framework import registry  # noqa: E402
from inv_framework.utils.reconstruction_comparison import (  # noqa: E402
    model_measurement, reconstruction_metrics)


def load_pack(path, device):
    """Return (operator, measurement, reference or None, pack metadata)."""
    with np.load(path, allow_pickle=False) as data:
        pack = {key: data[key] for key in data.files}
    sinogram = torch.from_numpy(pack["sinogram"].astype(np.float32))
    theta = torch.from_numpy(pack["theta"].astype(np.float32))
    mask = pack["det_mask"].astype(bool)
    n_angles, n_det = sinogram.shape[-2:]
    inner = int(mask.sum())
    kind = str(pack["operator"])
    if kind in ("astra_cone", "astra_fan", "astra_parallel3d"):
        if torch.device(device).type != "cuda":
            raise SystemExit("this pack uses the ASTRA CUDA projector; pass --device cuda")
        from inv_framework.operators.ct import (ConeBeamASTRA, FanBeamASTRA2D, ParallelBeamASTRA3D,
                                                circular_cone_geometry, fan_beam_geometry,
                                                parallel_beam_3d_geometry)
        lab = dict(sod=float(pack["sod"]), sdd=float(pack["sdd"]),
                   voxel_size=float(pack["voxel_size"]), angles=pack["theta"],
                   det_offset_u=float(pack["det_offset_u"]))
        image_shape = (tuple(pack["volume_shape"]) if "volume_shape" in pack else pack["image"].shape)
        if kind == "astra_parallel3d":
            del lab["sod"], lab["sdd"]
            geometry = parallel_beam_3d_geometry(image_shape, (sinogram.shape[0], n_det),
                                                 det_pixel=tuple(pack["det_pixel"]), **lab)
            operator = ParallelBeamASTRA3D(geometry, device=device)
        elif kind == "astra_cone":
            geometry = circular_cone_geometry(image_shape, (sinogram.shape[0], n_det),
                                              det_pixel=tuple(pack["det_pixel"]), **lab)
            operator = ConeBeamASTRA(geometry, device=device)
        else:
            geometry = fan_beam_geometry(image_shape[-1], n_det,
                                         det_pixel=float(pack["det_pixel"]), **lab)
            operator = FanBeamASTRA2D(geometry, device=device)
        expected = np.ones(n_det, bool)
        n_angles = operator.num_angles
    elif kind == "parallel_torch":
        from inv_framework.operators.ct.radon_torch import ParallelBeamRadon2D
        from inv_framework.operators.ct.truncated import TruncatedRadon2D
        operator = (ParallelBeamRadon2D(n_det, angles=theta, device=device) if inner == n_det
                    else TruncatedRadon2D(n_det, inner=inner, angles=theta, device=device))
        expected = (np.ones(n_det, bool) if inner == n_det
                    else operator.trunc._mask.cpu().numpy().astype(bool))
    elif kind == "tomopy_cuda":
        if torch.device(device).type != "cuda":
            raise SystemExit("this pack uses the CUDA projector; pass --device cuda")
        from inv_framework.operators.ct import TomoPyCUDARadon2D
        operator = TomoPyCUDARadon2D(n_det, inner=inner, center=float(pack["center"]),
                                     angles=theta, device=device)
        expected = operator._mask.cpu().numpy().astype(bool)
    else:
        raise SystemExit(f"unknown operator kind {kind!r} in {path}")
    if not np.array_equal(mask, expected):
        raise SystemExit("det_mask is not the centred band this operator measures")
    y = sinogram.reshape(1, *operator.range_shape).to(device)
    reference = (torch.from_numpy(pack["image"].astype(np.float32))
                 .reshape(1, *operator.domain_shape).to(device) if "image" in pack else None)
    info = {
        "operator": kind, "image_size": n_det, "num_angles": n_angles,
        "volume_3d": len(operator.domain_shape) == 3 and operator.domain_shape[0] > 1,
        "measured_detectors": inner, "center": float(pack["center"]),
        "image_scale": float(pack.get("image_scale", 1.0)),
        "image_offset": float(pack.get("image_offset", 0.0)),
        "meta": json.loads(str(pack["meta"])) if "meta" in pack else {},
    }
    return operator, y, reference, info


def volume_metrics(image, measurement, operator, reference):
    """PSNR over the volume and 3D SSIM, for cone-beam packs."""
    from skimage.metrics import structural_similarity
    with torch.no_grad():
        residual = torch.linalg.vector_norm(operator(image) - measurement)
        metrics = {"relative_measurement_l2": float(
            residual / torch.linalg.vector_norm(measurement).clamp_min(1e-30))}
    if reference is not None:
        x, ref = image[0].cpu().numpy(), reference[0].cpu().numpy()
        data_range = float(ref.max() - ref.min())
        mse = float(np.mean((x - ref) ** 2))
        metrics.update(psnr_db=10 * math.log10(data_range ** 2 / mse),
                       ssim=float(structural_similarity(ref, x, data_range=data_range)),
                       data_range=data_range)
    return metrics


def central_slices(volume):
    """Axial and coronal mid-slices side by side, for previews of 3D volumes."""
    nz, ny, nx = volume.shape
    axial, coronal = volume[nz // 2], volume[:, ny // 2, :]
    pad = np.zeros((max(axial.shape[0], coronal.shape[0]), axial.shape[1] + coronal.shape[1]), np.float32)
    pad[:axial.shape[0], :axial.shape[1]] = axial
    pad[:coronal.shape[0], axial.shape[1]:] = coronal
    return pad


def preview(path, image, reference):
    from PIL import Image
    if image.ndim == 3:
        image = central_slices(image)
        reference = None if reference is None else central_slices(reference)
    panels = [reference, image] if reference is not None else [image]
    base = reference if reference is not None else image
    lo, hi = np.percentile(base, [0.5, 99.5])
    tiles = [np.clip((p - lo) / max(hi - lo, 1e-12), 0, 1) for p in panels]
    if reference is not None:
        error = np.abs(image - reference)
        tiles.append(np.clip(error / max(hi - lo, 1e-12) * 4, 0, 1))
    strip = np.concatenate(tiles, axis=1)
    picture = Image.fromarray((strip * 255).astype(np.uint8))
    if picture.height > 512:
        picture = picture.resize((picture.width * 512 // picture.height, 512), Image.BILINEAR)
    picture.save(path)


def parse_overrides(items):
    overrides = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--set expects name=value, got {item!r}")
        try:
            overrides[name] = json.loads(value)
        except json.JSONDecodeError:
            overrides[name] = value
    return overrides


def list_algorithms():
    rows = {}
    for name, spec in registry.ALGORITHMS.items():
        rows[name] = {"title": spec.title, "learned": spec.learned, "weights": spec.weights,
                      "toy_data": spec.toy_data, "device": spec.device,
                      "defaults": registry.defaults(name)}
    print(json.dumps(rows, indent=2, ensure_ascii=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="print algorithms and defaults as JSON")
    parser.add_argument("--algo", choices=sorted(registry.ALGORITHMS))
    parser.add_argument("--input", type=Path, help=".npz pack (see examples/data/README.md)")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--weights", help="pretrained weights directory or file (learned methods)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                        help="override a parameter (JSON value)")
    args = parser.parse_args(argv)
    if args.list:
        list_algorithms()
        return 0
    if not (args.algo and args.input and args.output):
        parser.error("--algo, --input and --output are required")

    spec = registry.ALGORITHMS[args.algo]
    params = registry.resolve_params(args.algo, parse_overrides(args.set))
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    operator, y, reference, info = load_pack(args.input, device)

    # Learned priors work in their training range; non-learned methods in the
    # pack's physical units.
    scale, offset = ((info["image_scale"], info["image_offset"]) if spec.learned else (1.0, 0.0))
    target = model_measurement(y, operator, scale, offset)
    solve = registry.build(args.algo, operator, params, weights=args.weights,
                           device=device, seed=args.seed)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = solve(target)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    image = (result.detach() * scale + offset).float()
    if not torch.isfinite(image).all():
        raise SystemExit("reconstruction contains non-finite values")

    roi = (info["measured_detectors"] / 2 if info["measured_detectors"] < info["image_size"]
           else None)
    if info["volume_3d"]:
        metrics = volume_metrics(image, y, operator, reference)
    else:
        metrics = reconstruction_metrics(image, y, operator, reference)
    if roi is not None and reference is not None:
        roi_metrics = reconstruction_metrics(image, y, operator, reference, roi_radius=roi)
        metrics.update(roi_radius=roi, roi_psnr_db=roi_metrics["psnr_db"],
                       roi_ssim=roi_metrics["ssim"])

    args.output.mkdir(parents=True, exist_ok=True)
    recon = image[0].cpu().numpy()
    recon = recon if info["volume_3d"] else recon[0]
    np.save(args.output / "reconstruction.npy", recon)
    truth = None if reference is None else reference[0].cpu().numpy()
    preview(args.output / "preview.png", recon,
            None if truth is None else (truth if info["volume_3d"] else truth[0]))
    summary = {
        "algorithm": args.algo, "title": spec.title, "parameters": params,
        "seed": args.seed, "device": str(device), "seconds": round(seconds, 3),
        "input": {"path": str(args.input),
                  "sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(), **info},
        "weights": args.weights, "metrics": metrics,
        "versions": {"python": platform.python_version(), "torch": torch.__version__},
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    shown = {k: (round(v, 4) if isinstance(v, float) and math.isfinite(v) else v)
             for k, v in metrics.items() if k != "data_range"}
    print(json.dumps({"algorithm": args.algo, "seconds": summary["seconds"], **shown},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
