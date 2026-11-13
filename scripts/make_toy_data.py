"""Regenerate the toy datasets in examples/data/ (see examples/data/README.md).

    python scripts/make_toy_data.py --output examples/data
    python scripts/make_toy_data.py --output examples/data \
        --lodoind-slice /path/to/LoDoInd_preprocessed/ref_normalized/test/2750.tif
    python scripts/make_toy_data.py --output examples/data --device cuda \
        --pmf-weights weights/nasn-pmf-2048-epoch449-ema.safetensors

The two synthetic 2D parallel-beam packs need only torch and run on CPU; the
ASTRA packs (3D parallel beam, cone and fan beam) need astra-toolbox and a CUDA
GPU, which ``--device cuda`` enables. The LoDoInd packs need one preprocessed LoDoInd
test slice; the pMF packs need the released NaSn pMF weights. Every pack is
written with ``np.savez_compressed`` and follows the same key layout, which
``examples/run.py`` reads.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.ct_demo import shepp_logan  # noqa: E402
from inv_framework.operators.ct.radon_torch import ParallelBeamRadon2D  # noqa: E402
from inv_framework.operators.ct.truncated import TruncatedRadon2D  # noqa: E402


def angles(count):
    return np.linspace(0, np.pi, count, endpoint=False, dtype=np.float32)


def save_pack(path, image, sinogram, theta, det_mask, center, operator,
              noise_sigma, meta, image_scale=1.0, image_offset=0.0,
              image_dtype=np.float32, geometry=None):
    extra = {}
    if geometry is not None:
        # Lab geometries: distances and pitches in voxel units (voxel_size = 1).
        extra = {key: np.asarray(geometry[key], dtype=np.float32)
                 for key in ("sod", "sdd", "voxel_size", "det_pixel", "det_offset_u")}
    np.savez_compressed(
        path,
        **extra,
        image=np.asarray(image, dtype=image_dtype),
        sinogram=np.asarray(sinogram, dtype=np.float32),
        theta=np.asarray(theta, dtype=np.float32),
        det_mask=np.asarray(det_mask, dtype=bool),
        center=np.float32(center),
        operator=np.str_(operator),
        image_scale=np.float32(image_scale),
        image_offset=np.float32(image_offset),
        noise_sigma=np.float32(noise_sigma),
        meta=np.str_(json.dumps(meta, ensure_ascii=False)),
    )
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    print(f"{digest}  {Path(path).name}  ({Path(path).stat().st_size / 1e6:.2f} MB)")
    return digest


def parallel_pack(output, size=128, num_angles=30, noise_sigma=0.5, seed=0):
    """Sparse-view Shepp-Logan, the same case as examples/ct_demo.py."""
    theta = angles(num_angles)
    A = ParallelBeamRadon2D(size, angles=torch.from_numpy(theta))
    image = shepp_logan(size)[None, None]
    clean = A(image)
    noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(seed))
    sinogram = clean + noise_sigma * noise
    meta = {
        "description": "Modified Shepp-Logan phantom, sparse-view parallel-beam CT with additive Gaussian noise",
        "source": "synthetic (examples/ct_demo.py:shepp_logan)",
        "license": "CC0-1.0 (generated data)",
        "preprocessing": "none; sinogram is already a line-integral (post -log) projection",
        "units": {"image": "attenuation per pixel, range [0, 1]",
                  "sinogram": "line integral in pixel units (sum of image values along a ray)",
                  "theta": "radians"},
        "geometry": "2D parallel beam, detector pitch = image pixel, rotation centre = n_det/2",
        "seed": seed,
    }
    return save_pack(output / "toy_parallel_ct_128.npz", image[0, 0], sinogram[0, 0],
                     theta, np.ones(size, bool), size / 2, "parallel_torch",
                     noise_sigma, meta)


def truncated_pack(output, size=128, num_angles=90, inner=48):
    """Interior tomography: only the central `inner` detector columns are measured."""
    theta = angles(num_angles)
    A = TruncatedRadon2D(size, inner=inner, angles=torch.from_numpy(theta))
    image = shepp_logan(size)[None, None]
    sinogram = A(image)
    mask = A.trunc._mask.numpy().astype(bool)
    meta = {
        "description": "Modified Shepp-Logan phantom, interior (truncated-detector) parallel-beam CT, noiseless",
        "source": "synthetic (examples/ct_demo.py:shepp_logan), same case as examples/truncated_ct_demo.py",
        "license": "CC0-1.0 (generated data)",
        "preprocessing": "none; unmeasured detector columns are stored as 0 and flagged by det_mask",
        "units": {"image": "attenuation per pixel, range [0, 1]",
                  "sinogram": "line integral in pixel units", "theta": "radians"},
        "geometry": f"2D parallel beam, {inner}/{size} central detector columns measured "
                    f"(columns {(size - inner) // 2}..{(size - inner) // 2 + inner - 1})",
        "seed": None,
    }
    return save_pack(output / "toy_truncated_ct_128.npz", image[0, 0], sinogram[0, 0],
                     theta, mask, size / 2, "parallel_torch", 0.0, meta)


def lodoind_pack(output, slice_path, num_angles=40):
    """One LoDoInd test slice in the DM4CT 512 sparse-view setting."""
    import tifffile
    stored = tifffile.imread(slice_path).astype(np.float32)
    if stored.shape != (512, 512):
        raise ValueError(f"expected a 512x512 DM4CT-preprocessed slice, got {stored.shape}")
    theta = angles(num_angles)
    A = ParallelBeamRadon2D(512, angles=torch.from_numpy(theta))
    image = (stored + 1) / 2                      # model range [-1, 1] -> [0, 1]
    sinogram = A(torch.from_numpy(image)[None, None])
    meta = {
        "description": "One LoDoInd reference slice, sparse-view (40 angles) parallel-beam CT, noiseless (DM4CT NoNoise setting)",
        "source": f"LoDoInd (Shi et al., doi:10.5281/zenodo.10391412), test slice {Path(slice_path).name} "
                  "after DM4CT preprocess_lodoind.py (0.5/99.5 percentile clip, [-1,1] rescale, LANCZOS resize to 512)",
        "license": "CC BY-NC 4.0 (derived from LoDoInd); attribution required, non-commercial use only",
        "preprocessing": "image = (stored + 1) / 2; sinogram = A(image) with the torch parallel-beam operator",
        "units": {"image": "normalised attenuation, approx. [0, 1] (LANCZOS overshoot allowed)",
                  "sinogram": "line integral in pixel units", "theta": "radians"},
        "geometry": "2D parallel beam, 512 detector columns, 40 angles in [0, pi) "
                    "(DM4CT used np.linspace(0, pi, 40) with the endpoint included)",
        "model_domain": "pretrained DM4CT priors use x_model in [-1, 1]; x_phys = 0.5 * x_model + 0.5",
        "seed": None,
    }
    return save_pack(output / "toy_lodoind_512.npz", image, sinogram[0, 0], theta,
                     np.ones(512, bool), 256.0, "parallel_torch", 0.0, meta,
                     image_scale=0.5, image_offset=0.5)


def pmf_pack(output, weights, device, seed=20261113, steps=16):
    """pMF-generated 2048 slice measured with the NaSn synthetic-benchmark geometry.

    The ground truth is an unconditional sample of the released prior (seed
    ``seed``), so the pack contains no NaSn scan data. Reconstructions must use
    a different noise seed, otherwise the prior would reproduce the sample
    without looking at the measurement.
    """
    from inv_framework.applications.nasn_ct.random import BatchGenerator
    from inv_framework.applications.nasn_ct.weights import load_pmf
    from inv_framework.operators.ct import TomoPyCUDARadon2D
    from inv_framework.solvers.diffusion import DDSSolver, PMFVelocityPrior
    pmf = load_pmf(weights, device=device)
    solver = DDSSolver(PMFVelocityPrior(pmf), num_inference_steps=steps, dc_mode="none")
    A = TomoPyCUDARadon2D(2048, 2048, 512, 1024.0, device=device)
    noise = BatchGenerator(device, [seed]).randn((1, *A.domain_shape))
    with torch.no_grad():
        model_image = solver.solve(torch.zeros(1, *A.range_shape, device=device), A,
                                   initial_noise=noise)
        image = (model_image + 1) / 2
        sinogram = A(image)
    mask = A._mask.cpu().numpy().astype(bool)
    meta = {
        "description": "Unconditional NaSn pMF sample used as ground truth, interior CT with the NaSn synthetic-benchmark geometry, noiseless",
        "source": f"generated by nasn-pmf-2048-epoch449-ema (model-v0.1.0, MIT), {steps}-step mean-flow sampling, "
                  f"BatchGenerator seed {seed}; contains no NaSn scan data",
        "license": "MIT (output of the MIT-licensed release weights)",
        "preprocessing": "image = (pMF sample + 1) / 2; sinogram = A(image) with TomoPyCUDARadon2D(2048, 2048 angles, inner=512, center=1024)",
        "units": {"image": "normalised attenuation in the pMF training range mapped to approx. [0, 1]",
                  "sinogram": "line integral in pixel units; unmeasured columns stored as 0", "theta": "radians"},
        "geometry": "2D parallel beam, 2048 angles in [0, pi), 2048 detector columns of which the central 512 are measured (25% FOV), centre 1024",
        "model_domain": "x_phys = 0.5 * x_model + 0.5 (same mapping as the NaSn synthetic benchmark)",
        "seed": seed,
        "image_dtype": "float16 to keep the pack small",
    }
    return save_pack(output / "toy_nasn_pmf_2048.npz", image[0, 0].cpu().numpy(),
                     sinogram[0, 0].cpu().numpy(), A.angles.cpu().numpy(), mask, 1024.0,
                     "tomopy_cuda", 0.0, meta, image_scale=0.5, image_offset=0.5,
                     image_dtype=np.float16)


LAB_UNITS = ("lengths in voxels (voxel_size = 1): sod/sdd/det_pixel are given in the same unit, "
             "so the sinogram is a line integral in voxel units")


def battery_phantom(shape):
    """Cylindrical cell: steel can, layered jelly roll, centre pin, caps, a few inclusions.

    Values are attenuation per voxel; modelled on the cylinder/shell/rod
    phantom of the battery multi-cell benchmark's operator check.
    """
    nz, ny, nx = shape
    y, x = np.meshgrid(np.arange(ny) - (ny - 1) / 2, np.arange(nx) - (nx - 1) / 2, indexing="ij")
    r = np.hypot(x, y)
    radius = 0.4 * nx
    can = (r > radius - 3) & (r <= radius)
    roll = (r > 6) & (r <= radius - 3)
    layers = 0.25 + 0.15 * (np.cos(2 * np.pi * r / 4.0) > 0)
    pin = r <= 4
    plane = np.where(can, 1.0, 0) + np.where(roll, layers, 0) + np.where(pin, 0.8, 0)
    volume = np.repeat(plane[None], nz, 0).astype(np.float32)
    z = np.arange(nz)
    cap = (z < 0.12 * nz) | (z >= 0.88 * nz)
    outside = (z < 0.08 * nz) | (z >= 0.92 * nz)
    volume[cap] = np.where(r <= radius, 0.9, 0)[None]
    volume[outside] = 0
    rng = np.random.default_rng(3)
    zz, yy, xx = np.meshgrid(np.arange(nz), np.arange(ny) - (ny - 1) / 2,
                             np.arange(nx) - (nx - 1) / 2, indexing="ij")
    for _ in range(6):
        cz = rng.uniform(0.25, 0.75) * nz
        angle, rad = rng.uniform(0, 2 * np.pi), rng.uniform(10, radius - 10)
        cy, cx = rad * np.sin(angle), rad * np.cos(angle)
        blob = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= rng.uniform(2, 4) ** 2
        volume[blob] = 1.2
    return volume


def cone_pack(output, device, shape=(64, 128, 128), det_shape=(80, 144), views=120,
              sod=400.0, sdd=600.0, noise_rel=0.005, seed=0):
    """Lab cone-beam scan (full turn, sparse views) of a synthetic cylindrical cell."""
    from inv_framework.operators.ct import ConeBeamASTRA, circular_cone_geometry
    geometry = circular_cone_geometry(shape, det_shape, sod, sdd, num_angles=views)
    A = ConeBeamASTRA(geometry, device=device)
    volume = battery_phantom(shape)
    clean = A(torch.from_numpy(volume)[None].to(device))[0].cpu()
    sigma = noise_rel * float(clean.max())
    noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(seed))
    sinogram = clean + sigma * noise
    meta = {
        "description": "Synthetic cylindrical battery cell (can, layered jelly roll, pin, caps, inclusions), "
                       "laboratory circular cone-beam CT, 120 views over 360 deg, additive Gaussian noise",
        "source": "synthetic (scripts/make_toy_data.py:battery_phantom)",
        "license": "CC0-1.0 (generated data)",
        "preprocessing": "none; sinogram is a line-integral (post -log) projection in ASTRA order (rows, views, cols)",
        "units": {"image": "attenuation per voxel, (nz, ny, nx), range [0, 1.2]",
                  "sinogram": LAB_UNITS, "theta": "source angle, radians"},
        "geometry": f"ASTRA cone_vec circular orbit, SOD {sod:g}, SDD {sdd:g} (magnification {sdd / sod:g}), "
                    f"detector {det_shape[0]} x {det_shape[1]} pixels of pitch {sdd / sod:g}, centred, no offset",
        "noise": f"Gaussian, sigma = {noise_rel:g} x max(sinogram) = {sigma:.4g}",
        "seed": seed,
    }
    return save_pack(output / "toy_cone_ct_3d.npz", volume, sinogram.numpy(), geometry["angles"],
                     np.ones(det_shape[1], bool), (det_shape[1] - 1) / 2, "astra_cone", sigma, meta,
                     geometry=geometry)


def parallel_3d_pack(output, device, shape=(64, 128, 128), views=90, noise_rel=0.005, seed=0):
    """Synchrotron-style 3D parallel-beam scan (180 deg, sparse views) of the same cell."""
    from inv_framework.operators.ct import ParallelBeamASTRA3D, parallel_beam_3d_geometry
    det_shape = (shape[0], shape[2])
    geometry = parallel_beam_3d_geometry(shape, det_shape, num_angles=views)
    A = ParallelBeamASTRA3D(geometry, device=device)
    volume = battery_phantom(shape)
    clean = A(torch.from_numpy(volume)[None].to(device))[0].cpu()
    sigma = noise_rel * float(clean.max())
    noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(seed))
    sinogram = clean + sigma * noise
    meta = {
        "description": "Synthetic cylindrical battery cell (same phantom as toy_cone_ct_3d), synchrotron-style "
                       f"3D parallel-beam CT, {views} views over 180 deg, additive Gaussian noise",
        "source": "synthetic (scripts/make_toy_data.py:battery_phantom)",
        "license": "CC0-1.0 (generated data)",
        "preprocessing": "none; sinogram is a line-integral (post -log) projection in ASTRA order (rows, views, cols)",
        "units": {"image": "attenuation per voxel, (nz, ny, nx), range [0, 1.2]",
                  "sinogram": "line integral in voxel units (detector pitch = voxel)", "theta": "radians"},
        "geometry": f"ASTRA parallel3d_vec, detector {det_shape[0]} x {det_shape[1]} pixels of pitch 1 voxel, "
                    "rotation axis on the detector centre; each detector row maps to one slice",
        "noise": f"Gaussian, sigma = {noise_rel:g} x max(sinogram) = {sigma:.4g}",
        "seed": seed,
    }
    return save_pack(output / "toy_parallel_ct_3d.npz", volume, sinogram.numpy(), geometry["angles"],
                     np.ones(det_shape[1], bool), (det_shape[1] - 1) / 2, "astra_parallel3d", sigma, meta,
                     geometry=dict(geometry, sod=0.0, sdd=0.0))


def fan_geometry(size, views):
    from inv_framework.operators.ct import fan_beam_geometry
    sod = 2.0 * size
    return fan_beam_geometry(size, size + size // 8, sod=sod, sdd=1.5 * sod, num_angles=views)


def lodoind_fan_pack(output, slice_path, device, views=80):
    """The LoDoInd slice of toy_lodoind_512 measured with a lab fan-beam scanner."""
    import tifffile
    from inv_framework.operators.ct import FanBeamASTRA2D
    stored = tifffile.imread(slice_path).astype(np.float32)
    image = (stored + 1) / 2
    geometry = fan_geometry(512, views)
    A = FanBeamASTRA2D(geometry, device=device)
    sinogram = A(torch.from_numpy(image)[None, None].to(device))[0, 0].cpu().numpy()
    meta = {
        "description": "One LoDoInd reference slice, laboratory fan-beam (cone-beam mid-plane) CT, 80 views over 360 deg, noiseless",
        "source": f"LoDoInd (Shi et al., doi:10.5281/zenodo.10391412), test slice {Path(slice_path).name} after DM4CT preprocess_lodoind.py",
        "license": "CC BY-NC 4.0 (derived from LoDoInd); attribution required, non-commercial use only",
        "preprocessing": "image = (stored + 1) / 2; sinogram = A(image) with the ASTRA fan-beam operator",
        "units": {"image": "normalised attenuation, approx. [0, 1]", "sinogram": LAB_UNITS,
                  "theta": "source angle, radians"},
        "geometry": f"ASTRA fanflat_vec, SOD {geometry['sod']:g}, SDD {geometry['sdd']:g}, "
                    f"{geometry['det_cols']} detector pixels of pitch {geometry['det_pixel']:g}, full turn "
                    "(same angular density as 40 parallel-beam views over 180 deg)",
        "model_domain": "pretrained DM4CT priors use x_model in [-1, 1]; x_phys = 0.5 * x_model + 0.5",
        "seed": None,
    }
    return save_pack(output / "toy_lodoind_fan_512.npz", image, sinogram, geometry["angles"],
                     np.ones(geometry["det_cols"], bool), (geometry["det_cols"] - 1) / 2, "astra_fan",
                     0.0, meta, image_scale=0.5, image_offset=0.5, geometry=geometry)


def pmf_fan_pack(output, pmf_parallel_pack, device, views=180):
    """The pMF ground truth of toy_nasn_pmf_2048 measured with a lab fan-beam scanner."""
    from inv_framework.operators.ct import FanBeamASTRA2D
    with np.load(pmf_parallel_pack) as data:
        image = data["image"].astype(np.float32)
        source_meta = json.loads(str(data["meta"]))
    geometry = fan_geometry(2048, views)
    A = FanBeamASTRA2D(geometry, device=device)
    sinogram = A(torch.from_numpy(image)[None, None].to(device))[0, 0].cpu().numpy()
    meta = {
        "description": "The pMF-generated slice of toy_nasn_pmf_2048 (no NaSn scan data), laboratory fan-beam CT, "
                       f"{views} views over 360 deg, full field of view, noiseless",
        "source": source_meta["source"],
        "license": "MIT (output of the MIT-licensed release weights)",
        "preprocessing": "image as in toy_nasn_pmf_2048 (float16); sinogram = A(image) with the ASTRA fan-beam operator",
        "units": {"image": "normalised attenuation, approx. [0, 1]", "sinogram": LAB_UNITS,
                  "theta": "source angle, radians"},
        "geometry": f"ASTRA fanflat_vec, SOD {geometry['sod']:g}, SDD {geometry['sdd']:g}, "
                    f"{geometry['det_cols']} detector pixels of pitch {geometry['det_pixel']:g}, full turn",
        "model_domain": "x_phys = 0.5 * x_model + 0.5",
        "seed": source_meta.get("seed"),
        "image_dtype": "float16 to keep the pack small",
    }
    return save_pack(output / "toy_nasn_pmf_fan_2048.npz", image, sinogram, geometry["angles"],
                     np.ones(geometry["det_cols"], bool), (geometry["det_cols"] - 1) / 2, "astra_fan",
                     0.0, meta, image_scale=0.5, image_offset=0.5, image_dtype=np.float16,
                     geometry=geometry)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("examples/data"))
    parser.add_argument("--lodoind-slice", type=Path,
                        help="DM4CT-preprocessed LoDoInd test slice (512x512 TIFF)")
    parser.add_argument("--pmf-weights", type=Path,
                        help="nasn-pmf-2048-epoch449-ema.safetensors (JSON config alongside)")
    parser.add_argument("--device", default="cpu",
                        help="cuda also writes the ASTRA cone/fan-beam packs")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    lab = torch.device(args.device).type == "cuda"
    parallel_pack(args.output)
    truncated_pack(args.output)
    if lab:
        parallel_3d_pack(args.output, args.device)
        cone_pack(args.output, args.device)
    if args.lodoind_slice:
        lodoind_pack(args.output, args.lodoind_slice)
        if lab:
            lodoind_fan_pack(args.output, args.lodoind_slice, args.device)
    if args.pmf_weights:
        pmf_pack(args.output, args.pmf_weights, args.device)
        pmf_fan_pack(args.output, args.output / "toy_nasn_pmf_2048.npz", args.device)


if __name__ == "__main__":
    main()
