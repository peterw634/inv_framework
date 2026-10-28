"""Compare a real slice with an independent nasn-truncated-ct checkout.

The external checkout is used only as a regression reference. Normal use of
inv_framework does not import it. Outputs and an explicit numerical gate
are written to the requested directory; source data are only read.
"""

import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--sinogram-dir", required=True)
    parser.add_argument("--scale-stats", required=True)
    parser.add_argument("--slice", type=int, default=1150)
    parser.add_argument("--num-steps", type=int, default=16)
    parser.add_argument("--dc-iter", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import numpy as np
    import tifffile
    import torch
    from inv_framework.applications.nasn_ct.measurement import load_scale_stats
    from inv_framework.applications.nasn_ct.reconstruct import reconstruct_slices

    sys.path.insert(0, str(Path(args.reference_root).resolve() / "src"))
    from nasn_ct.reconstruct import reconstruct_slices as original_reconstruct

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real-data migration check")
    scale = load_scale_stats(args.scale_stats)
    output = Path(args.output)
    options = dict(slices=[args.slice], num_steps=args.num_steps, dc_iter=args.dc_iter)
    print("Running original NaSn reconstruction...", flush=True)
    old = original_reconstruct(args.weights, args.sinogram_dir, scale, output / "original", **options)
    gc.collect()
    torch.cuda.empty_cache()
    print("Running framework reconstruction...", flush=True)
    new = reconstruct_slices(args.weights, args.sinogram_dir, scale, output / "framework", **options)
    filename = f"recon_slice_{args.slice:06d}.tif"
    a = tifffile.imread(output / "original" / filename).astype(np.float64)
    b = tifffile.imread(output / "framework" / filename).astype(np.float64)
    difference = b - a
    old_res = old["slices"][str(args.slice)]["measured_band_relative_l2"]
    new_res = new["slices"][str(args.slice)]["measured_band_relative_l2"]
    relative_l2 = float(np.linalg.norm(difference) / max(np.linalg.norm(a), 1e-30))
    # Native BP uses atomic accumulation; separate runs need not be bit-exact.
    gates = {"relative_image_l2": 1e-3, "absolute_residual_difference": 1e-4}
    report = {
        "reference_root": str(Path(args.reference_root).resolve()),
        "slice": args.slice, "num_steps": args.num_steps, "dc_iter": args.dc_iter,
        "original_residual": old_res, "framework_residual": new_res,
        "relative_image_l2": relative_l2,
        "maximum_absolute_difference": float(np.abs(difference).max()),
        "mean_absolute_difference": float(np.abs(difference).mean()),
        "gates": gates,
        "passed": (relative_l2 <= gates["relative_image_l2"]
                   and abs(old_res - new_res) <= gates["absolute_residual_difference"]),
    }
    (output / "comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
