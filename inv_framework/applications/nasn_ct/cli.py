"""Command-line interface for data preparation and reconstruction."""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from .config import load_config

_HERE = Path(__file__).resolve().parent


def _add_manifest(parser):
    parser.add_argument("--manifest", default=str(_HERE / "manifests/data-v0.1.0.json"))
    parser.add_argument("--profile", choices=["quickstart", "target", "training", "all"],
                        default="quickstart")
    parser.add_argument("--scan", type=int, choices=[1, 2, 3, 4])


def build_parser():
    parser = argparse.ArgumentParser(prog="inv-nasn-ct", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("data", help="download and prepare data")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    fetch = data_commands.add_parser("fetch", help="download and verify release archives")
    _add_manifest(fetch); fetch.add_argument("--output", default="data/downloads")
    fetch.add_argument("--extract", action="store_true")
    verify = data_commands.add_parser("verify", help="verify downloaded archives")
    _add_manifest(verify); verify.add_argument("--directory", default="data/downloads")
    prepare_target = data_commands.add_parser("prepare-target", help="prepare raw NaSn sinograms")
    prepare_target.add_argument("--config", default=str(_HERE / "configs/target_legacy.yaml"))
    prepare_target.add_argument("--raw-root", required=True); prepare_target.add_argument("--output", required=True)
    prepare_target.add_argument("--slice-start", type=int); prepare_target.add_argument("--slice-stop", type=int)
    prepare_target.add_argument("--device"); prepare_target.add_argument("--batch-projections", type=int, default=2)
    prepare_full = data_commands.add_parser("prepare-fullct", help="prepare/reconstruct a FullCT scan")
    prepare_full.add_argument("--config", required=True); prepare_full.add_argument("--raw-root", required=True)
    prepare_full.add_argument("--output", required=True); prepare_full.add_argument("--stage", choices=["sinograms", "all"], default="all")
    prepare_full.add_argument("--slice-start", type=int); prepare_full.add_argument("--slice-stop", type=int)
    prepare_full.add_argument("--device"); prepare_full.add_argument("--batch-projections", type=int, default=2)
    calibrate = data_commands.add_parser("calibrate-target", help="recompute target-domain scale")
    calibrate.add_argument("--sinogram-dir", required=True); calibrate.add_argument("--output", required=True)
    calibrate.add_argument("--center", type=float, default=1172.0); calibrate.add_argument("--samples", type=int, default=80)
    corpus = data_commands.add_parser("build-corpus", help="build 3003/1001 pMF corpus")
    corpus.add_argument("--train", nargs=3, required=True); corpus.add_argument("--val", nargs=1, required=True)
    corpus.add_argument("--output", required=True)
    weights = commands.add_parser("weights", help="release-weight utilities")
    weight_commands = weights.add_subparsers(dest="weight_command", required=True)
    export = weight_commands.add_parser("export", help="export EMA as safetensors")
    export.add_argument("--checkpoint", required=True); export.add_argument("--output", required=True)
    reconstruct = commands.add_parser("reconstruct", help="run real NaSn DDS")
    reconstruct.add_argument("--weights", required=True); reconstruct.add_argument("--model-config")
    reconstruct.add_argument("--sinogram-dir", required=True); reconstruct.add_argument("--scale-stats", required=True)
    reconstruct.add_argument("--output", required=True); reconstruct.add_argument("--slices", nargs="+", type=int, default=[1150])
    reconstruct.add_argument("--num-steps", type=int, default=16); reconstruct.add_argument("--dc-lam", type=float, default=0.05)
    reconstruct.add_argument("--dc-iter", type=int, default=10); reconstruct.add_argument("--seed", type=int, default=0)
    reconstruct.add_argument("--omega", type=float, default=1.0); reconstruct.add_argument("--t-start", type=float, default=1.0)
    reconstruct.add_argument("--warm-start", action="store_true")
    reconstruct.add_argument("--device", default="cuda")
    benchmark = commands.add_parser("benchmark", help="run held-out synthetic CUDA benchmark")
    benchmark.add_argument("--weights", required=True); benchmark.add_argument("--model-config")
    benchmark.add_argument("--val-root", required=True); benchmark.add_argument("--output", required=True)
    benchmark.add_argument("--slices", nargs="+", type=int, default=list(range(991, 1001)))
    benchmark.add_argument("--steps", nargs="+", type=int, default=[16, 32])
    benchmark.add_argument("--device", default="cuda")
    commands.add_parser("doctor", help="print runtime capability information")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        import torch
        report = {"python": sys.version.split()[0], "platform": platform.platform(),
                  "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
                  "torch_cuda": torch.version.cuda}
        if torch.cuda.is_available():
            report["gpu"] = torch.cuda.get_device_name(0)
        print(json.dumps(report, indent=2)); return 0
    if args.command == "data" and args.data_command in {"fetch", "verify"}:
        from .data.download import UnpublishedRelease, fetch, verify
        try:
            if args.data_command == "fetch":
                paths = fetch(args.manifest, args.output, args.profile, args.scan, args.extract)
                print("\n".join(map(str, paths)))
            else:
                results = verify(args.manifest, args.directory, args.profile, args.scan)
                for name, valid, detail in results:
                    print(f"{'OK' if valid else 'FAIL'} {name}: {detail}")
                return 0 if all(result[1] for result in results) else 1
        except UnpublishedRelease as error:
            print(str(error), file=sys.stderr); return 2
        return 0
    if args.command == "data":
        from .data.preprocess import (
            build_corpus, calibrate_target_scale, prepare_sinograms, reconstruct_fullct,
        )
        if args.data_command in {"prepare-target", "prepare-fullct"}:
            config = load_config(args.config)
            selected_range = None
            if args.slice_start is not None or args.slice_stop is not None:
                if args.slice_start is None or args.slice_stop is None:
                    raise ValueError("slice-start and slice-stop must be supplied together")
                selected_range = [args.slice_start, args.slice_stop]
            sino = prepare_sinograms(
                args.raw_root, args.output, config, selected_range,
                args.device, args.batch_projections,
            )
            print(sino)
            if args.data_command == "prepare-fullct" and args.stage == "all":
                destination = Path(args.output) / "recon_gridrec"
                reconstruct_fullct(sino, destination, config, device=args.device or "cuda")
                print(destination)
        elif args.data_command == "calibrate-target":
            print(json.dumps(calibrate_target_scale(
                args.sinogram_dir, args.output, args.center, args.samples
            ), indent=2))
        elif args.data_command == "build-corpus":
            print(json.dumps(build_corpus(args.train, args.val, args.output), indent=2))
        return 0
    if args.command == "weights":
        from .weights import export_ema
        print("\n".join(map(str, export_ema(args.checkpoint, args.output))))
        return 0
    if args.command == "reconstruct":
        from .reconstruct import reconstruct_slices
        from .measurement import load_scale_stats
        scale = load_scale_stats(args.scale_stats)
        report = reconstruct_slices(
            args.weights, args.sinogram_dir, scale, args.output, args.slices,
            args.model_config, args.num_steps, args.dc_lam, args.dc_iter,
            args.seed, args.omega, args.t_start, args.warm_start,
            device=args.device,
        )
        print(json.dumps(report, indent=2)); return 0
    if args.command == "benchmark":
        from .benchmark import benchmark_synthetic
        report = benchmark_synthetic(
            args.weights, args.val_root, args.output, args.model_config,
            args.slices, args.steps,
            device=args.device,
        )
        print(json.dumps(report["aggregate"], indent=2))
        return 0 if all(report["acceptance"].values()) else 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
