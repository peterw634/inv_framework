"""Run every registered algorithm on its toy packs and collect algorithms/results.json.

    python scripts/run_toy_benchmarks.py --gpus 2 6 7
    python scripts/run_toy_benchmarks.py --gpus 3 --only sirt dps --jobs-per-gpu 2

Each (algorithm, pack) pair runs ``examples/run.py`` in its own process with
the recommended defaults; outputs go to outputs/toy_benchmarks/<algo>/<pack>/.
FBP/SIRT/Landweber/CG on the parallel-beam pack run on CPU (8 threads).
Results already present in results.json are kept unless --rerun is given.
"""
import argparse
import json
import os
import platform
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inv_framework import registry  # noqa: E402

WEIGHTS = {"dm4ct-pixel": "weights/lodoind_pixel_diffusion",
           "dm4ct-latent": "weights/lodoind_latent_diffusion",
           "pmf": "weights/nasn-pmf-2048-epoch449-ema.safetensors"}


# Fast non-learned methods are benchmarked on CPU (parallel-beam pack only); DIP/INR
# train a network per image and run on GPU, where they take ~1 min instead of ~1 h.
CPU_BENCHMARK = {"fbp", "sirt", "landweber", "cg_tikhonov"}
CPU_THREADS = "8"   # keep CPU runs from starving the GPU jobs' launch threads


def jobs(only):
    for name, spec in registry.ALGORITHMS.items():
        if only and name not in only:
            continue
        for pack in spec.toy_data:
            cpu = name in CPU_BENCHMARK and Path(pack).name == "toy_parallel_ct_128.npz"
            yield name, pack, cpu


def run_one(name, pack, cpu, gpu, output_root):
    spec = registry.ALGORITHMS[name]
    out = output_root / name / Path(pack).stem
    cmd = [sys.executable, "examples/run.py", "--algo", name, "--input", pack,
           "--output", str(out), "--device", "cpu" if cpu else "cuda"]
    if spec.weights:
        cmd += ["--weights", WEIGHTS[spec.weights]]
    env = dict(os.environ, NUMPY_MADVISE_HUGEPAGE="0")
    env["CUDA_VISIBLE_DEVICES"] = "" if cpu else str(gpu)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env[var] = CPU_THREADS
    started = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    record = {"algorithm": name, "pack": Path(pack).name, "device": "cpu" if cpu else "cuda",
              "wall_seconds": round(time.time() - started, 1),
              "command": " ".join(cmd[1:]).replace(f"{ROOT}/", "")}
    if proc.returncode == 0:
        summary = json.loads((out / "summary.json").read_text())
        record.update(status="passed", seconds=summary["seconds"], metrics=summary["metrics"],
                      parameters=summary["parameters"])
    else:
        record.update(status="failed", error=(proc.stderr or proc.stdout)[-2000:])
    return record


def gpu_name(gpu):
    try:
        return subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--results", type=Path, default=ROOT / "algorithms/results.json")
    parser.add_argument("--outputs", type=Path, default=ROOT / "outputs/toy_benchmarks")
    args = parser.parse_args()

    results = json.loads(args.results.read_text()) if args.results.exists() else {"runs": {}}
    pending = [job for job in jobs(args.only)
               if args.rerun or f"{job[0]}:{Path(job[1]).name}" not in results["runs"]
               or results["runs"][f"{job[0]}:{Path(job[1]).name}"]["status"] != "passed"]
    work = queue.Queue()
    for job in sorted(pending, key=lambda j: j[2]):     # GPU jobs first, CPU jobs fill in
        work.put(job)
    lock = threading.Lock()

    def worker(gpu):
        while True:
            try:
                name, pack, cpu = work.get_nowait()
            except queue.Empty:
                return
            record = run_one(name, pack, cpu, gpu, args.outputs)
            with lock:
                results["runs"][f"{name}:{Path(pack).name}"] = record
                results["environment"] = {"python": platform.python_version(),
                                          "gpu": gpu_name(args.gpus[0])}
                args.results.parent.mkdir(parents=True, exist_ok=True)
                args.results.write_text(json.dumps(results, indent=2, ensure_ascii=False))
                shown = record.get("metrics", {})
                print(f"[gpu {gpu}] {name:14s} {Path(pack).stem:24s} {record['status']:6s} "
                      f"{record.get('seconds', '-')}s psnr={shown.get('psnr_db', '-')}", flush=True)

    threads = [threading.Thread(target=worker, args=(gpu,))
               for gpu in args.gpus for _ in range(args.jobs_per_gpu)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    failed = [k for k, v in results["runs"].items() if v["status"] != "passed"]
    print("failed:", failed or "none")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
