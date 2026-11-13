"""Algorithm registry, unified runner, toy packs and generated cards."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from inv_framework import registry

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/data"


def test_registry_entries_are_complete():
    for name, spec in registry.ALGORITHMS.items():
        names = [p.name for p in spec.params]
        assert len(names) == len(set(names)), name
        assert spec.task and spec.scope and spec.code and spec.summary, name
        assert spec.learned == (spec.weights is not None), name
        for pack in spec.toy_data:
            assert (ROOT / pack).exists(), pack
        assert all(p.source and p.constraint for p in spec.params), name


def test_resolve_params_rejects_unknown_names():
    assert registry.resolve_params("sirt", {"num_iterations": 3})["num_iterations"] == 3
    with pytest.raises(KeyError):
        registry.resolve_params("sirt", {"iterations": 3})


def test_toy_packs_match_checksums():
    for line in (DATA / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == digest, name


def test_toy_pack_layout():
    for path in DATA.glob("*.npz"):
        with np.load(path) as pack:
            keys = set(pack.files)
            assert {"image", "sinogram", "theta", "det_mask", "operator", "meta"} <= keys, path
            assert pack["theta"].shape[0] in pack["sinogram"].shape, path
            assert pack["det_mask"].shape[0] == pack["sinogram"].shape[-1], path
            json.loads(str(pack["meta"]))


@pytest.mark.parametrize("algo,overrides,minimum_psnr", [
    ("fbp", [], 15.0),
    ("sirt", ["--set", "num_iterations=50"], 19.0),
    ("cg_tikhonov", [], 18.0),
])
def test_runner_on_parallel_toy_pack(tmp_path, algo, overrides, minimum_psnr):
    out = tmp_path / algo
    subprocess.run([sys.executable, str(ROOT / "examples/run.py"), "--algo", algo,
                    "--input", str(DATA / "toy_parallel_ct_128.npz"), "--output", str(out),
                    "--device", "cpu", *overrides], check=True, cwd=ROOT,
                   env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin"})
    summary = json.loads((out / "summary.json").read_text())
    assert summary["metrics"]["psnr_db"] > minimum_psnr
    assert np.load(out / "reconstruction.npy").shape == (128, 128)
    assert (out / "preview.png").stat().st_size > 0


def test_runner_list_is_json():
    proc = subprocess.run([sys.executable, str(ROOT / "examples/run.py"), "--list"],
                          check=True, capture_output=True, text=True, cwd=ROOT)
    listed = json.loads(proc.stdout)
    assert set(listed) == set(registry.ALGORITHMS)


def test_algorithm_cards_are_up_to_date():
    proc = subprocess.run([sys.executable, str(ROOT / "scripts/gen_algorithm_cards.py"), "--check"],
                          capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_runner_on_cone_beam_pack(tmp_path):
    pytest.importorskip("astra")
    out = tmp_path / "sirt_cone"
    subprocess.run([sys.executable, str(ROOT / "examples/run.py"), "--algo", "sirt",
                    "--input", str(DATA / "toy_cone_ct_3d.npz"), "--output", str(out),
                    "--device", "cuda", "--set", "num_iterations=30"], check=True, cwd=ROOT)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["metrics"]["psnr_db"] > 20
    assert np.load(out / "reconstruction.npy").shape == (64, 128, 128)
