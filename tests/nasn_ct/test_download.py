import hashlib
from pathlib import Path

from inv_framework.applications.nasn_ct.data.download import read_manifest, selected_archives, sha256


ROOT = Path(__file__).resolve().parents[2] / "inv_framework/applications/nasn_ct"


def test_profiles_are_complete_and_unique():
    manifest = read_manifest(ROOT / "manifests/data-v0.1.0.json")
    assert len(selected_archives(manifest, "quickstart")) == 1
    assert len(selected_archives(manifest, "target")) == 4
    assert len(selected_archives(manifest, "training")) == 4
    assert len(selected_archives(manifest, "all")) == 9
    assert len(selected_archives(manifest, "target", 3)) == 1


def test_sha256(tmp_path):
    path = tmp_path / "value"
    path.write_bytes(b"nasn")
    assert sha256(path) == hashlib.sha256(b"nasn").hexdigest()
