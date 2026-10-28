from pathlib import Path

from inv_framework.applications.nasn_ct.config import inclusive_range, load_config


ROOT = Path(__file__).resolve().parents[2] / "inv_framework/applications/nasn_ct"


def test_locked_geometry_and_scan_counts():
    target = load_config(ROOT / "configs/target_legacy.yaml")
    assert list(inclusive_range(target["slice_range_1based"])) == list(range(550, 1751))
    assert target["dds_geometry"]["center"] == 1006.283203125
    expected = {1: 1356, 2: 1346, 3: 1336, 4: 1346}
    for scan, count in expected.items():
        config = load_config(ROOT / f"configs/fullct_scan{scan}.yaml")
        assert len(inclusive_range(config["tomo_range_1based"])) == count
        assert len(inclusive_range(config["slice_range_1based"])) == 1001


def test_split_is_locked():
    config = load_config(ROOT / "configs/pmf_2048.yaml")
    assert config["split"] == {
        "train_scans": [1, 2, 3], "train_slices": 3003,
        "val_scans": [4], "val_slices": 1001,
    }
