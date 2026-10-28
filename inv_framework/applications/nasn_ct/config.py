"""Configuration helpers with explicit, relocatable paths."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    value["_config_path"] = str(path)
    return value


def inclusive_range(value) -> range:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("range must be [first, last], both inclusive")
    first, last = map(int, value)
    if first < 1 or last < first:
        raise ValueError(f"invalid inclusive range: {value}")
    return range(first, last + 1)
