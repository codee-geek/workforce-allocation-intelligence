import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _resolve_config_path(path: Path | None = None) -> Path:
    if path is not None:
        return path
    env = os.environ.get("WORKFORCE_CONFIG", "").strip()
    if env:
        p = Path(env)
        return p if p.is_absolute() else ROOT / p
    return ROOT / "config.yaml"


def load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = _resolve_config_path(path)
    with open(config_path) as f:
        return yaml.safe_load(f)
