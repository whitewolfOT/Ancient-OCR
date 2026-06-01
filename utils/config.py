"""Load and expose config.yaml as a nested dotted-access object."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class _Config:
    """Thin wrapper around a dict that allows dotted attribute access."""

    def __init__(self, data: dict) -> None:
        for key, value in data.items():
            setattr(self, key, _Config(value) if isinstance(value, dict) else value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __repr__(self) -> str:  # pragma: no cover
        return f"_Config({self.__dict__!r})"


_config_singleton: _Config | None = None
_config_path_used: str | None = None

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config(path: str | Path | None = None) -> _Config:
    """Load config.yaml and return a dotted-access Config object.

    Re-reads the file if path differs from the previously loaded path.
    """
    global _config_singleton, _config_path_used

    resolved = str(Path(path) if path else _DEFAULT_CONFIG_PATH)

    if _config_singleton is not None and _config_path_used == resolved:
        return _config_singleton

    if not os.path.exists(resolved):
        raise FileNotFoundError(
            f"config.yaml not found at {resolved}. "
            "Make sure you are running from the project root."
        )

    with open(resolved, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    _config_singleton = _Config(raw)
    _config_path_used = resolved
    return _config_singleton


def get_config() -> _Config:
    """Return the singleton config, loading from the default path if needed."""
    if _config_singleton is None:
        return load_config()
    return _config_singleton


def reload_config(path: str | Path | None = None) -> _Config:
    """Force a reload (useful in tests)."""
    global _config_singleton, _config_path_used
    _config_singleton = None
    _config_path_used = None
    return load_config(path)
