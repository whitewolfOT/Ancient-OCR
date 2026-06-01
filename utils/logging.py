"""Central logger factory for the Ancient-OCR pipeline."""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

_LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}

_configured = False


def _configure_root(level: str = "INFO", structured: bool = False) -> None:
    global _configured
    if _configured:
        return

    numeric = _LOG_LEVELS.get(level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(numeric)

    if structured:
        fmt = "%(asctime)s level=%(levelname)s logger=%(name)s %(message)s"
    else:
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%dT%H:%M:%S"))

    root = logging.getLogger()
    root.setLevel(numeric)
    if not root.handlers:
        root.addHandler(handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, configuring the root handler on first call."""
    if not _configured:
        try:
            from utils.config import get_config

            cfg = get_config()
            level = getattr(cfg.logging, "level", "INFO")
            structured = getattr(cfg.logging, "structured", False)
        except Exception:
            level, structured = "INFO", False
        _configure_root(level, structured)

    return logging.getLogger(name)
