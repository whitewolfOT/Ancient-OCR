"""Disk-backed cache keyed by content + config hash."""

from __future__ import annotations

import hashlib
import json
from typing import Any

_cache_instance: Any = None  # diskcache.Cache, loaded lazily
_cache_enabled: bool | None = None


def _get_cache():
    global _cache_instance, _cache_enabled

    if _cache_enabled is False:
        return None

    if _cache_instance is not None:
        return _cache_instance

    try:
        from utils.config import get_config

        cfg = get_config()
        enabled = getattr(cfg.cache, "enabled", True)
        path = getattr(cfg.cache, "path", ".cache/")
    except Exception:
        enabled, path = True, ".cache/"

    if not enabled:
        _cache_enabled = False
        return None

    try:
        import diskcache

        _cache_instance = diskcache.Cache(path)
        _cache_enabled = True
        return _cache_instance
    except ImportError:
        from utils.logging import get_logger

        get_logger(__name__).warning("diskcache not installed; caching disabled")
        _cache_enabled = False
        return None


def make_key(namespace: str, content: str | bytes, config_hash: str = "") -> str:
    """Build a deterministic cache key."""
    raw = (content if isinstance(content, bytes) else content.encode()) + config_hash.encode()
    digest = hashlib.sha256(raw).hexdigest()
    return f"{namespace}:{digest}"


def get(namespace: str, key: str) -> Any | None:
    """Return cached value or None if absent / cache disabled."""
    cache = _get_cache()
    if cache is None:
        return None
    full_key = f"{namespace}:{key}"
    return cache.get(full_key)


def set(namespace: str, key: str, value: Any, ttl: int | None = None) -> None:
    """Store value; silently no-ops if cache is disabled."""
    cache = _get_cache()
    if cache is None:
        return
    full_key = f"{namespace}:{key}"
    if ttl is None:
        try:
            from utils.config import get_config

            ttl = getattr(get_config().cache, "ttl", 86400)
        except Exception:
            ttl = 86400
    cache.set(full_key, value, expire=ttl)


def config_hash(cfg_path: str = "config.yaml") -> str:
    """Return a short hash of config.yaml for cache invalidation."""
    try:
        with open(cfg_path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()[:16]
    except OSError:
        return "nocfg"


def invalidate(namespace: str, key: str) -> None:
    """Remove a single entry."""
    cache = _get_cache()
    if cache is None:
        return
    full_key = f"{namespace}:{key}"
    cache.delete(full_key)
