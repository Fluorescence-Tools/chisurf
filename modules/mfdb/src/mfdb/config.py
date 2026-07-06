"""Runtime configuration for the standalone MFDB package."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime paths and identity defaults used by MFDB.

    Parameters
    ----------
    settings_dir : Path or None
        Base settings directory. Defaults to ``$MFDB_SETTINGS_DIR`` or
        ``~/.chisurf``.
    database_path : Path or None
        Explicit MFDB SQLite path. Defaults to ``settings_dir/flr/sample_management.db``.
    source_database_path : Path or None
        Optional curated source database path copied into the user database path.
    object_store_root : Path or None
        Explicit object-store root. Defaults to ``settings_dir/objects``.
    default_user_id : str or None
        Default local user id. When unset, ``$MFDB_DEFAULT_USER_ID`` or
        ``user_default`` is used.

    """

    settings_dir: Path | None = None
    database_path: Path | None = None
    source_database_path: Path | None = None
    object_store_root: Path | None = None
    default_user_id: str | None = None


_CONFIG = RuntimeConfig()


def configure_runtime(**kwargs: object) -> RuntimeConfig:
    """Update the process-local MFDB runtime configuration.

    Parameters
    ----------
    **kwargs
        Fields of :class:`RuntimeConfig` to update. Path-like values are
        normalized to :class:`~pathlib.Path`.

    Returns
    -------
    RuntimeConfig
        The updated runtime configuration.

    """
    global _CONFIG
    normalized: dict[str, object] = {}
    path_fields = {
        "settings_dir",
        "database_path",
        "source_database_path",
        "object_store_root",
    }
    for key, value in kwargs.items():
        if key in path_fields and value is not None:
            normalized[key] = Path(value).expanduser()
        elif value is not None:
            normalized[key] = value
    _CONFIG = replace(_CONFIG, **normalized)
    return _CONFIG


def reset_runtime_config() -> RuntimeConfig:
    """Reset process-local runtime configuration to environment/default values."""
    global _CONFIG
    _CONFIG = RuntimeConfig()
    return _CONFIG


def get_runtime_config() -> RuntimeConfig:
    """Return the process-local runtime configuration."""
    return _CONFIG


def configured_settings_dir() -> Path:
    """Return the configured settings directory."""
    if _CONFIG.settings_dir is not None:
        return _CONFIG.settings_dir
    env_value = os.environ.get("MFDB_SETTINGS_DIR")
    if env_value:
        return Path(os.path.expandvars(os.path.expanduser(env_value)))
    return Path.home() / ".chisurf"


def configured_database_path() -> Path | None:
    """Return an explicitly configured database path, if any."""
    if _CONFIG.database_path is not None:
        return _CONFIG.database_path
    env_value = os.environ.get("MFDB_DATABASE_PATH")
    if env_value:
        return Path(os.path.expandvars(os.path.expanduser(env_value)))
    return None


def configured_source_database_path() -> Path | None:
    """Return an explicitly configured source database path, if any."""
    if _CONFIG.source_database_path is not None:
        return _CONFIG.source_database_path
    env_value = os.environ.get("MFDB_SOURCE_DATABASE_PATH")
    if env_value:
        return Path(os.path.expandvars(os.path.expanduser(env_value)))
    return None


def configured_object_store_root() -> Path | None:
    """Return an explicitly configured object-store root, if any."""
    if _CONFIG.object_store_root is not None:
        return _CONFIG.object_store_root
    env_value = os.environ.get("MFDB_OBJECT_STORE_ROOT")
    if env_value:
        return Path(os.path.expandvars(os.path.expanduser(env_value)))
    return None


def configured_default_user_id() -> str:
    """Return the configured default user id."""
    if _CONFIG.default_user_id:
        return _CONFIG.default_user_id
    env_value = os.environ.get("MFDB_DEFAULT_USER_ID")
    return env_value or "user_default"
