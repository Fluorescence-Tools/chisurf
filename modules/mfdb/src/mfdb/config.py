"""Runtime configuration for the standalone MFDB package."""

from __future__ import annotations

import os
from collections.abc import Callable
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

# Optional host-supplied callable resolving the default user id live. A host
# application (e.g. ChiSurf) whose own settings are the source of truth injects
# this so runtime changes propagate without MFDB importing the host. MFDB stays
# standalone: the resolver is host-agnostic and purely optional.
_DEFAULT_USER_ID_RESOLVER: Callable[[], str | None] | None = None


def set_default_user_id_resolver(resolver: Callable[[], str | None] | None) -> None:
    """Register (or clear) a live resolver for the default user id.

    Parameters
    ----------
    resolver : callable returning str or None, or None
        Called by :func:`configured_default_user_id` when no explicit
        ``default_user_id`` override is set on the runtime config. Pass ``None``
        to clear a previously registered resolver.

    """
    global _DEFAULT_USER_ID_RESOLVER
    _DEFAULT_USER_ID_RESOLVER = resolver


# Optional host-supplied callable resolving the active authentication config
# (provider selection + per-provider settings such as the LDAP block). Mirrors
# the default-user-id resolver so a host application can supply live auth config
# without MFDB importing the host. See :func:`configured_auth_config`.
_AUTH_CONFIG_RESOLVER: Callable[[], dict | None] | None = None


def set_auth_config_resolver(resolver: Callable[[], dict | None] | None) -> None:
    """Register (or clear) a live resolver for the authentication config.

    The resolver returns a dict like ``{"auth_provider": "ldap", "ldap": {...}}``
    or ``None`` for the default (local). Pass ``None`` to clear it.
    """
    global _AUTH_CONFIG_RESOLVER
    _AUTH_CONFIG_RESOLVER = resolver


def _ldap_env_config() -> dict:
    """Build an LDAP config block from ``MFDB_LDAP_*`` environment variables."""
    import json as _json

    def _split(name: str) -> list[str]:
        raw = os.environ.get(name, "")
        return [p for p in (s.strip() for s in raw.split(",")) if p]

    cfg: dict = {
        "host": os.environ.get("MFDB_LDAP_HOST"),
        "base_dn": os.environ.get("MFDB_LDAP_BASE_DN"),
        "bind_dn": os.environ.get("MFDB_LDAP_BIND_DN"),
        "bind_password": os.environ.get("MFDB_LDAP_BIND_PASSWORD"),
    }
    if os.environ.get("MFDB_LDAP_PORT"):
        cfg["port"] = int(os.environ["MFDB_LDAP_PORT"])
    if os.environ.get("MFDB_LDAP_USE_SSL"):
        cfg["use_ssl"] = os.environ["MFDB_LDAP_USE_SSL"].lower() in ("1", "true", "yes")
    for key, env in (
        ("user_filter", "MFDB_LDAP_USER_FILTER"),
        ("uid_attr", "MFDB_LDAP_UID_ATTR"),
        ("mail_attr", "MFDB_LDAP_MAIL_ATTR"),
        ("name_attr", "MFDB_LDAP_NAME_ATTR"),
        ("memberof_attr", "MFDB_LDAP_MEMBEROF_ATTR"),
    ):
        if os.environ.get(env):
            cfg[key] = os.environ[env]
    if os.environ.get("MFDB_LDAP_GROUP_MAP"):
        try:
            cfg["group_map"] = _json.loads(os.environ["MFDB_LDAP_GROUP_MAP"])
        except ValueError:
            pass
    if _split("MFDB_LDAP_ADMIN_GROUPS"):
        cfg["admin_groups"] = _split("MFDB_LDAP_ADMIN_GROUPS")
    return {k: v for k, v in cfg.items() if v is not None}


def configured_auth_config() -> dict | None:
    """Return the active auth config: host resolver → ``MFDB_AUTH_PROVIDER`` env → None.

    ``None`` means the default ``local`` provider. A returned dict carries at
    least ``auth_provider`` and, for LDAP, an ``ldap`` block.
    """
    if _AUTH_CONFIG_RESOLVER is not None:
        try:
            resolved = _AUTH_CONFIG_RESOLVER()
        except Exception:
            resolved = None
        if resolved:
            return resolved
    provider = os.environ.get("MFDB_AUTH_PROVIDER")
    if provider:
        config: dict = {"auth_provider": provider}
        if provider.lower() == "ldap":
            config["ldap"] = _ldap_env_config()
        return config
    return None


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
    """Return the configured default user id.

    Resolution order: an explicit ``default_user_id`` on the runtime config, then
    a host-registered live resolver (see :func:`set_default_user_id_resolver`),
    then ``$MFDB_DEFAULT_USER_ID``, then ``"user_default"``.
    """
    if _CONFIG.default_user_id:
        return _CONFIG.default_user_id
    if _DEFAULT_USER_ID_RESOLVER is not None:
        try:
            resolved = _DEFAULT_USER_ID_RESOLVER()
        except Exception:
            resolved = None
        if resolved:
            return resolved
    env_value = os.environ.get("MFDB_DEFAULT_USER_ID")
    return env_value or "user_default"
