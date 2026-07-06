"""Resolve the canonical MFDB sample database path and object store root."""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

from mfdb.config import (
    configured_database_path,
    configured_object_store_root,
    configured_settings_dir,
    configured_source_database_path,
)

logger = logging.getLogger(__name__)

SOURCE_DB_NAME = "sample_management.db"
USER_DB_RELATIVE = Path("flr") / SOURCE_DB_NAME


def source_database_path() -> Path:
    """Return the curated source database shipped with MFDB."""
    configured = configured_source_database_path()
    if configured is not None:
        return configured
    data_dir = Path(__file__).resolve().parent / "data"
    packaged_source = data_dir / SOURCE_DB_NAME
    if packaged_source.exists():
        return packaged_source
    return data_dir / "example.db"


def user_database_path() -> Path:
    """Return the per-user editable sample database path."""
    configured = configured_database_path()
    if configured is not None:
        return configured
    return configured_settings_dir() / USER_DB_RELATIVE


def object_store_root() -> Path:
    """Return the shared object store root path.

    The object store is shared across all users on the same machine.
    By default it is located at ``{settings_dir}/objects/``. Configure
    ``mfdb.object_store.root`` in ``settings_chisurf.yaml`` to override
    it. Relative configured paths are resolved relative to the settings
    directory.
    """
    settings_dir = configured_settings_dir()
    root = configured_object_store_root()
    if root is None:
        return settings_dir / "objects"
    root = Path(os.path.expandvars(os.path.expanduser(str(root))))
    if not root.is_absolute():
        root = settings_dir / root
    return root


def resolve_database_path() -> Path:
    """Ensure the user sample database exists and return its path."""
    source_path = source_database_path()
    user_path = user_database_path()
    user_path.parent.mkdir(parents=True, exist_ok=True)

    if not user_path.exists():
        if source_path.exists():
            _copy_database(source_path, user_path)
        else:
            _create_empty_database(user_path)
    return user_path


def backup_database(db_path: Path) -> Path:
    """Create an immediate backup copy of a database."""
    db_path = Path(db_path)
    if str(db_path) == ":memory:":
        raise ValueError("Cannot back up an in-memory database")
    backup_dir = db_path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = backup_dir / f"{db_path.name}.{stamp}.bak"
    shutil.copy2(db_path, backup_path)
    logger.info("Backed up sample database: %s", backup_path)
    return backup_path


def backup_database_before_migration(db_path: Path, target_version: int) -> Path | None:
    """Back up a database before schema migration if migration is needed."""
    db_path = Path(db_path)
    if str(db_path) == ":memory:":
        return None

    conn = sqlite3.connect(str(db_path))
    try:
        try:
            conn.execute("PRAGMA wal_checkpoint(FULL)")
        except sqlite3.OperationalError:
            pass
        version = _read_schema_version(conn)
        if version is not None and version >= target_version:
            return None
    finally:
        conn.close()

    backup_dir = db_path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = backup_dir / f"{db_path.name}.{stamp}.bak"
    shutil.copy2(db_path, backup_path)
    logger.info("Backed up sample database before migration: %s", backup_path)
    return backup_path


def _copy_database(source_path: Path, user_path: Path) -> None:
    tmp_path = user_path.with_suffix(user_path.suffix + ".tmp")
    try:
        shutil.copy2(source_path, tmp_path)
        tmp_path.replace(user_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _create_empty_database(path: Path) -> None:
    from mfdb import schema

    conn = sqlite3.connect(str(path))
    try:
        schema.migrate_schema(conn)
        conn.commit()
    finally:
        conn.close()


def _read_schema_version(conn: sqlite3.Connection) -> int | None:
    try:
        row = conn.execute("SELECT version FROM _schema_version").fetchone()
    except sqlite3.OperationalError:
        return None
    return int(row[0]) if row is not None else None
