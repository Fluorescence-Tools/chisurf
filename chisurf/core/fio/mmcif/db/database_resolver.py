"""Resolve and protect the ChiSurf fluorescence sample database."""

from __future__ import annotations

import logging
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from chisurf.core.settings.path_utils import get_path

from . import schema

logger = logging.getLogger(__name__)

SOURCE_DB_NAME = "sample_management.db"
USER_DB_RELATIVE = Path("flr") / SOURCE_DB_NAME


def source_database_path() -> Path:
    """Return the curated source database shipped with ChiSurf."""
    return Path(__file__).resolve().parent / SOURCE_DB_NAME


def user_database_path() -> Path:
    """Return the per-user editable sample database path."""
    return get_path("settings") / USER_DB_RELATIVE


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


def backup_database_before_migration(db_path: Path, target_version: int) -> Optional[Path]:
    """Back up a database before schema migration if migration is needed.

    Parameters
    ----------
    db_path : pathlib.Path
        Database path.
    target_version : int
        Target schema version.

    Returns
    -------
    pathlib.Path or None
        Backup path, or None if no backup was needed.
    """
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
    """Copy source database to user path atomically."""
    tmp_path = user_path.with_suffix(user_path.suffix + ".tmp")
    try:
        shutil.copy2(source_path, tmp_path)
        tmp_path.replace(user_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _create_empty_database(path: Path) -> None:
    """Create an empty database using the current schema."""
    conn = sqlite3.connect(str(path))
    try:
        schema.migrate_schema(conn)
        conn.commit()
    finally:
        conn.close()


def _read_schema_version(conn: sqlite3.Connection) -> Optional[int]:
    """Read schema version if the version table exists."""
    try:
        row = conn.execute("SELECT version FROM _schema_version").fetchone()
    except sqlite3.OperationalError:
        return None
    return int(row[0]) if row is not None else None
