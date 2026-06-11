"""Tests for fluorescence sample database path resolution and backup."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

from chisurf.core.fio.mmcif.db import database_resolver, schema
from chisurf.core.fio.mmcif.db.database_resolver import backup_database_before_migration


def test_backup_before_migration_copies_existing_database():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "sample_management.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE _schema_version (version INTEGER)")
        conn.execute("INSERT INTO _schema_version VALUES (8)")
        conn.commit()
        conn.close()

        backup_path = backup_database_before_migration(db_path, schema.SCHEMA_VERSION)

        assert backup_path is not None
        assert backup_path.exists()
        assert backup_path.read_bytes() == db_path.read_bytes()


def test_copy_source_to_user_path(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "src"
        user_root = root / "user"
        source.mkdir()
        user_root.mkdir()
        source_db = source / "sample_management.db"
        user_db = user_root / "flr" / "sample_management.db"
        source_db.write_bytes(b"curated-db")

        monkeypatch.setattr(database_resolver, "source_database_path", lambda: source_db)
        monkeypatch.setattr(database_resolver, "user_database_path", lambda: user_db)

        resolved = database_resolver.resolve_database_path()

        assert resolved == user_db
        assert user_db.read_bytes() == b"curated-db"
