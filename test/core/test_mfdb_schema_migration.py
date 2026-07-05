"""Tests for the MFDB schema migration waterfall (DATA-02)."""

from __future__ import annotations

import sqlite3
import tempfile
import os

import pytest

from chisurf.core.mfdb import schema


def test_fresh_db_gets_full_schema():
    conn = sqlite3.connect(":memory:")
    report = schema.migrate_schema(conn)
    assert report is not None
    assert report.from_version == 0
    assert report.to_version == schema.SCHEMA_VERSION
    assert schema.get_schema_version(conn) == schema.SCHEMA_VERSION
    conn.close()


def test_already_current_db_returns_none():
    conn = sqlite3.connect(":memory:")
    schema.migrate_schema(conn)
    report = schema.migrate_schema(conn)
    assert report is None
    assert schema.get_schema_version(conn) == schema.SCHEMA_VERSION
    conn.close()


def test_migration_waterfall_resumes_from_version():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE IF NOT EXISTS _schema_version (version INTEGER)")
    conn.execute("INSERT INTO _schema_version (version) VALUES (1)")
    conn.execute("CREATE TABLE IF NOT EXISTS mfdb_schema_version (version INTEGER)")
    conn.execute("INSERT INTO mfdb_schema_version (version) VALUES (1)")

    report = schema.migrate_schema(conn)
    assert report is not None
    assert report.from_version == 1
    assert report.to_version == schema.SCHEMA_VERSION
    assert schema.get_schema_version(conn) == schema.SCHEMA_VERSION
    conn.close()


def test_migration_uses_registered_steps():
    assert len(schema.MIGRATIONS) >= 1
    for version, fn in schema.MIGRATIONS.items():
        assert callable(fn)
        assert version <= schema.SCHEMA_VERSION


def test_persistent_db_reopened_is_idempotent():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        conn = sqlite3.connect(db_path)
        schema.migrate_schema(conn)
        ver = schema.get_schema_version(conn)
        conn.close()

        conn2 = sqlite3.connect(db_path)
        report = schema.migrate_schema(conn2)
        assert report is None
        assert schema.get_schema_version(conn2) == ver
        conn2.close()
    finally:
        os.unlink(db_path)


def test_new_migration_step_applies_on_bump():
    """Simulate bumping SCHEMA_VERSION and adding a new migration step."""
    orig_version = schema.SCHEMA_VERSION

    try:
        dummy_called = False

        def _dummy_v41(_conn):
            nonlocal dummy_called
            dummy_called = True

        # Bump SCHEMA_VERSION to 41 and register v41 step
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            # Create a v40 DB on disk
            conn = sqlite3.connect(db_path)
            schema.migrate_schema(conn)
            assert schema.get_schema_version(conn) == 40
            conn.close()

            # Now bump version and add migration
            schema.SCHEMA_VERSION = 41
            schema.MIGRATIONS[41] = _dummy_v41

            conn2 = sqlite3.connect(db_path)
            report = schema.migrate_schema(conn2)
            assert report is not None
            assert report.to_version == 41
            assert dummy_called, "v41 migration was not called"
            assert schema.get_schema_version(conn2) == 41
            conn2.close()
        finally:
            os.unlink(db_path)
    finally:
        schema.SCHEMA_VERSION = orig_version
        schema.MIGRATIONS.pop(41, None)
