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


def test_canonical_and_permissive_ddl_differ_only_by_check():
    """Verify the canonical and permissive DDL for each mfdb_* table
    are structurally identical modulo CHECK constraints (DATA-03 guardrail)."""
    import re
    for tn in schema._CANONICAL_TABLE_NAMES:
        permissive = schema._build_permissive_ddl(tn)
        canonical = schema._build_canonical_ddl(tn)

        # Same table name
        m_p = re.search(r"CREATE TABLE IF NOT EXISTS (\w+)", permissive)
        m_c = re.search(r"CREATE TABLE IF NOT EXISTS (\w+)", canonical)
        assert m_p and m_c and m_p.group(1) == m_c.group(1)

        # Same columns (ignoring CHECK clauses)
        col_re = re.compile(r"^\s+(\S+\s+\S+)")
        p_cols = {m.group(1) for line in permissive.split("\n") if (m := col_re.match(line))}
        c_cols = {m.group(1) for line in canonical.split("\n") if (m := col_re.match(line))}
        assert p_cols == c_cols, f"{tn}: columns differ between permissive and canonical"

        # Same table constraints
        p_constraints = {line.strip() for line in permissive.split("\n")
                         if line.strip().startswith(("PRIMARY", "UNIQUE", "FOREIGN"))}
        c_constraints = {line.strip() for line in canonical.split("\n")
                         if line.strip().startswith(("PRIMARY", "UNIQUE", "FOREIGN"))}
        assert p_constraints == c_constraints, f"{tn}: table constraints differ"

        # Canonical has CHECK, permissive does not
        assert "CHECK" not in permissive, f"{tn}: permissive must not contain CHECK"
        assert "CHECK" in canonical or tn in ("mfdb_object", "mfdb_parameter"), (
            f"{tn}: canonical should have CHECK constraints"
        )


def test_create_tables_no_duplicate_canonical_ddl():
    """All canonical mfdb_* DDL in CREATE_TABLES_SQL is generated from
    _CANONICAL_TABLE_DEFS — no hand-written duplicates (DATA-03 guardrail)."""
    import re
    for tn in schema._CANONICAL_TABLE_NAMES:
        expected = schema._build_permissive_ddl(tn)
        matches = [
            sql for sql in schema.CREATE_TABLES_SQL
            if isinstance(sql, str) and re.match(
                rf"CREATE TABLE IF NOT EXISTS {re.escape(tn)}\b", sql
            )
        ]
        assert len(matches) == 1, (
            f"{tn}: expected exactly 1 DDL in CREATE_TABLES_SQL, found {len(matches)}"
        )
        assert matches[0] == expected, f"{tn}: DDL mismatch with generated permissive form"


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
