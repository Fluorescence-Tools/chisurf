"""Tests for PRD-19: MFDB collapse to one flrCIF-rooted family.

Verifies that:
1. Model vocabulary constants match the .dic enumerations.
2. reconcile_schema creates missing extension tables and columns.
3. _drop_legacy_tables removes all fdb_* and duplicate mfdb_* tables.
4. A fresh MFDatabase has no fdb_*, mfdb_sample, or mfdb_experiment tables.
5. Sample creation writes to flr_sample and the ORM graph is consistent.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile

import pytest

from mfdb.models import (
    ARTIFACT_KINDS,
    OPERATION_TYPES,
    DIRECTIONS,
    RELATIONSHIP_TYPES,
    STATUS_VALUES,
    VALIDATION_STATUS_VALUES,
    STORAGE_MODES,
    PARAMETER_TYPES,
    LIFECYCLE_STATUSES,
    DATA_FORMATS,
)
from mfdb.pdbx_metadata import MmcifDictionary
from mfdb.repository import MFDatabase
from mfdb.schema import (
    FRESH_DB_SCHEMA_SQL,
    CREATE_TABLES_SQL,
    _drop_legacy_tables,
    _get_dict_ddl,
)
from mfdb.schema_from_dictionary import reconcile_schema
from mfdb.sample_manager import create_sample, get_sample_full_description
from mfdb.models import SampleDefinition, EntityDefinition, ProbeDefinition


_DIC = MmcifDictionary.load_bundled()

_FIELD_TO_ENUM_KEY = {
    "artifact_kind": ARTIFACT_KINDS,
    "data_format": DATA_FORMATS,
    "operation_type": OPERATION_TYPES,
    "direction": DIRECTIONS,
    "relationship_type": RELATIONSHIP_TYPES,
    "status": STATUS_VALUES,
    "validation_status": VALIDATION_STATUS_VALUES,
    "storage_mode": STORAGE_MODES,
    "parameter_type": PARAMETER_TYPES,
    "lifecycle_status": LIFECYCLE_STATUSES,
}

_DIC_ITEM = {
    "artifact_kind": "_mfdb_artifact.artifact_kind",
    "data_format": "_mfdb_artifact.data_format",
    "operation_type": "_mfdb_operation.operation_type",
    "direction": "_mfdb_operation_artifact.direction",
    "relationship_type": "_mfdb_edge.relationship_type",
    "status": "_mfdb_operation.status",
    "validation_status": "_mfdb_operation.validation_status",
    "storage_mode": "_mfdb_object.storage_mode",
    "parameter_type": "_mfdb_parameter.parameter_type",
    "lifecycle_status": "_mfdb_branch.lifecycle_status",
}


# ── 1. Vocab from dictionary ─────────────────────────────────────────────

@pytest.mark.parametrize("field", sorted(_FIELD_TO_ENUM_KEY))
def test_vocabulary_matches_dictionary(field: str) -> None:
    """All model vocabulary constants match the .dic enumerations."""
    model_values = set(v.lower() if isinstance(v, str) else v for v in _FIELD_TO_ENUM_KEY[field])
    dic_values = set(_DIC.get_enumerations(_DIC_ITEM[field]))
    assert model_values == dic_values, (
        f"Mismatch for {field}: "
        f"model has {model_values - dic_values} extra, "
        f"dictionary has {dic_values - model_values} extra"
    )


# ── 2. reconcile_schema ──────────────────────────────────────────────────

def _create_conn() -> tuple[sqlite3.Connection, str]:
    """Return (conn, path) for a temporary SQLite database."""
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test_reconcile.db")
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn, path


def test_reconcile_schema_creates_missing_extension_tables() -> None:
    """reconcile_schema adds extension tables that are absent from the live DB."""
    conn, path = _create_conn()
    try:
        # Create only the base setup table (parent for FKs)
        conn.execute("CREATE TABLE IF NOT EXISTS mfdb_setup (setup_id TEXT PRIMARY KEY, name TEXT)")
        conn.commit()

        # Verify extension tables do not exist yet
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'mfdb_%'"
        )
        existing = {row[0] for row in cursor.fetchall()}
        assert "mfdb_operation" not in existing

        # Run reconcile_schema
        reconcile_schema(conn, _DIC)

        # Verify extension tables were created
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'mfdb_%'"
        )
        after = {row[0] for row in cursor.fetchall()}
        assert "mfdb_operation" in after, "mfdb_operation table not created"
        assert "mfdb_operation_artifact" in after
        assert "mfdb_edge" in after
        assert "mfdb_parameter" in after
        assert "mfdb_object" in after
        assert "mfdb_artifact" in after
        assert "mfdb_branch" in after
    finally:
        conn.close()
        os.remove(path)


def test_reconcile_schema_adds_missing_columns() -> None:
    """reconcile_schema adds columns that are missing from an existing table."""
    conn, path = _create_conn()
    try:
        # Create mfdb_setup table missing the name column
        conn.execute("""
            CREATE TABLE IF NOT EXISTS mfdb_setup (
                setup_id TEXT PRIMARY KEY,
                version TEXT
            )
        """)
        conn.commit()

        # Run reconcile_schema
        reconcile_schema(conn, _DIC)

        # Verify columns were added
        cursor = conn.execute("PRAGMA table_info(mfdb_setup)")
        cols = {row[1] for row in cursor.fetchall()}
        assert "name" in cols, "name column not added by reconcile_schema"
        assert "version" in cols, "existing version column removed"
    finally:
        conn.close()
        os.remove(path)


# ── 3. Drop legacy tables ────────────────────────────────────────────────

def test_drop_legacy_tables_removes_fdb_tables() -> None:
    """_drop_legacy_tables removes all fdb_* and duplicate mfdb_* tables."""
    conn, path = _create_conn()
    try:
        # Create fdb_ tables and duplicate mfdb_ tables
        legacy_tables = [
            "fdb_raw_data", "fdb_processing_run", "fdb_processing_input",
            "fdb_processed_data", "fdb_provenance_edge", "fdb_setup_definition",
            "fdb_analysis_run", "fdb_analysis_parameter", "fdb_audit_log",
            "fdb_setup", "fdb_artifact", "fdb_operation",
            "fdb_operation_artifact", "fdb_edge", "fdb_parameter",
            "mfdb_sample", "mfdb_experiment",
        ]
        for tbl in legacy_tables:
            conn.execute(f"CREATE TABLE IF NOT EXISTS {tbl} (id TEXT PRIMARY KEY)")
        conn.execute("CREATE TABLE IF NOT EXISTS mfdb_setup (setup_id TEXT PRIMARY KEY, name TEXT)")

        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        before = {row[0] for row in cursor.fetchall()}
        for tbl in legacy_tables:
            assert tbl in before, f"{tbl} should exist before drop"

        # Run _drop_legacy_tables
        _drop_legacy_tables(conn)

        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        after = {row[0] for row in cursor.fetchall()}
        for tbl in legacy_tables:
            assert tbl not in after, f"{tbl} should have been dropped"
        assert "mfdb_setup" in after, "non-legacy table should survive"
    finally:
        conn.close()
        os.remove(path)


# ── 4. Fresh DB has no legacy tables ─────────────────────────────────────

def test_fresh_database_has_no_legacy_tables() -> None:
    """A fresh MFDatabase created via FRESH_DB_SCHEMA_SQL has no fdb_* or
    duplicate mfdb_* tables."""
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test_fresh.db")
    db = MFDatabase(path)
    try:
        cursor = db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        table_names = {row[0] for row in cursor.fetchall()}
        # No legacy tables
        for prefix in ["fdb_", "mfdb_sample", "mfdb_experiment"]:
            for t in table_names:
                assert not t.startswith(prefix), (
                    f"Legacy table {t} should not exist in fresh database"
                )
        # flr_sample should exist and be the canonical sample table
        assert "flr_sample" in table_names
        # mfdb extension tables should exist
        mfdb_tables = [t for t in table_names if t.startswith("mfdb_")]
        assert len(mfdb_tables) > 0, "No mfdb_ extension tables found"
    finally:
        db.close()
        os.remove(path)


# ── 5. Sample graph round-trip via flr_sample ────────────────────────────

def test_sample_roundtrip_via_flr_sample() -> None:
    """Create a sample via ORM and verify it's readable from flr_sample."""
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test_roundtrip.db")
    db = MFDatabase(path)
    try:
        definition = SampleDefinition(
            name="roundtrip_test",
            description="PRD-19 round-trip test sample",
            entities=[
                EntityDefinition(
                    name="entity_1",
                    entity_type="protein",
                    sequence="MKTAYIAKQRQ",
                ),
            ],
            probes=[
                ProbeDefinition(
                    name="Cy3B",
                    position=50,
                    seq_id=50,
                    comp_id="CYS",
                    asym_id="A",
                    entity_index=0,
                ),
            ],
            fret_pairs=[],
        )
        sample_id = create_sample(db, definition)

        # Read back from flr_sample directly
        row = db.conn.execute(
            "SELECT sample_id, description, details FROM flr_sample WHERE sample_id = ?",
            (sample_id,),
        ).fetchone()
        assert row is not None, "Sample not found in flr_sample"
        assert row["description"] == "roundtrip_test"

        # Read back via the raw-SQL full-description reader
        desc = get_sample_full_description(db, sample_id)
        assert desc is not None
        assert desc["sample_id"] == sample_id
        assert len(desc["entities"]) == 1
        assert desc["entities"][0]["entity_id"] == "entity_1"
    finally:
        db.close()
        os.remove(path)
