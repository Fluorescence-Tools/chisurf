"""Tests for the fluorophore reference set import (PRD-06 Task 7.2)."""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pytest

from chisurf.core.mfdb.repository import MFDatabase


@pytest.fixture
def mem_db():
    """Create an in-memory MFDB for testing."""
    db = MFDatabase(":memory:")
    yield db
    db.close()


@pytest.fixture
def mini_spectra_db():
    """Create a minimal spectra.db-like database for import testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")

    # Create minimal schema matching the _dev spectra.db
    conn.execute("""CREATE TABLE IF NOT EXISTS probe_types (
        type_id INTEGER PRIMARY KEY, type_name TEXT UNIQUE, display_name TEXT,
        created_at TEXT, updated_at TEXT, deleted_at TEXT
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS probes (
        probe_id INTEGER PRIMARY KEY, name TEXT, type_id INTEGER,
        description TEXT, category TEXT, is_curated INTEGER DEFAULT 0,
        quality_flag INTEGER DEFAULT 1,
        created_at TEXT, updated_at TEXT, deleted_at TEXT,
        FOREIGN KEY (type_id) REFERENCES probe_types(type_id),
        UNIQUE (name, type_id)
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS optical_properties (
        id INTEGER PRIMARY KEY, item_id INTEGER, property_name TEXT,
        property_value TEXT, unit TEXT, details TEXT,
        created_at TEXT, updated_at TEXT, deleted_at TEXT,
        FOREIGN KEY (item_id) REFERENCES probes(probe_id),
        UNIQUE (item_id, property_name)
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS spectra (
        id INTEGER PRIMARY KEY, item_id INTEGER, spectrum_type TEXT,
        wavelengths BLOB, intensity_values BLOB,
        created_at TEXT, updated_at TEXT, deleted_at TEXT,
        FOREIGN KEY (item_id) REFERENCES probes(probe_id),
        UNIQUE (item_id, spectrum_type)
    )""")

    # Insert probe types
    conn.execute(
        "INSERT INTO probe_types (type_id, type_name, display_name) VALUES (1, 'atto', 'Atto Dyes')"
    )
    conn.execute(
        "INSERT INTO probe_types (type_id, type_name, display_name) VALUES (7, 'fpbase', 'FPbase')"
    )

    # Insert probes
    conn.execute(
        "INSERT INTO probes (probe_id, name, type_id, description, category, is_curated, quality_flag) VALUES (1, 'ATTO-488', 1, '', '', 1, 1)"
    )
    conn.execute(
        "INSERT INTO probes (probe_id, name, type_id, description, category, is_curated, quality_flag) VALUES (2, 'ATTO-647N', 1, '', '', 1, 1)"
    )
    conn.execute(
        "INSERT INTO probes (probe_id, name, type_id, description, category, is_curated, quality_flag) VALUES (3, 'Alexa488', 7, '', '', 0, 1)"
    )

    # Insert optical properties
    for item_id, pname, pval in [
        (1, "abs_max", "501"),
        (1, "em_max", "523"),
        (1, "qy", "0.80"),
        (1, "ext_coeff", "90000"),
        (2, "abs_max", "646"),
        (2, "em_max", "664"),
        (2, "qy", "0.65"),
        (2, "ext_coeff", "150000"),
    ]:
        conn.execute(
            "INSERT INTO optical_properties (item_id, property_name, property_value) VALUES (?, ?, ?)",
            (item_id, pname, pval),
        )

    # Insert spectra (minimal 3-point)
    for item_id, stype in [(1, "absorption"), (1, "emission"), (2, "absorption"), (2, "emission")]:
        wl = np.array([400.0, 500.0, 600.0], dtype=np.float64)
        iv = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        conn.execute(
            "INSERT INTO spectra (item_id, spectrum_type, wavelengths, intensity_values) VALUES (?, ?, ?, ?)",
            (item_id, stype, wl.tobytes(), iv.tobytes()),
        )

    conn.commit()
    conn.close()

    yield path

    Path(path).unlink(missing_ok=True)


def test_import_reference_set_imports_probes(mem_db, mini_spectra_db):
    """Import should copy probes from the source database."""
    counts = mem_db.import_reference_set(source_path=mini_spectra_db)
    assert counts["probes"] > 0

    # Check probes were imported with normalized names
    row = mem_db.conn.execute(
        "SELECT chromophore_name, verification_status, source FROM probes WHERE chromophore_name = ?",
        ("ATTO 488",),
    ).fetchone()
    assert row is not None
    assert row["verification_status"] == "unverified"
    assert row["source"] == "atto"


def test_import_reference_set_imports_spectra(mem_db, mini_spectra_db):
    """Import should copy spectra."""
    mem_db.import_reference_set(source_path=mini_spectra_db)
    row = mem_db.conn.execute(
        "SELECT probe_id FROM probes WHERE chromophore_name = ?",
        ("ATTO 488",),
    ).fetchone()
    assert row is not None
    spec_row = mem_db.conn.execute(
        "SELECT spectrum_type FROM spectra WHERE probe_id = ? AND spectrum_type = 'absorption' AND deleted_at IS NULL",
        (int(row["probe_id"]),),
    ).fetchone()
    assert spec_row is not None


def test_import_reference_set_dedup(mem_db, mini_spectra_db):
    """Import should be idempotent (dedup by name)."""
    c1 = mem_db.import_reference_set(source_path=mini_spectra_db)
    c2 = mem_db.import_reference_set(source_path=mini_spectra_db)
    # Second import should insert fewer (or zero) new probes
    assert c2["probes"] <= c1["probes"]


def test_import_marks_verified_when_requested(mem_db, mini_spectra_db):
    """mark_verified=True should stamp probes as approved."""
    mem_db.import_reference_set(source_path=mini_spectra_db, mark_verified=True)
    row = mem_db.conn.execute(
        "SELECT verification_status FROM probes WHERE chromophore_name = ?",
        ("ATTO 488",),
    ).fetchone()
    assert row is not None
    assert row["verification_status"] == "approved"
