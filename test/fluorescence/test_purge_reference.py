"""Replace mode for the reference set (``purge_reference_probes`` / ``replace``).

When the MFDB's optical-component set is messy, it is rebuilt cleanly: purge the
existing probes (+ spectra / optical properties / images) and re-import from a
freshly scraped staging DB. The purge refuses to run if any sample / FRET /
reagent row still references a probe, so user data is never orphaned.
"""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pytest

from chisurf.core.mfdb.repository import MFDatabase


@pytest.fixture
def mfdb():
    db = MFDatabase(":memory:")
    yield db
    db.close()


def _seed_probe(db, name="oldjunk", category="other"):
    pid = db.conn.execute(
        "INSERT INTO probes (chromophore_name, category, verification_status, created_at, updated_at) "
        "VALUES (?, ?, 'unverified', '', '')",
        (name, category),
    ).lastrowid
    db.conn.execute(
        "INSERT INTO spectra (probe_id, spectrum_type, wavelengths, intensity_values, created_at, updated_at) "
        "VALUES (?, 'absorption', ?, ?, '', '')",
        (pid, np.zeros(3).tobytes(), np.zeros(3).tobytes()),
    )
    db.conn.execute(
        "INSERT INTO optical_properties (probe_id, property_name, property_value, created_at, updated_at) "
        "VALUES (?, 'abs_max', '500', '', '')",
        (pid,),
    )
    db.conn.commit()
    return pid


def _mini_staging():
    fd, path = tempfile.mkstemp(suffix=".db")
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE probe_types (type_id INTEGER PRIMARY KEY, type_name TEXT UNIQUE,
            display_name TEXT, created_at TEXT, updated_at TEXT, deleted_at TEXT);
        CREATE TABLE probes (probe_id INTEGER PRIMARY KEY, chromophore_name TEXT,
            type_id INTEGER, description TEXT, category TEXT, source TEXT, source_ref TEXT,
            is_curated INTEGER DEFAULT 0, quality_flag INTEGER DEFAULT 1,
            created_at TEXT, updated_at TEXT, deleted_at TEXT);
        CREATE TABLE optical_properties (id INTEGER PRIMARY KEY, probe_id INTEGER,
            property_name TEXT, property_value TEXT, unit TEXT, details TEXT,
            created_at TEXT, updated_at TEXT, deleted_at TEXT);
        CREATE TABLE spectra (id INTEGER PRIMARY KEY, probe_id INTEGER, spectrum_type TEXT,
            wavelengths BLOB, intensity_values BLOB, wavelength_unit TEXT, intensity_unit TEXT,
            details TEXT, created_at TEXT, updated_at TEXT, deleted_at TEXT);
        INSERT INTO probe_types (type_id, type_name, display_name) VALUES (1,'atto_organic_dye','Atto Organic Dye');
        INSERT INTO probes (probe_id, chromophore_name, type_id, category, source, source_ref)
            VALUES (1,'ATTO 488',1,'organic_dye','atto','atto488');
        INSERT INTO optical_properties (probe_id, property_name, property_value)
            VALUES (1,'component_kind','organic_dye'),(1,'em_max','523');
        """
    )
    wl = np.linspace(400, 700, 8).astype(np.float64)
    conn.execute(
        "INSERT INTO spectra (probe_id, spectrum_type, wavelengths, intensity_values) VALUES (1,'emission',?,?)",
        (wl.tobytes(), np.ones(8).tobytes()),
    )
    conn.commit()
    conn.close()
    return path


def test_replace_purges_then_imports(mfdb):
    _seed_probe(mfdb, "oldjunk")
    assert mfdb.conn.execute("SELECT COUNT(*) FROM probes").fetchone()[0] == 1

    staging = _mini_staging()
    counts = mfdb.import_reference_set(source_path=staging, replace=True)

    assert counts["purged"]["probes"] == 1
    assert counts["purged"]["spectra"] == 1
    rows = mfdb.conn.execute(
        "SELECT chromophore_name, category, source FROM probes WHERE deleted_at IS NULL"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["chromophore_name"] == "ATTO 488"
    assert rows[0]["category"] == "organic_dye"
    assert rows[0]["source"] == "atto"
    Path(staging).unlink(missing_ok=True)


def test_purge_refuses_when_probe_is_referenced(mfdb):
    pid = _seed_probe(mfdb, "used")
    # a sample references the probe → purge must refuse rather than orphan it.
    # FK enforcement is disabled only to seed the reference without a full sample.
    mfdb.conn.execute("PRAGMA foreign_keys=OFF")
    mfdb.conn.execute(
        "INSERT INTO flr_sample_probe (sample_id, probe_id) VALUES ('S1', ?)", (pid,)
    )
    mfdb.conn.commit()
    mfdb.conn.execute("PRAGMA foreign_keys=ON")
    with pytest.raises(RuntimeError, match="reference probes"):
        mfdb.purge_reference_probes()
    # nothing deleted
    assert mfdb.conn.execute("SELECT COUNT(*) FROM probes").fetchone()[0] == 1
