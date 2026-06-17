"""Tests for curated sample database seed data."""

from __future__ import annotations

import tempfile
from pathlib import Path

from chisurf.core.mfdb.repository import MFDatabase
from chisurf.core.mfdb.seed_data import seed_curated_database


def test_seed_curated_database_contains_samples():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "sample_management.db"
        seed_curated_database(db_path)
        with MFDatabase(db_path) as db:
            assert db._get_schema_version() >= 11
            assert len(db.list_samples()) >= 3
            assert len(db.get_probes()) >= 7
            assert len(db.get_sample_probe_mappings()) >= 3
            assert len(db.get_users()) >= 2
            assert len(db.get_devices()) >= 2
            assert len(db.get_experiment_types()) >= 7
            assert len(db.get_experiments()) >= 4
            assert len(db.get_experiment_data("1RTD_spectra_001")) == 1
            assert db.get_experiment_data("1RTD_spectra_001")[0]["reading_options_json"]
            assert len(db.get_sample_key_values("148L_65_141_A488_A594")) >= 2
