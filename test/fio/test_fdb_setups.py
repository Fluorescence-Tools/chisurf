"""Tests for fdb Phase 3 setup definitions and experiment linkage."""

from __future__ import annotations

import pathlib
import sqlite3
from unittest.mock import patch

from chisurf.core.mfdb import schema
from chisurf.core.mfdb.repository import MFDatabase
from chisurf.plugins.sample_database.backend.setup_services import (
    delete_setup_handler,
    get_setup_handler,
    list_setups_handler,
    save_setup_handler,
    validate_setup_config,
    validate_setup_handler,
)


def test_v13_database_migrates_to_v14(tmp_path: pathlib.Path) -> None:
    """Verify legacy database upgrades to v14 and adds setup definition tables/columns."""
    db_path = tmp_path / "legacy_v13.db"
    conn = sqlite3.connect(db_path)
    try:
        # Create all tables except setups-related ones by filtering schema SQLs
        for sql in schema.CREATE_TABLES_SQL:
            if "fdb_setup_definition" not in sql:
                conn.execute(sql)
        conn.execute("DELETE FROM _schema_version")
        conn.execute("INSERT INTO _schema_version (version) VALUES (13)")
        conn.commit()
    finally:
        conn.close()

    with MFDatabase(db_path) as db:
        assert db._get_schema_version() == schema.SCHEMA_VERSION

        
        # Verify setups table exists
        tables = {
            row["name"]
            for row in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "fdb_setup_definition" in tables
        
        # Verify setup_definition_id column was added to flr_experiment
        cols = {
            r[1] for r in db.conn.execute("PRAGMA table_info(flr_experiment)").fetchall()
        }
        assert "setup_definition_id" in cols


def test_setup_definition_repository_crud_and_linkage(tmp_path: pathlib.Path) -> None:
    """Verify setup CRUD operations and linkage to experiments in MFDatabase."""
    db_path = tmp_path / "test_setups.db"
    
    with MFDatabase(db_path) as db:
        # 1. Test Add and Get
        db.add_setup_definition(
            setup_id="setup_mfd_1",
            name="MFD Setup 1",
            version=1,
            description="Confocal MFD setup with green/red channels",
            configuration={"laser_wavelengths": [485, 640]},
            detectors={"green": [0, 1], "red": [2, 3]},
        )
        
        row = db.get_setup_definition("setup_mfd_1")
        assert row is not None
        assert row["name"] == "MFD Setup 1"
        assert row["setup_id"] == "setup_mfd_1"
        
        decoded = db._decode_setup_definition_row(row)
        assert decoded["configuration"] == {"laser_wavelengths": [485, 640]}
        assert decoded["detectors"] == {"green": [0, 1], "red": [2, 3]}

        # 2. Test List
        setups = db.list_setup_definitions()
        assert len(setups) == 1
        assert setups[0]["setup_id"] == "setup_mfd_1"

        # 3. Test Linkage to Experiment
        db.add_sample("sample_1")
        db.add_experiment(
            experiment_id="exp_mfd",
            sample_id="sample_1",
            setup_definition_id="setup_mfd_1",
            status="complete",
        )
        
        exp_row = db.get_experiment("exp_mfd")
        assert exp_row is not None
        assert exp_row["setup_definition_id"] == "setup_mfd_1"
        assert exp_row["setup_name"] == "MFD Setup 1"

        # 4. Test Delete
        db.delete_setup_definition("setup_mfd_1")
        setup = db.get_setup_definition("setup_mfd_1")
        assert setup is not None
        assert setup["deleted_at"] is not None
        
        # Link remains intact because soft-delete keeps the setup record in the DB
        exp_row = db.get_experiment("exp_mfd")
        assert exp_row is not None
        assert exp_row["setup_definition_id"] == "setup_mfd_1"
        assert exp_row["setup_name"] == "MFD Setup 1"


def test_setup_configuration_validation() -> None:
    """Verify validation helper detects valid/invalid configurations."""
    # Test valid configuration
    valid_config = {
        "laser_wavelengths": [485.5, 640],
        "detector_channels": {"green": [0, 1], "red": [2, 3]},
        "pie_enabled": True,
        "pie_window": [0, 100],
    }
    res = validate_setup_config(valid_config)
    assert res["valid"] is True
    assert len(res["errors"]) == 0

    # Test invalid configurations
    invalid_lasers = {"laser_wavelengths": "not-a-list"}
    res = validate_setup_config(invalid_lasers)
    assert res["valid"] is False
    assert "laser_wavelengths must be a list of numbers" in res["errors"]

    empty_lasers = {"laser_wavelengths": []}
    res = validate_setup_config(empty_lasers)
    assert res["valid"] is False
    assert "laser_wavelengths list cannot be empty" in res["errors"]

    negative_lasers = {"laser_wavelengths": [485, -10]}
    res = validate_setup_config(negative_lasers)
    assert res["valid"] is False
    assert "laser_wavelengths[1] must be a positive number: -10" in res["errors"]

    invalid_detectors = {"detector_channels": "not-a-dict-or-list"}
    res = validate_setup_config(invalid_detectors)
    assert res["valid"] is False
    assert "detector_channels must be a dictionary or list representing detector mappings" in res["errors"]

    missing_pie_window = {"pie_enabled": True}
    res = validate_setup_config(missing_pie_window)
    assert res["valid"] is False
    assert "pie_window must be defined when pie_enabled is True" in res["errors"]


def test_setup_service_handlers(tmp_path: pathlib.Path) -> None:
    """Verify setup service endpoints handle CRUD and validation requests."""
    db_path = tmp_path / "test_setup_services.db"

    patcher = patch(
        "chisurf.plugins.sample_database.backend.setup_services.resolve_database_path",
        return_value=db_path,
    )
    patcher.start()

    try:
        # 1. Test save_setup_handler
        setup_payload = {
            "setup_id": "setup_1",
            "name": "Validation Setup",
            "configuration": {"laser_wavelengths": [488]},
        }
        res = save_setup_handler(setup=setup_payload)
        assert res["ok"] is True
        saved = res["setup"]
        assert saved["setup_id"] == "setup_1"
        assert saved["name"] == "Validation Setup"
        assert saved["validation"]["valid"] is True

        # 2. Test get_setup_handler
        res = get_setup_handler(setup_id="setup_1")
        assert res["ok"] is True
        assert res["setup"]["name"] == "Validation Setup"

        # 3. Test list_setups_handler
        res = list_setups_handler()
        assert res["ok"] is True
        assert len(res["setups"]) == 1
        assert res["setups"][0]["setup_id"] == "setup_1"

        # 4. Test validate_setup_handler
        res = validate_setup_handler(configuration={"laser_wavelengths": [488]})
        assert res["valid"] is True

        # 5. Test delete_setup_handler
        res = delete_setup_handler(setup_id="setup_1")
        assert res["ok"] is True
        assert res["setup_id"] == "setup_1"

        res = list_setups_handler()
        assert len(res["setups"]) == 0

    finally:
        patcher.stop()
