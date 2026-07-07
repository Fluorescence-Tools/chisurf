"""Tests for fdb Phase 3 setup definitions and experiment linkage."""

from __future__ import annotations

import pathlib
from unittest.mock import patch

from mfdb.repository import MFDatabase
from mfdb.admin.backend.services import (
    delete_setup_handler,
    get_setup_handler,
    list_setups_handler,
    save_setup_handler,
)


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


def test_setup_service_handlers(tmp_path: pathlib.Path) -> None:
    """Verify canonical MFDB Admin setup handlers handle CRUD requests."""
    db_path = tmp_path / "test_setup_services.db"

    with (
        patch("mfdb.admin.backend.services.resolve_database_path", return_value=db_path),
        patch("mfdb.admin.backend.services._require_auth", return_value=None),
    ):
        setup_payload = {
            "setup_id": "setup_1",
            "name": "Validation Setup",
            "laser_wavelengths": "[488]",
        }
        res = save_setup_handler(setup=setup_payload)
        saved = res["setup"]
        assert saved["setup_id"] == "setup_1"
        assert saved["name"] == "Validation Setup"
        assert saved["laser_wavelengths"] == [488]

        res = get_setup_handler(setup_id="setup_1")
        assert res["setup"]["name"] == "Validation Setup"

        res = list_setups_handler()
        assert len(res["setups"]) == 1
        assert res["setups"][0]["setup_id"] == "setup_1"

        res = delete_setup_handler(setup_id="setup_1")
        assert res["ok"] is True
        assert res["setup_id"] == "setup_1"

        res = list_setups_handler()
        assert len(res["setups"]) == 0
