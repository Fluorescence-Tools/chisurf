from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def seeded_db(tmp_path, monkeypatch):
    from mfdb.repository import MFDatabase
    from chisurf.plugins.core.database_connector import services

    db_path = tmp_path / "mfdb.db"
    source_path = tmp_path / "source.db"
    object_root = tmp_path / "objects"
    with MFDatabase(db_path) as db:
        db.add_user("user_default", "Default User")
        db.add_user("user_a", "User A")
        db.add_device("dev_1", "Sample Detector")
        type_id = db.add_experiment_type(
            "TCSPC",
            category="fluorescence",
            description="Time-correlated single photon counting",
        )
        db.add_sample(
            "sample_1",
            description="Sample data",
            measured_by_user_id="user_a",
            measured_by_device_id="dev_1",
        )
        db.add_experiment(
            "exp_1",
            type_id=type_id,
            sample_id="sample_1",
            measured_by_user_id="user_a",
            measured_by_device_id="dev_1",
            status="complete",
        )
    source_path.write_bytes(db_path.read_bytes())

    monkeypatch.setattr(services, "resolve_database_path", lambda: db_path)
    monkeypatch.setattr(services, "source_database_path", lambda: source_path)
    monkeypatch.setattr(services, "user_database_path", lambda: db_path)
    monkeypatch.setattr(
        "mfdb.database_resolver.resolve_database_path",
        lambda: db_path,
    )
    monkeypatch.setattr(
        "mfdb.database_resolver.object_store_root",
        lambda: object_root,
    )
    services.close_handler()
    yield db_path
    services.close_handler()


def test_repository_reports_seeded_mfdb_counts(seeded_db):
    from chisurf.plugins.core.database_connector.services import repository_handler

    result = repository_handler()

    assert Path(result["database_path"]) == seeded_db
    assert result["sample_count"] == 1
    assert result["user_count"] >= 2
    assert result["device_count"] == 1
    assert result["experiment_type_count"] == 1
    assert result["experiment_count"] == 1


def test_open_status_uses_explicit_database_path(seeded_db):
    from chisurf.plugins.core.database_connector.services import close_handler, open_handler

    opened = open_handler(database_path=str(seeded_db))

    assert opened["active_database"] == str(seeded_db)
    assert opened["sample_count"] == 1
    assert close_handler()["was_connected"] is True


def test_repository_is_available_through_inprocess_rpc(seeded_db):
    from chisurf.core.plugin.client import InProcessClient
    from chisurf.plugins.core.database_connector.services import register_services
    from chisurf.server.dispatcher import ServiceDispatcher
    from chisurf.server.session import SessionState

    dispatcher = ServiceDispatcher(SessionState())
    register_services(dispatcher)
    client = InProcessClient(dispatcher)

    result = client.call("database_connector.repository")

    assert result["database_path"] == str(seeded_db)
    assert result["sample_count"] == 1
    assert "database_connector.repository" in dispatcher.list_methods()


def test_backup_import_export_and_reset_use_temporary_mfdb(seeded_db, tmp_path):
    from mfdb.repository import MFDatabase
    from chisurf.plugins.core.database_connector.services import (
        backup_handler,
        export_sample_handler,
        import_file_handler,
        repository_handler,
        reset_from_source_handler,
    )

    backup = backup_handler()
    backup_path = Path(backup["backup_path"])
    assert backup_path.exists()
    assert backup_path.parent == seeded_db.parent / "backups"

    imported_cif = tmp_path / "connector_import_sample.cif"
    imported_cif.write_text("data_connector_import_sample\n#\n", encoding="utf-8")
    imported = import_file_handler(str(imported_cif))["summary"]
    assert imported["samples"] == ["connector_import_sample"]
    with MFDatabase(seeded_db) as db:
        assert db.get_sample("connector_import_sample") is not None
    assert repository_handler()["sample_count"] == 2

    export_text = export_sample_handler("sample_1")["text"]
    assert "data_chisurf_flr_export" in export_text

    output_path = tmp_path / "sample_1.cif"
    exported = export_sample_handler("sample_1", output_path=str(output_path))
    assert exported["output_path"] == str(output_path)
    assert output_path.exists()
    assert "data_chisurf_flr_export" in output_path.read_text(encoding="utf-8")

    reset = reset_from_source_handler()
    assert reset["ok"] is True
    assert reset["backup_path"]
    assert Path(reset["backup_path"]).exists()
    with MFDatabase(seeded_db) as db:
        assert db.get_sample("sample_1") is not None
        assert db.get_sample("connector_import_sample") is None
    assert repository_handler()["sample_count"] == 1
