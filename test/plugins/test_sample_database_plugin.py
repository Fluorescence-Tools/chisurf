"""Tests for the sample database plugin package."""

from __future__ import annotations

import tempfile
from pathlib import Path

from chisurf.core.fio.mmcif.db import FluorophoreDatabase
from chisurf.core.plugin.manifest import load_manifest
from chisurf.plugins.sample_database.backend import measurement_services
from chisurf.plugins.sample_database.backend.services import register_services
from chisurf.plugins.sample_database.gui.client import SampleDatabaseClient
from chisurf.server.dispatcher import ServiceDispatcher
from chisurf.server.session import SessionState


def test_sample_database_manifest_is_valid():
    manifest_path = (
        Path(__file__).resolve().parents[2]
        / "chisurf"
        / "plugins"
        / "sample_database"
        / "manifest.json"
    )
    manifest = load_manifest(manifest_path)
    assert manifest is not None
    assert manifest.id == "sample_database"
    assert "sample_database.samples.list" in [method.name for method in manifest.rpc_methods]
    assert "sample_database.export_table" in [method.name for method in manifest.rpc_methods]
    assert "raw_data.register" in [method.name for method in manifest.rpc_methods]
    assert "archive.burst_processing_manifest.export" in [
        method.name for method in manifest.rpc_methods
    ]


def test_sample_database_services_register():
    dispatcher = ServiceDispatcher(SessionState())
    register_services(dispatcher)
    methods = set(dispatcher.list_methods())
    assert "sample_database.samples.list" in methods
    assert "sample_database.users.list" in methods
    assert "sample_database.devices.list" in methods
    assert "sample_database.export_table" in methods
    assert "raw_data.register" in methods
    assert "processing.burst_selection.record" in methods
    assert "archive.burst_processing_manifest.export" in methods


def test_measurement_provenance_services_roundtrip(tmp_path, monkeypatch):
    """Exercise the fdb RPC handlers through the dispatcher."""
    db_path = tmp_path / "service.db"
    raw_path = tmp_path / "input.spc"
    raw_path.write_bytes(b"fake photons")
    bur_path = tmp_path / "output.bur"
    bur_path.write_text("n_ph\tduration\n10\t0.1\n", encoding="utf-8")

    with FluorophoreDatabase(db_path) as db:
        db.add_sample("sample_1")
        db.add_experiment("exp_1", sample_id="sample_1")

    monkeypatch.setattr(measurement_services, "resolve_database_path", lambda: db_path)
    dispatcher = ServiceDispatcher(SessionState())
    register_services(dispatcher)

    raw_result = dispatcher.dispatch(
        "raw_data.register",
        {
            "experiment_id": "exp_1",
            "data_type": "SPC",
            "storage_mode": "local_file",
            "file_path": str(raw_path),
            "validation_status": "valid",
        },
    )
    assert raw_result["ok"] is True
    raw_id = raw_result["raw_data"]["raw_data_id"]
    assert raw_result["raw_data"]["checksum"]

    record_result = dispatcher.dispatch(
        "processing.burst_selection.record",
        {
            "experiment_id": "exp_1",
            "raw_data_ids": [raw_id],
            "settings": {"burst_detection": {"min_photons": 10}},
            "output_paths": {"bur": str(bur_path)},
            "result_metadata": {"n_files": 1, "n_photons": 100, "n_selected": 50, "n_bursts": 1},
        },
    )
    assert record_result["ok"] is True
    processing_id = record_result["processing_run"]["processing_id"]
    product_id = record_result["processing_run"]["processed_data"][0]["processed_data_id"]

    trace_result = dispatcher.dispatch(
        "provenance.trace_processed_data",
        {"processed_data_id": product_id},
    )
    assert trace_result["ok"] is True
    assert trace_result["trace"]["processing_run"]["processing_id"] == processing_id

    manifest_result = dispatcher.dispatch(
        "archive.burst_processing_manifest.export",
        {"processing_id": processing_id},
    )
    assert manifest_result["ok"] is True
    assert manifest_result["manifest"]["raw_data"][0]["raw_data_id"] == raw_id
    assert manifest_result["processed_data_id"]


def test_sample_database_client_status():
    client = SampleDatabaseClient()
    status = client.status()
    assert "user_database" in status
    assert "schema_version" in status
    assert isinstance(client.list_users(), list)
    assert isinstance(client.list_devices(), list)


def test_sample_database_client_export_table():
    client = SampleDatabaseClient()
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "samples.csv"
        result = client.export_table(str(output))
        assert Path(result["output_path"]).exists()
        assert "sample_id" in output.read_text()
