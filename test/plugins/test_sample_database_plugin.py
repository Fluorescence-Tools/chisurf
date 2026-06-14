"""Tests for the MFDB admin plugin."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import chisurf.gui as chisurf_gui
from chisurf.core.fio.mmcif.db import FluorophoreDatabase
from chisurf.core.plugin.manifest import load_manifest
from chisurf.plugins.core.mfdb_admin.backend import measurement_services, services
from chisurf.plugins.core.mfdb_admin.backend.services import register_services
from chisurf.plugins.core.mfdb_admin.gui.client import MFDBClient
from chisurf.plugins.core.mfdb_admin.gui.tool import (
    MFDBWidget,
    PasswordChangeDialog,
    _experiment_id_for_processing_id,
    _experiment_processing_ids,
    _processed_row_count,
    _qurl_for_location,
    _scope_processed_products_by_experiment,
    _unwrap_provenance_graph_response,
)
from chisurf.plugins.sample_database.gui.client import SampleDatabaseClient
from chisurf.server.dispatcher import ServiceDispatcher
from chisurf.server.session import SessionState


def test_mfdb_admin_manifest_is_valid():
    manifest_path = (
        Path(__file__).resolve().parents[2]
        / "chisurf"
        / "plugins"
        / "core"
        / "mfdb_admin"
        / "manifest.json"
    )
    manifest = load_manifest(manifest_path)
    assert manifest is not None
    assert manifest.id == "mfdb_admin"
    assert manifest.statefulness.enabled is True
    assert manifest.statefulness.window.settings_key is None
    assert "mfdb.samples.list" in [method.name for method in manifest.rpc_methods]
    assert "mfdb.export_table" in [method.name for method in manifest.rpc_methods]
    assert "mfdb.samples.search" in [method.name for method in manifest.rpc_methods]
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
    assert "sample_database.setups.list" in methods
    assert "mfdb.samples.list" in methods
    assert "mfdb.samples.search" in methods
    assert "mfdb.users.list" in methods
    assert "mfdb.auth.login" in methods
    assert "mfdb.auth.change_password" in methods
    assert "mfdb.devices.list" in methods
    assert "mfdb.export_table" in methods
    assert "raw_data.register" in methods
    assert "processing.burst_selection.record" in methods
    assert "archive.burst_processing_manifest.export" in methods


def test_mfdb_client_default_uses_zmq_transport(monkeypatch):
    backend_modules = [
        name
        for name in sys.modules
        if name.startswith("chisurf.plugins.core.mfdb_admin.backend")
    ]
    for name in backend_modules:
        monkeypatch.delitem(sys.modules, name, raising=False)

    def make_zmq_client(self, **kwargs):
        return kwargs

    monkeypatch.setattr(MFDBClient, "_make_zmq_client", make_zmq_client)

    client = MFDBClient(host="example.test", cmd_port=1234, pub_port=5678, timeout_ms=99)

    assert client._client == {
        "host": "example.test",
        "cmd_port": 1234,
        "pub_port": 5678,
        "timeout_ms": 99,
    }
    assert not any(
        name.startswith("chisurf.plugins.core.mfdb_admin.backend")
        for name in sys.modules
    )


def test_mfdb_client_change_password_uses_rpc_method():
    """Password changes are sent through the auth RPC method."""
    calls = []

    class Client:
        def call(self, method, params):
            calls.append((method, params))
            return {"ok": True}

    client = MFDBClient(client=Client())

    assert client.change_password("user_1", "StrongPassword!123", requester_id="admin") == {"ok": True}
    assert calls == [
        (
            "mfdb.auth.change_password",
            {
                "password": "StrongPassword!123",
            },
        )
    ]


def test_password_change_dialog_scores_strength():
    """Password dialog uses red/yellow/green strength buckets."""
    weak_score, _ = PasswordChangeDialog._score_password("abc")
    medium_score, _ = PasswordChangeDialog._score_password("Password123")
    strong_score, _ = PasswordChangeDialog._score_password("StrongPassword!123")

    assert weak_score <= 2
    assert 3 <= medium_score <= 4
    assert strong_score == 5


def test_gui_starts_embedded_mfdb_rpc_when_unavailable(monkeypatch):
    """GUI startup creates the embedded RPC server before login."""
    calls = {"available": 0, "started": False}

    class Server:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def serve_forever(self):
            calls["started"] = True

        def stop(self):
            pass

    class Thread:
        def __init__(self, target, daemon=False, name=None):
            self.target = target
            self.daemon = daemon
            self.name = name

        def start(self):
            self.target()

    def available(timeout_ms=500):
        calls["available"] += 1
        return calls["available"] > 1

    monkeypatch.delattr(chisurf_gui.chisurf, "__mfdb_rpc_server__", raising=False)
    monkeypatch.setattr(chisurf_gui, "_mfdb_rpc_is_available", available)
    monkeypatch.setattr(chisurf_gui.threading, "Thread", Thread)
    monkeypatch.setattr("chisurf.server.app.ChiSurfServer", Server)

    chisurf_gui._ensure_mfdb_rpc_server()

    assert calls["started"] is True
    assert isinstance(chisurf_gui.chisurf.__mfdb_rpc_server__, Server)


def test_mfdb_client_search_samples():
    client = MFDBClient(inprocess=True)
    assert client.login("guest").get("ok") is True
    results = client.search_samples("")
    assert isinstance(results, list)


def test_legacy_compat_imports():
    from chisurf.plugins.core.mfdb_admin import MFDBWidget
    from chisurf.plugins.core.mfdb_admin.gui.client import MFDBClient
    from chisurf.plugins.sample_database import SampleDatabaseWidget
    from chisurf.plugins.sample_database.gui.client import SampleDatabaseClient as SDC

    assert SampleDatabaseWidget is MFDBWidget
    assert issubclass(SDC, MFDBClient)



def test_mfdb_widget_helpers_handle_url_locations():
    """MFDB location helpers preserve URL schemes."""
    assert _qurl_for_location("http://example.test/data.bur").toString() == "http://example.test/data.bur"
    assert _qurl_for_location("/tmp/data.bur").isLocalFile()


def test_mfdb_widget_helpers_scope_processed_data_by_processing_runs():
    """Processed products are scoped through processing runs for an experiment."""
    class Client:
        def list_processing_runs(self, experiment_id=None):
            if experiment_id == "exp_empty":
                return []
            return [{"processing_id": "proc_1"}]

        def get_processing_run(self, processing_id):
            return {"experiment_id": "exp_1"}

    client = Client()
    products = [
        {"processed_data_id": "prod_1", "processing_id": "proc_1", "row_count": 10},
        {"processed_data_id": "prod_2", "processing_id": "proc_2", "row_count": 20},
    ]

    assert _experiment_processing_ids(client, "exp_1") == {"proc_1"}
    assert _processed_row_count(products[0]) == 10
    assert _scope_processed_products_by_experiment(products, client, "exp_1") == [products[0]]
    assert _scope_processed_products_by_experiment(products, client, "exp_empty") == []
    assert _experiment_id_for_processing_id(client, "proc_1") == "exp_1"


def test_mfdb_widget_helper_unwraps_export_response():
    """Full provenance export responses expose their graph payload."""
    graph = {"nodes": [], "edges": []}
    assert _unwrap_provenance_graph_response({"ok": True, "graph": graph}) == graph
    assert _unwrap_provenance_graph_response(graph) == graph


def test_mfdb_sample_condition_and_probe_services(tmp_path, monkeypatch):
    """Condition and probe services are available through the MFDB client boundary."""
    db_path = tmp_path / "services.db"
    with FluorophoreDatabase(db_path) as db:
        db.add_probe(7, name="ATTO 488", category="standard")
        db.add_optical_property(7, "abs_max", 495.0, unit="nm")
        db.add_optical_property(7, "em_max", 519.0, unit="nm")

    monkeypatch.setattr(services, "resolve_database_path", lambda: db_path)
    dispatcher = ServiceDispatcher(SessionState())
    register_services(dispatcher)

    condition_result = dispatcher.dispatch(
        "mfdb.sample_conditions.save",
        {
            "condition": {
                "condition_id": "pbs",
                "ph": 7.4,
                "temperature": 25.0,
                "ionic_strength": 0.15,
                "buffer_composition": "PBS",
                "details": "phosphate buffered saline",
            }
        },
    )
    assert condition_result["condition"]["condition_id"] == "pbs"

    condition = dispatcher.dispatch(
        "mfdb.sample_conditions.get",
        {"condition_id": "pbs"},
    )["condition"]
    assert condition["buffer_composition"] == "PBS"

    probes = dispatcher.dispatch("mfdb.probes.list", {})["probes"]
    assert probes[0]["probe_id"] == 7
    assert {prop["property_name"] for prop in probes[0]["optical_properties"]} == {"abs_max", "em_max"}



def test_mfdb_widget_refresh_handles_transport_errors(qapp):
    """MFDB widget construction remains usable when ZMQ transport is unavailable."""
    del qapp

    class FailingClient:
        def status(self):
            raise RuntimeError("receive timeout")

    widget = MFDBWidget(client=FailingClient())

    assert widget.transport_connected is False
    assert "MFDB transport unavailable" in widget.status_label.text()
    assert widget.sample_table.rowCount() == 0
    assert widget.refresh_action.isEnabled()


def test_mfdb_client_list_raw_data_uses_raw_data_key():
    """MFDB raw-data client helper consumes the raw_data RPC key."""
    client = MFDBClient()
    client._client = type("Client", (), {"call": lambda _self, _method, _params: {"raw_data": [{"raw_data_id": "raw_1"}]}})()

    assert client.list_raw_data() == [{"raw_data_id": "raw_1"}]


def test_mfdb_client_list_processed_data_uses_processed_data_key():
    """MFDB processed-data client helper consumes the processed_data RPC key."""
    client = MFDBClient()
    client._client = type("Client", (), {"call": lambda _self, _method, _params: {"processed_data": [{"processed_data_id": "prod_1"}]}})()

    assert client.list_processed_data() == [{"processed_data_id": "prod_1"}]


def test_mfdb_client_dependencies_send_node_type_and_id():
    """Provenance dependency helpers send node_type and node_id parameters."""
    client = MFDBClient()
    calls = []

    def call(_self, method, params):
        calls.append((method, params))
        return {"ok": True, "result": {"nodes": [], "edges": []}}

    client._client = type("Client", (), {"call": call})()

    assert client.dependencies_upstream("raw_data", "raw_1")["nodes"] == []
    assert client.dependencies_downstream("processed_data", "prod_1")["nodes"] == []
    assert calls == [
        ("provenance.dependencies.upstream", {"node_type": "raw_data", "node_id": "raw_1"}),
        ("provenance.dependencies.downstream", {"node_type": "processed_data", "node_id": "prod_1"}),
    ]


def test_mfdb_client_list_provenance_edges_uses_provenance_edges_key():
    """Provenance edge client helper consumes the provenance_edges RPC key."""
    client = MFDBClient()
    client._client = type(
        "Client",
        (),
        {"call": lambda _self, _method, _params: {"provenance_edges": [{"edge_id": "edge_1"}]}}
    )()

    assert client.list_provenance_edges() == [{"edge_id": "edge_1"}]



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
