from __future__ import annotations

import threading
import time

import pytest

from chisurf.core.api._client import ChisurfClient, RemoteError
from chisurf.server.dispatcher import ServiceDispatcher
from chisurf.server.session import SessionState
from chisurf.server.transport.zmq import ZmqServer
from test.server.helpers import find_free_port


@pytest.fixture
def zmq_server():
    cmd_port = find_free_port()
    pub_port = find_free_port()
    dispatcher = ServiceDispatcher(SessionState())
    dispatcher._build_default_registry()
    server = ZmqServer(
        handler=dispatcher.dispatch,
        cmd_port=cmd_port,
        pub_port=pub_port,
    )
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.5)
    yield cmd_port, pub_port, dispatcher, server
    server.stop()


# ── Legacy methods ────────────────────────────────────────────────

def test_client_list_methods(zmq_server):
    cmd_port, pub_port, dispatcher, server = zmq_server
    client = ChisurfClient(cmd_port=cmd_port, pub_port=pub_port)
    methods = client.list_methods()
    assert isinstance(methods, list)
    assert "list_datasets" in methods
    assert "list_fits" in methods
    assert "list_methods" in methods
    client.close()


def test_client_list_datasets(zmq_server):
    cmd_port, pub_port, dispatcher, server = zmq_server
    client = ChisurfClient(cmd_port=cmd_port, pub_port=pub_port)
    result = client.list_datasets()
    assert isinstance(result, list)
    client.close()


def test_client_list_fits(zmq_server):
    cmd_port, pub_port, dispatcher, server = zmq_server
    client = ChisurfClient(cmd_port=cmd_port, pub_port=pub_port)
    result = client.list_fits()
    assert isinstance(result, list)
    client.close()


def test_client_unknown_method(zmq_server):
    cmd_port, pub_port, dispatcher, server = zmq_server
    client = ChisurfClient(cmd_port=cmd_port, pub_port=pub_port)
    with pytest.raises(RemoteError) as exc:
        client._call("nonexistent_method")
    assert "not found" in str(exc.value)
    assert exc.value.error_code == "METHOD_NOT_FOUND"
    assert exc.value.jsonrpc_code == -32601
    client.close()


def test_client_connect_timeout():
    port = find_free_port()
    client = ChisurfClient(cmd_port=port, timeout_ms=500)
    with pytest.raises(Exception):
        client.list_methods()
    client.close()


# ── Namespaced methods ───────────────────────────────────────────

def test_client_meta_ping(zmq_server):
    cmd_port, pub_port, _, _ = zmq_server
    client = ChisurfClient(cmd_port=cmd_port, pub_port=pub_port)
    result = client.meta__ping()
    assert result.get("ok") is True
    assert result.get("status") == "alive"
    client.close()


def test_client_meta_methods(zmq_server):
    cmd_port, pub_port, _, _ = zmq_server
    client = ChisurfClient(cmd_port=cmd_port, pub_port=pub_port)
    methods = client.meta__methods()
    assert isinstance(methods, list)
    assert "dataset.list" in methods
    assert "dataset.get" in methods
    assert "fit.list" in methods
    assert "fit.run" in methods
    assert "parameter.get" in methods
    assert "parameter.set_value" in methods
    assert "project.info" in methods
    assert "project.save" in methods
    assert "meta.ping" in methods
    assert "meta.methods" in methods
    assert "editor.document.list" in methods
    client.close()


def test_client_meta_protocol(zmq_server):
    cmd_port, pub_port, _, _ = zmq_server
    client = ChisurfClient(cmd_port=cmd_port, pub_port=pub_port)
    result = client.meta__protocol()
    assert result.get("ok") is True
    assert isinstance(result.get("protocol_version"), str)
    assert "meta" in result.get("catalogue", {})
    assert "meta.protocol" in result.get("schemas", {})
    client.close()


def test_client_dataset_list_namespaced(zmq_server):
    cmd_port, pub_port, _, _ = zmq_server
    client = ChisurfClient(cmd_port=cmd_port, pub_port=pub_port)
    result = client.dataset__list()
    assert isinstance(result, list)
    client.close()


def test_client_fit_list_namespaced(zmq_server):
    cmd_port, pub_port, _, _ = zmq_server
    client = ChisurfClient(cmd_port=cmd_port, pub_port=pub_port)
    result = client.fit__list()
    assert isinstance(result, list)
    client.close()


def test_client_dataset_clear_namespaced(zmq_server):
    cmd_port, pub_port, _, _ = zmq_server
    client = ChisurfClient(cmd_port=cmd_port, pub_port=pub_port)
    result = client.dataset__clear()
    assert result.get("ok") is True
    client.close()


def test_client_fit_clear_namespaced(zmq_server):
    cmd_port, pub_port, _, _ = zmq_server
    client = ChisurfClient(cmd_port=cmd_port, pub_port=pub_port)
    result = client.fit__clear()
    assert result.get("ok") is True
    client.close()


def test_client_project_info_namespaced(zmq_server):
    cmd_port, pub_port, _, _ = zmq_server
    client = ChisurfClient(cmd_port=cmd_port, pub_port=pub_port)
    result = client.project__info()
    assert result.get("ok") is True
    assert "fit_count" in result
    assert "dataset_count" in result
    client.close()


def test_client_call_public_method(zmq_server):
    cmd_port, pub_port, _, _ = zmq_server
    client = ChisurfClient(cmd_port=cmd_port, pub_port=pub_port)
    result = client.call("meta.ping")
    assert result.get("ok") is True
    assert result.get("status") == "alive"
    client.close()


def test_client_legacy_methods_still_work(zmq_server):
    cmd_port, pub_port, _, _ = zmq_server
    client = ChisurfClient(cmd_port=cmd_port, pub_port=pub_port)
    # Legacy methods should still work
    assert client.ping().get("ok") is True
    assert isinstance(client.list_methods(), list)
    assert isinstance(client.list_datasets(), list)
    assert isinstance(client.list_fits(), list)
    assert client.get_project_info().get("ok") is True
    client.close()


# ── New namespaced mutation methods ───────────────────────────────

def test_client_dataset_rename_no_dataset(zmq_server):
    """dataset__rename raises RemoteError when dataset does not exist."""
    cmd_port, pub_port, _, _ = zmq_server
    client = ChisurfClient(cmd_port=cmd_port, pub_port=pub_port)
    with pytest.raises(RemoteError):
        client.dataset__rename("NewName", dataset_index=0)
    client.close()


def test_client_dataset_rename_with_dataset(zmq_server):
    """dataset__rename succeeds on a real dataset."""
    cmd_port, pub_port, _, _ = zmq_server
    client = ChisurfClient(cmd_port=cmd_port, pub_port=pub_port)
    ds = client.call("add_dataset", {
        "reader_name": "RenameReader",
        "filename": "/tmp/rename_test.dat",
        "name": "OriginalName",
        "curve_data": {"x": [0.0, 1.0], "y": [2.0, 3.0]},
    })
    assert ds.get("ok") is True
    result = client.dataset__rename("Renamed", dataset_index=ds["dataset_index"])
    assert result.get("ok") is True
    info = client.dataset__get(dataset_index=ds["dataset_index"])
    assert info.get("name") == "Renamed"
    client.close()


def test_client_dataset_group_empty(zmq_server):
    """dataset__group raises RemoteError for non-existent indices."""
    cmd_port, pub_port, _, _ = zmq_server
    client = ChisurfClient(cmd_port=cmd_port, pub_port=pub_port)
    with pytest.raises(RemoteError):
        client.dataset__group([0, 1])
    client.close()


def test_client_dataset_ungroup_empty(zmq_server):
    """dataset__ungroup raises RemoteError for non-existent indices."""
    cmd_port, pub_port, _, _ = zmq_server
    client = ChisurfClient(cmd_port=cmd_port, pub_port=pub_port)
    with pytest.raises(RemoteError):
        client.dataset__ungroup([0])
    client.close()


def test_client_dataset_group_returns_error_on_nonexistent(zmq_server):
    """dataset__group with non-existent indices raises RemoteError."""
    cmd_port, pub_port, _, _ = zmq_server
    client = ChisurfClient(cmd_port=cmd_port, pub_port=pub_port)
    with pytest.raises(RemoteError):
        client.dataset__group([0, 1], group_name="TestGroup")
    client.close()


def test_client_dataset_ungroup_returns_error_on_nonexistent(zmq_server):
    """dataset__ungroup with non-existent indices raises RemoteError."""
    cmd_port, pub_port, _, _ = zmq_server
    client = ChisurfClient(cmd_port=cmd_port, pub_port=pub_port)
    with pytest.raises(RemoteError):
        client.dataset__ungroup([0])
    client.close()


def test_client_fit_set_dataset_no_fit(zmq_server):
    """fit__set_dataset raises RemoteError when fit not found."""
    cmd_port, pub_port, _, _ = zmq_server
    client = ChisurfClient(cmd_port=cmd_port, pub_port=pub_port)
    with pytest.raises(RemoteError):
        client.fit__set_dataset(fit_index=0, dataset_index=0)
    client.close()


def test_client_fit_set_result_idx_no_fit(zmq_server):
    """fit__set_result_idx raises RemoteError when fit not found."""
    cmd_port, pub_port, _, _ = zmq_server
    client = ChisurfClient(cmd_port=cmd_port, pub_port=pub_port)
    with pytest.raises(RemoteError):
        client.fit__set_result_idx(fit_index=0, result_idx=1)
    client.close()


def test_client_parameter_link_no_fit(zmq_server):
    """parameter__link raises RemoteError when fit not found."""
    cmd_port, pub_port, _, _ = zmq_server
    client = ChisurfClient(cmd_port=cmd_port, pub_port=pub_port)
    with pytest.raises(RemoteError):
        client.parameter__link("tau1", "tau2", fit_index=0)
    client.close()


def test_client_parameter_unlink_no_fit(zmq_server):
    """parameter__unlink raises RemoteError when parameter not found."""
    cmd_port, pub_port, _, _ = zmq_server
    client = ChisurfClient(cmd_port=cmd_port, pub_port=pub_port)
    with pytest.raises(RemoteError):
        client.parameter__unlink("tau1", fit_index=0)
    client.close()


def test_client_graph_build_fits_empty(zmq_server):
    """graph__build_fits returns a graph with zero nodes when no fits exist."""
    cmd_port, pub_port, _, _ = zmq_server
    client = ChisurfClient(cmd_port=cmd_port, pub_port=pub_port)
    result = client.graph__build_fits()
    assert result.get("ok") is True
    nodes = result.get("nodes", [])
    edges = result.get("edges", [])
    assert len(nodes) == 0
    assert len(edges) == 0
    client.close()


# ── Structured error tests ─────────────────────────────────────────

def test_service_error_has_error_code(zmq_server):
    """Service-level 'fit not found' error raised as RemoteError."""
    cmd_port, pub_port, _, _ = zmq_server
    client = ChisurfClient(cmd_port=cmd_port, pub_port=pub_port)
    with pytest.raises(RemoteError) as exc:
        client.call("fit.get", {"fit_index": 0})
    assert exc.value.error_code == "NOT_FOUND"
    assert exc.value.jsonrpc_code == -32601
    client.close()


def test_service_error_has_error_code_params(zmq_server):
    """Service-level 'parameter not found' error raised as RemoteError."""
    cmd_port, pub_port, _, _ = zmq_server
    client = ChisurfClient(cmd_port=cmd_port, pub_port=pub_port)
    with pytest.raises(RemoteError) as exc:
        client.call("parameter.get", {"parameter_name": "nonexistent", "fit_index": 0})
    assert exc.value.error_code == "NOT_FOUND"
    assert exc.value.jsonrpc_code == -32601
    client.close()


def test_service_error_invalid_input_has_code(zmq_server):
    """Service-level validation errors raised as RemoteError."""
    cmd_port, pub_port, _, _ = zmq_server
    client = ChisurfClient(cmd_port=cmd_port, pub_port=pub_port)
    with pytest.raises(RemoteError) as exc:
        client.call("dataset.remove", {})
    assert exc.value.error_code == "INVALID_INPUT"
    client.close()


def test_service_error_invalid_state_has_code(zmq_server):
    """Service-level state errors raised as RemoteError."""
    cmd_port, pub_port, _, _ = zmq_server
    client = ChisurfClient(cmd_port=cmd_port, pub_port=pub_port)
    with pytest.raises(RemoteError) as exc:
        client.call("fit.create", {"dataset_index": 0})
    assert exc.value.error_code == "INVALID_STATE"
    client.close()


def test_remote_error_structured_on_transport_error():
    """RemoteError carries structured attributes on transport JSON-RPC error."""
    from chisurf.core.api._client import RemoteError
    err = RemoteError("test error", error_code="NOT_FOUND", jsonrpc_code=-32601, exception_type="ValueError")
    assert err.error_code == "NOT_FOUND"
    assert err.jsonrpc_code == -32601
    assert err.exception_type == "ValueError"
    assert str(err) == "test error"


def test_remote_error_on_timeout():
    """Transport timeout raises RemoteError (already tested, just verify type)."""
    port = find_free_port()
    client = ChisurfClient(cmd_port=port, timeout_ms=500)
    with pytest.raises(RemoteError):
        client.list_methods()
    client.close()
