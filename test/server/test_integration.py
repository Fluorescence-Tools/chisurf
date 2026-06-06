from __future__ import annotations

import threading
import time
import pytest

from chisurf.core.api._client import ChisurfClient
from chisurf.server.app import ChiSurfServer


from test.server.helpers import find_free_port


@pytest.fixture
def server_client():
    cmd_port = find_free_port()
    pub_port = find_free_port()
    server = ChiSurfServer(cmd_port=cmd_port, pub_port=pub_port)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.5)
    client = ChisurfClient(cmd_port=cmd_port, pub_port=pub_port)
    client.connect()
    yield client, server
    client.close()
    server.stop()


# ── Legacy round-trip tests ─────────────────────────────────────

def test_full_round_trip_list_methods(server_client):
    client, server = server_client
    methods = client.list_methods()
    expected = {
        "list_datasets", "get_dataset_info", "add_dataset", "remove_datasets",
        "clear_datasets",
        "list_fits", "get_fit_info", "run_fit", "remove_fits", "clear_fits",
        "get_parameter", "set_parameter_value", "set_parameter_fixed",
        "set_parameter_bounds",
        "save_project", "load_project", "get_project_info",
        "list_methods", "ping",
    }
    missing = expected - set(methods)
    assert not missing, f"Missing legacy methods: {missing}"


def test_full_round_trip_list_datasets(server_client):
    client, server = server_client
    datasets = client.list_datasets()
    assert datasets == []


def test_full_round_trip_unknown_method(server_client):
    client, server = server_client
    result = client._call("does_not_exist")
    assert not result.get("ok")
    assert "not found" in result.get("error", "")


def test_full_round_trip_list_fits(server_client):
    client, server = server_client
    fits = client.list_fits()
    assert fits == []


# ── Namespaced round-trip tests ─────────────────────────────────

def test_namespaced_methods_present(server_client):
    client, server = server_client
    methods = client.meta__methods()
    expected_ns = {
        "dataset.list", "dataset.get", "dataset.remove", "dataset.clear",
        "fit.list", "fit.get", "fit.run", "fit.remove", "fit.clear",
        "parameter.get", "parameter.set_value", "parameter.set_fixed",
        "parameter.set_bounds",
        "project.info", "project.save", "project.load",
        "meta.ping", "meta.methods",
    }
    missing = expected_ns - set(methods)
    assert not missing, f"Missing namespaced methods: {missing}"


def test_namespaced_ping(server_client):
    client, server = server_client
    result = client.call("meta.ping")
    assert result.get("ok") is True
    assert result.get("status") == "alive"


def test_namespaced_server_alive(server_client):
    client, server = server_client
    result = client.meta__ping()
    assert result.get("ok") is True
    assert result.get("status") == "alive"


def test_namespaced_dataset_list(server_client):
    client, server = server_client
    result = client.dataset__list()
    assert isinstance(result, list)
    assert result == []


def test_namespaced_fit_list(server_client):
    client, server = server_client
    result = client.fit__list()
    assert isinstance(result, list)
    assert result == []


def test_namespaced_project_info(server_client):
    client, server = server_client
    result = client.project__info()
    assert result.get("ok") is True
    assert result.get("fit_count") == 0
    assert result.get("dataset_count") == 0


def test_namespaced_dataset_clear(server_client):
    client, server = server_client
    result = client.dataset__clear()
    assert result.get("ok") is True


def test_namespaced_fit_clear(server_client):
    client, server = server_client
    result = client.fit__clear()
    assert result.get("ok") is True


def test_legacy_and_namespaced_ping_equivalent(server_client):
    client, server = server_client
    legacy = client.ping()
    namespaced = client.meta__ping()
    assert legacy.get("ok") == namespaced.get("ok")
    assert legacy.get("status") == namespaced.get("status")


def test_legacy_and_namespaced_list_methods_includes_both(server_client):
    client, server = server_client
    methods = client.list_methods()
    # Namespaced methods should coexist with legacy
    assert "ping" in methods
    assert "meta.ping" in methods
    assert "list_fits" in methods
    assert "fit.list" in methods


def test_server_events_broadcast_over_zmq(server_client):
    """Verify that mutations publish events over the ZMQ PUB socket."""
    client, server = server_client

    # Subscribe to all events via ZMQ SUB
    received = []

    def _on_event(event):
        received.append(event)

    token = client.subscribe(topic="fit.clear", callback=_on_event)
    assert token is not None
    time.sleep(0.3)  # let SUB socket connect

    # Clear fits (should publish "fit.cleared")
    result = client.fit__clear()
    assert result.get("ok") is True
    time.sleep(0.3)  # let event propagate
    client.drain()  # dispatch queued events on the main thread

    # The event should have been broadcast over ZMQ
    found = any(e.get("cleared_count") is not None for e in received)
    assert found, f"Expected fit.clear event, got: {received}"


def test_session_restore(server_client):
    """Verify session.restore works (clear + reload)."""
    client, server = server_client
    result = client.session__restore()
    assert result.get("ok") is True
    info = client.session__describe()
    assert info.get("fit_count") == 0
    assert info.get("dataset_count") == 0


def test_session_restore_with_project_nonexistent(server_client):
    """Verify session.restore returns error for non-existent project."""
    client, server = server_client
    result = client.session__restore(project_path="/nonexistent/path")
    assert not result.get("ok")


def test_remote_dataset_add(server_client):
    """Verify add_dataset via reader_name + filename works remotely."""
    client, server = server_client
    result = client.call("add_dataset", {
        "reader_name": "DataCurve",
        "filename": "/tmp/test_data.dat",
        "name": "RemoteTest",
    })
    assert result.get("ok") is True
    assert "uid" in result

    # Dataset should appear in server state
    datasets = client.dataset__list()
    uids = [d.get("uid") for d in datasets]
    assert result["uid"] in uids


def test_remote_dataset_add_with_curve_data(server_client):
    """Verify add_dataset populates x/y arrays when curve_data is sent."""
    client, server = server_client
    x_vals = [0.0, 1.0, 2.0, 3.0]
    y_vals = [1.0, 4.0, 9.0, 16.0]
    result = client.call("add_dataset", {
        "reader_name": "FakeReader",
        "filename": "/tmp/test_curve.dat",
        "name": "CurveDataTest",
        "curve_data": {"x": x_vals, "y": y_vals},
    })
    assert result.get("ok") is True

    # Verify dataset info includes length
    info = client.dataset__get(dataset_index=result["dataset_index"])
    assert info.get("length") == 4
    assert info.get("name") == "CurveDataTest"


def test_extract_curve_data_from_dataset():
    """Unit test: _extract_curve_data extracts x/y from a DataCurve-like object."""
    from chisurf.core.api import _extract_curve_data

    class FakeCurve:
        x = [0.0, 1.0, 2.0]
        y = [3.0, 4.0, 5.0]
        ex = [0.1, 0.1, 0.1]
        ey = [0.2, 0.2, 0.2]

    result = _extract_curve_data(FakeCurve())
    assert result is not None
    assert result["x"] == [0.0, 1.0, 2.0]
    assert result["y"] == [3.0, 4.0, 5.0]
    assert result["ex"] == [0.1, 0.1, 0.1]
    assert result["ey"] == [0.2, 0.2, 0.2]


def test_extract_curve_data_none_when_no_y(server_client):
    """_extract_curve_data returns None for objects without y data."""
    from chisurf.core.api import _extract_curve_data
    assert _extract_curve_data(object()) is None
    assert _extract_curve_data(None) is None


def test_dataset_curve_data_endpoint(server_client):
    """Verify dataset.curve_data returns x/y arrays."""
    client, server = server_client
    x_vals = [0.0, 1.0, 2.0, 3.0, 4.0]
    y_vals = [0.0, 1.0, 4.0, 9.0, 16.0]
    result = client.call("add_dataset", {
        "reader_name": "TestReader",
        "filename": "/tmp/curve_test.dat",
        "name": "CurveEndpointTest",
        "curve_data": {"x": x_vals, "y": y_vals},
    })
    assert result.get("ok") is True
    di = result["dataset_index"]

    # Fetch curve data via the new endpoint
    curve = client.dataset__curve_data(dataset_index=di)
    assert curve.get("ok") is True
    assert curve.get("x") == x_vals
    assert curve.get("y") == y_vals


def test_dataset_curve_data_invalid_index(server_client):
    """dataset.curve_data returns error for out-of-range index."""
    client, server = server_client
    curve = client.dataset__curve_data(dataset_index=999)
    assert not curve.get("ok")


def test_fit_save_endpoint(server_client):
    """Verify fit.save works on server-side fits."""
    import os, tempfile
    client, server = server_client
    # Add dataset with curve data
    ds = client.call("add_dataset", {
        "reader_name": "TestReader",
        "filename": "/tmp/fit_save_test.dat",
        "name": "FitSaveTest",
        "curve_data": {"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 4.0]},
    })
    assert ds.get("ok") is True

    # Create a fit (may fail if no suitable model class available)
    ft = client.fit__create(dataset_index=ds["dataset_index"], model_name="TCSPC")
    if not ft.get("ok"):
        pytest.skip("TCSPC model not available in this environment")
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        tmp = f.name
    try:
        result = client.fit__save(fit_index=ft["fit_index"], filename=tmp, file_type="csv", save_curves=True)
        assert result.get("ok") is True
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def test_fit_curve_data_endpoint(server_client):
    """Verify fit.curve_data returns curve data."""
    client, server = server_client
    ds = client.call("add_dataset", {
        "reader_name": "TestReader",
        "filename": "/tmp/fit_curve_test.dat",
        "name": "FitCurveTest",
        "curve_data": {"x": [0.0, 1.0, 2.0, 3.0], "y": [0.0, 1.0, 4.0, 9.0]},
    })
    assert ds.get("ok") is True

    ft = client.fit__create(dataset_index=ds["dataset_index"], model_name="TCSPC")
    if not ft.get("ok"):
        pytest.skip("TCSPC model not available in this environment")
    curve = client.fit__curve_data(fit_index=ft["fit_index"])
    assert curve.get("ok") is True
    # Should have at least x/y from the dataset
    assert "x" in curve
    assert "y" in curve


def test_fit_list_includes_chi2r(server_client):
    """Verify list_fits includes chi2r at top level and model.chi2r."""
    client, server = server_client
    ds = client.call("add_dataset", {
        "reader_name": "TestReader",
        "filename": "/tmp/chi2r_test.dat",
        "name": "Chi2rTest",
        "curve_data": {"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 4.0]},
    })
    assert ds.get("ok") is True

    ft = client.fit__create(dataset_index=ds["dataset_index"], model_name="TCSPC")
    if not ft.get("ok"):
        pytest.skip("TCSPC model not available in this environment")
    fits = client.fit__list()
    fit_dto = None
    for f in fits:
        if f.get("uid") == ft.get("uid"):
            fit_dto = f
            break
    assert fit_dto is not None
    # chi2 and chi2r should be at top level
    assert "chi2" in fit_dto
    assert "chi2r" in fit_dto
    # model sub-dict should also have chi2r
    model = fit_dto.get("model", {})
    assert "chi2r" in model


def test_fit_set_fit_range_endpoint(server_client):
    """Verify fit.set_fit_range works on server-side fit."""
    client, server = server_client
    ds = client.call("add_dataset", {
        "reader_name": "TestReader",
        "filename": "/tmp/fit_range_test.dat",
        "name": "FitRangeTest",
        "curve_data": {"x": list(range(100)), "y": [float(i * i) for i in range(100)]},
    })
    assert ds.get("ok") is True

    ft = client.fit__create(dataset_index=ds["dataset_index"], model_name="TCSPC")
    if not ft.get("ok"):
        pytest.skip("TCSPC model not available in this environment")
    result = client.fit__set_fit_range(fit_index=ft["fit_index"], xmin=10, xmax=50)
    assert result.get("ok") is True


def test_model_finalize_endpoint(server_client):
    """Verify model.finalize works via ZMQ."""
    client, server = server_client
    ds = client.call("add_dataset", {
        "reader_name": "MFReader",
        "filename": "/tmp/mf_test.dat",
        "name": "ModelFinalize",
        "curve_data": {"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 4.0]},
    })
    assert ds.get("ok") is True
    ft = client.fit__create(dataset_index=ds["dataset_index"], model_name="TCSPC")
    if not ft.get("ok"):
        pytest.skip("TCSPC model not available in this environment")
    result = client.model__finalize(fit_index=ft["fit_index"])
    assert result.get("ok") is True


def test_model_set_parse_function_endpoint(server_client):
    """Verify model.set_parse_function works via ZMQ."""
    client, server = server_client
    ds = client.call("add_dataset", {
        "reader_name": "MSPFReader",
        "filename": "/tmp/mspf_test.dat",
        "name": "ModelParse",
        "curve_data": {"x": [0.0, 1.0], "y": [2.0, 3.0]},
    })
    assert ds.get("ok") is True
    ft = client.fit__create(dataset_index=ds["dataset_index"], model_name="TCSPC")
    if not ft.get("ok"):
        pytest.skip("TCSPC model not available in this environment")
    result = client.model__set_parse_function(
        parse_function="y = a * exp(-x/tau)",
        fit_index=ft["fit_index"],
    )
    assert result.get("ok") is True


def test_dataset_rename_endpoint(server_client):
    """Verify dataset.rename works via ZMQ."""
    client, server = server_client
    ds = client.call("add_dataset", {
        "reader_name": "RenReader",
        "filename": "/tmp/ren_test.dat",
        "name": "OriginalName",
        "curve_data": {"x": [0.0], "y": [1.0]},
    })
    assert ds.get("ok") is True
    result = client.dataset__rename("RenamedDataset", dataset_index=ds["dataset_index"])
    assert result.get("ok") is True
    assert result.get("new_name") == "RenamedDataset"
    info = client.dataset__get(dataset_index=ds["dataset_index"])
    assert info.get("name") == "RenamedDataset"


def test_fit_set_dataset_endpoint(server_client):
    """Verify fit.set_dataset works via ZMQ."""
    client, server = server_client
    ds1 = client.call("add_dataset", {
        "reader_name": "SDS1", "filename": "/tmp/sds1.dat",
        "name": "Data1", "curve_data": {"x": [0.0], "y": [1.0]},
    })
    ds2 = client.call("add_dataset", {
        "reader_name": "SDS2", "filename": "/tmp/sds2.dat",
        "name": "Data2", "curve_data": {"x": [0.0], "y": [2.0]},
    })
    assert ds1.get("ok") and ds2.get("ok")
    ft = client.fit__create(dataset_index=ds1["dataset_index"], model_name="TCSPC")
    if not ft.get("ok"):
        pytest.skip("TCSPC model not available in this environment")
    result = client.fit__set_dataset(
        fit_index=ft["fit_index"],
        dataset_index=ds2["dataset_index"],
    )
    assert result.get("ok") is True


def test_fit_set_result_idx_endpoint(server_client):
    """Verify fit.set_result_idx works via ZMQ."""
    client, server = server_client
    ds = client.call("add_dataset", {
        "reader_name": "SRIReader", "filename": "/tmp/sri.dat",
        "name": "ResultIdxTest", "curve_data": {"x": [0.0], "y": [1.0]},
    })
    assert ds.get("ok") is True
    ft = client.fit__create(dataset_index=ds["dataset_index"], model_name="TCSPC")
    if not ft.get("ok"):
        pytest.skip("TCSPC model not available in this environment")
    result = client.fit__set_result_idx(fit_index=ft["fit_index"], result_idx=1)
    assert result.get("ok") is True


def test_event_broadcast_on_dataset_add(server_client):
    """Verify ZMQ event broadcast on dataset.added."""
    client, server = server_client
    received = []

    def _on_event(event):
        received.append(event)

    token = client.subscribe(topic="dataset.added", callback=_on_event)
    assert token is not None
    import time
    time.sleep(0.3)

    result = client.call("add_dataset", {
        "reader_name": "EvtReader", "filename": "/tmp/evt.dat",
        "name": "EventTest", "curve_data": {"x": [0.0], "y": [1.0]},
    })
    assert result.get("ok") is True
    time.sleep(0.3)
    client.drain()  # dispatch queued events on the main thread

    found = any(e.get("dataset_index") is not None for e in received)
    assert found, f"Expected dataset.added event, got: {received}"


def test_event_broadcast_on_dataset_remove(server_client):
    """Verify ZMQ event broadcast on dataset.removed."""
    client, server = server_client
    ds = client.call("add_dataset", {
        "reader_name": "RemEvt", "filename": "/tmp/remevt.dat",
        "name": "RemoveEvent", "curve_data": {"x": [0.0], "y": [1.0]},
    })
    assert ds.get("ok") is True
    received = []

    def _on_event(event):
        received.append(event)

    token = client.subscribe(topic="dataset.removed", callback=_on_event)
    import time
    time.sleep(0.3)

    result = client.dataset__remove(dataset_indices=[ds["dataset_index"]])
    assert result.get("ok") is True
    time.sleep(0.3)
    client.drain()  # dispatch queued events on the main thread

    found = any(e.get("removed_count") is not None for e in received)
    assert found, f"Expected dataset.removed event, got: {received}"


def test_graph_build_fits_endpoint(server_client):
    """Verify graph.build_fits works via ZMQ."""
    client, server = server_client
    result = client.graph__build_fits()
    assert result.get("ok") is True
    assert "graph" in result
    assert result["graph"]["nodes"] == []


def test_parameter_link_endpoint(server_client):
    """Verify parameter.link works via ZMQ."""
    client, server = server_client
    ds = client.call("add_dataset", {
        "reader_name": "LinkReader", "filename": "/tmp/link.dat",
        "name": "LinkTest", "curve_data": {"x": [0.0], "y": [1.0]},
    })
    assert ds.get("ok") is True
    ft = client.fit__create(dataset_index=ds["dataset_index"], model_name="TCSPC")
    if not ft.get("ok"):
        pytest.skip("TCSPC model not available in this environment")
    info = client.fit__get(fit_index=ft["fit_index"])
    params = info.get("model", {}).get("parameters_all", [])
    if len(params) >= 2:
        result = client.parameter__link(
            params[0]["name"], params[1]["name"],
            fit_index=ft["fit_index"],
        )
        assert result.get("ok") is True
        result2 = client.parameter__unlink(params[0]["name"], fit_index=ft["fit_index"])
        assert result2.get("ok") is True


def test_event_broadcast_on_fit_run(server_client):
    """Verify ZMQ event broadcast on fit.run."""
    client, server = server_client
    ds = client.call("add_dataset", {
        "reader_name": "RunEvt", "filename": "/tmp/run_evt.dat",
        "name": "RunEventTest", "curve_data": {"x": [0.0], "y": [1.0]},
    })
    assert ds.get("ok") is True
    ft = client.fit__create(dataset_index=ds["dataset_index"], model_name="TCSPC")
    if not ft.get("ok"):
        pytest.skip("TCSPC model not available in this environment")
    received = []

    def _on_event(event):
        received.append(event)

    token = client.subscribe(topic="fit.ran", callback=_on_event)
    import time
    time.sleep(0.3)

    result = client.fit__run(fit_index=ft["fit_index"])
    time.sleep(0.3)
    client.drain()  # dispatch queued events on the main thread
    # fit.run may fail without real model, but should still publish event
    found = any(e.get("fit_index") is not None for e in received)
    assert found, f"Expected fit.ran event, got: {received}"


class TestMetaProtocol:

    def test_meta_ping_returns_version(self, server_client):
        client, server = server_client
        raw = client._client.call("meta.ping")
        assert raw.get("id") is not None
        result = raw.get("result", {})
        assert result.get("ok") is True
        assert result.get("status") == "alive"
        assert isinstance(result.get("version"), str)
        assert isinstance(result.get("protocol_version"), str)
        assert isinstance(result.get("dataset_count"), int)
        assert isinstance(result.get("fit_count"), int)

    def test_meta_protocol_returns_catalogue(self, server_client):
        client, server = server_client
        result = client.call("meta.protocol")
        assert result.get("ok") is True
        assert isinstance(result.get("protocol_version"), str)
        catalogue = result.get("catalogue", {})
        assert "meta" in catalogue
        assert "dataset" in catalogue
        assert "fit" in catalogue
        assert "parameter" in catalogue
        assert "project" in catalogue
        assert "session" in catalogue
        assert "model" in catalogue
        assert "graph" in catalogue
        for ns, info in catalogue.items():
            assert "description" in info
            assert isinstance(info["methods"], list)
        schemas = result.get("schemas", {})
        assert "meta.ping" in schemas
        assert "dataset.list" in schemas
        assert "fit.create" in schemas
        assert schemas["fit.create"]["result"] == "FitCreateResult"
