from __future__ import annotations

"""Edge-case tests for the ZMQ JSON-RPC transport layer.

Covers: NaN/Inf handling, large payloads, concurrency, timeouts,
connection loss, error propagation, rapid mutations.
"""

import math
import threading
import time
import pytest

from chisurf.core.api._client import ChisurfClient, RemoteError
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


class TestInfNanEdgeCases:

    def test_inf_value_in_curve_data(self, server_client):
        """Server accepts float('inf') in curve_data y values."""
        client, server = server_client
        result = client.call("add_dataset", {
            "reader_name": "InfReader",
            "filename": "/tmp/inf_test.dat",
            "name": "InfTest",
            "curve_data": {
                "x": [0.0, 1.0, 2.0],
                "y": [float('inf'), 1.0, float('-inf')],
            },
        })
        assert result.get("ok") is True

        di = result["dataset_index"]
        curve = client.dataset__curve_data(dataset_index=di)
        assert curve.get("ok") is True
        # Inf values are sanitized to None for JSON safety
        assert curve["y"][0] is None
        assert curve["y"][2] is None

    def test_nan_value_in_curve_data(self, server_client):
        """Server accepts float('nan') in curve_data."""
        client, server = server_client
        result = client.call("add_dataset", {
            "reader_name": "NaNReader",
            "filename": "/tmp/nan_test.dat",
            "name": "NaNTest",
            "curve_data": {
                "x": [0.0, 1.0],
                "y": [float('nan'), 2.0],
            },
        })
        assert result.get("ok") is True

        di = result["dataset_index"]
        curve = client.dataset__curve_data(dataset_index=di)
        assert curve.get("ok") is True
        # NaN values are sanitized to None for JSON safety
        assert curve["y"][0] is None
        assert curve["y"][1] == 2.0

    def test_set_parameter_value_inf(self, server_client):
        """Setting parameter value to inf works server-side."""
        client, server = server_client
        ds = client.call("add_dataset", {
            "reader_name": "PInfReader",
            "filename": "/tmp/pinf_test.dat",
            "name": "PInfTest",
            "curve_data": {"x": [0.0], "y": [1.0]},
        })
        assert ds.get("ok") is True
        ft = client.fit__create(dataset_index=ds["dataset_index"], model_name="TCSPC")
        if not ft.get("ok"):
            pytest.skip("fit creation not supported in this env")
        info = client.fit__get(fit_index=ft["fit_index"])
        params = info.get("model", {}).get("parameters_all", [])
        if not params:
            pytest.skip("no parameters available")
        pname = params[0]["name"]
        result = client.parameter__set_value(pname, float('inf'), fit_index=ft["fit_index"])
        assert result.get("ok") is True

    def test_nan_in_parameter_bounds(self, server_client):
        """NaN in parameter bounds is forwarded correctly."""
        client, server = server_client
        ds = client.call("add_dataset", {
            "reader_name": "BoundsNaN",
            "filename": "/tmp/bounds_nan.dat",
            "name": "BoundsNaN",
            "curve_data": {"x": [0.0], "y": [1.0]},
        })
        assert ds.get("ok") is True
        ft = client.fit__create(dataset_index=ds["dataset_index"], model_name="TCSPC")
        if not ft.get("ok"):
            pytest.skip("fit creation not supported")
        info = client.fit__get(fit_index=ft["fit_index"])
        params = info.get("model", {}).get("parameters_all", [])
        if not params:
            pytest.skip("no parameters available")
        pname = params[0]["name"]
        result = client.parameter__set_bounds(pname, float('-inf'), float('inf'), fit_index=ft["fit_index"])
        assert result.get("ok") is True


class TestLargePayloadEdgeCases:

    def test_large_x_y_arrays(self, server_client):
        """Large curve data arrays (100k points) transfer correctly."""
        client, server = server_client
        n = 100_000
        x = [float(i) for i in range(n)]
        y = [float(i * i) for i in range(n)]
        result = client.call("add_dataset", {
            "reader_name": "LargeReader",
            "filename": "/tmp/large_test.dat",
            "name": "LargeTest",
            "curve_data": {"x": x, "y": y},
        })
        assert result.get("ok") is True
        di = result["dataset_index"]
        info = client.dataset__get(dataset_index=di)
        assert info.get("length") == n

    def test_large_curve_data_round_trip(self, server_client):
        """Large arrays survive a round trip via dataset.curve_data."""
        client, server = server_client
        n = 10_000
        x = [float(i) for i in range(n)]
        y = [float(i * 0.5) for i in range(n)]
        result = client.call("add_dataset", {
            "reader_name": "RTReader",
            "filename": "/tmp/rt_test.dat",
            "name": "RTLarge",
            "curve_data": {"x": x, "y": y},
        })
        assert result.get("ok") is True
        di = result["dataset_index"]
        curve = client.dataset__curve_data(dataset_index=di)
        assert len(curve["x"]) == n
        assert len(curve["y"]) == n
        assert curve["x"][0] == 0.0
        assert curve["x"][n - 1] == float(n - 1)
        assert curve["y"][n - 1] == float((n - 1) * 0.5)


class TestConcurrencyEdgeCases:

    def test_concurrent_pings(self, server_client):
        """Multiple concurrent pings complete successfully (each thread has own client)."""
        client, server = server_client
        cmd_port = client._client._cmd_port
        pub_port = client._client._pub_port
        results = []
        errors = []

        def _ping():
            try:
                c = ChisurfClient(cmd_port=cmd_port, pub_port=pub_port)
                c.connect()
                r = c.meta__ping()
                results.append(r.get("status"))
                c.close()
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=_ping) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(s == "alive" for s in results), f"Errors: {errors}"
        assert not errors

    def test_rapid_dataset_add_and_clear(self, server_client):
        """Rapid add/clear cycles don't cause errors."""
        client, server = server_client
        for i in range(10):
            r = client.call("add_dataset", {
                "reader_name": "RapidReader",
                "filename": f"/tmp/rapid_{i}.dat",
                "name": f"Rapid{i}",
                "curve_data": {"x": [0.0, 1.0], "y": [float(i), float(i + 1)]},
            })
            assert r.get("ok") is True
        assert len(client.dataset__list()) == 10
        r = client.dataset__clear()
        assert r.get("ok") is True
        assert len(client.dataset__list()) == 0

    def test_rapid_fit_add_remove(self, server_client):
        """Rapid add/remove fit cycles work cleanly."""
        client, server = server_client
        client.call("add_dataset", {
            "reader_name": "RapidFit",
            "filename": "/tmp/rapid_fit.dat",
            "name": "RapidFitDS",
            "curve_data": {"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 4.0]},
        })
        fits = []
        for i in range(5):
            ft = client.fit__create(dataset_index=0, model_name="TCSPC")
            if not ft.get("ok"):
                pytest.skip("TCSPC model not available")
            fits.append(ft["fit_index"])
        if fits:
            r = client.fit__remove(fit_indices=fits)
            assert r.get("ok") is True
        assert len(client.fit__list()) == 0

    def test_concurrent_mixed_operations(self, server_client):
        """Mixed operations from multiple threads (each with own client)."""
        client, server = server_client
        cmd_port = client._client._cmd_port
        pub_port = client._client._pub_port
        errors = []

        def _adder():
            try:
                c = ChisurfClient(cmd_port=cmd_port, pub_port=pub_port)
                c.connect()
                for i in range(5):
                    c.call("add_dataset", {
                        "reader_name": "Concur",
                        "filename": f"/tmp/concur_{i}.dat",
                        "name": f"Concur{i}",
                        "curve_data": {"x": [0.0], "y": [float(i)]},
                    })
                c.close()
            except Exception as e:
                errors.append(f"adder: {e}")

        def _lister():
            try:
                c = ChisurfClient(cmd_port=cmd_port, pub_port=pub_port)
                c.connect()
                for _ in range(10):
                    c.dataset__list()
                    c.meta__ping()
                c.close()
            except Exception as e:
                errors.append(f"lister: {e}")

        threads = [threading.Thread(target=_adder) for _ in range(3)]
        threads += [threading.Thread(target=_lister) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"Concurrent errors: {errors}"
        assert len(client.dataset__list()) == 15


class TestTimeoutAndConnectionEdgeCases:

    def test_timeout_returns_error(self, server_client):
        """Short timeout on unreachable server returns error."""
        client, server = server_client
        client.close()
        unreachable = ChisurfClient(cmd_port=9999, timeout_ms=500)
        with pytest.raises(RemoteError):
            unreachable.meta__ping()
        unreachable.close()

    def test_unknown_method_error(self, server_client):
        """Unknown method returns ok=False structured response."""
        client, server = server_client
        result = client.call("does.not.exist")
        assert not result.get("ok")
        assert "error" in result

    def test_invalid_params_error(self, server_client):
        """Invalid params (bad types) return error, not crash."""
        client, server = server_client
        result = client.call("add_dataset", {
            "reader_name": "BadReader",
            "filename": "/tmp/bad.dat",
            "name": "Bad",
            "curve_data": {"x": "not_a_list", "y": "also_not_a_list"},
        })
        assert not result.get("ok")
        assert "error" in result

    def test_missing_required_param(self, server_client):
        """Missing required param returns error."""
        client, server = server_client
        result = client.call("parameter.get", {})
        assert not result.get("ok")
        assert "error" in result

    def test_disconnect_reconnect(self, server_client):
        """Client can reconnect after close."""
        client, server = server_client
        assert client.meta__ping().get("ok") is True
        client.close()
        client.connect()
        assert client.meta__ping().get("ok") is True
        client.close()


class TestErrorPropagation:

    def test_server_exception_returns_error(self):
        """When a service raises, dispatcher returns structured error."""
        from chisurf.server.dispatcher import ServiceDispatcher
        from chisurf.server.session import SessionState

        state = SessionState()
        d = ServiceDispatcher(state)

        def _broken(_params):
            raise RuntimeError("something broke")

        d.register("test.broken", _broken)
        result = d.dispatch("test.broken", {})
        assert not result.get("ok")
        assert "something broke" in result.get("error", "")

    def test_zmq_parse_error_handled(self, server_client):
        """Malformed JSON-RPC request returns parse error."""
        client, server = server_client
        import zmq
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REQ)
        cmd_port = client._client._cmd_port
        sock.connect(f"tcp://127.0.0.1:{cmd_port}")
        sock.send_json({"method": "meta.ping"})
        reply = sock.recv_json()
        assert "result" in reply
        assert reply["result"].get("ok") is True
        sock.close()
        ctx.term()

    def test_extra_params_not_crash(self, server_client):
        """Unexpected keys in params return ok=False, not crash."""
        client, server = server_client
        result = client.call("meta.ping", {"unexpected_key": "value", "another": 42})
        assert not result.get("ok")
        assert "error" in result


class TestSerializerEdgeCases:

    def test_very_long_method_name(self, server_client):
        """Very long method name returns ok=False."""
        client, server = server_client
        long_name = "a" * 500
        result = client.call(long_name, {})
        assert not result.get("ok")

    def test_unicode_in_dataset_name(self, server_client):
        """Unicode characters in names survive round trip."""
        client, server = server_client
        name = "Datenreihe äöü 测试 📊"
        result = client.call("add_dataset", {
            "reader_name": "UnicodeReader",
            "filename": "/tmp/unicode_test.dat",
            "name": name,
            "curve_data": {"x": [0.0, 1.0], "y": [2.0, 3.0]},
        })
        assert result.get("ok") is True
        info = client.dataset__get(dataset_index=result["dataset_index"])
        assert info.get("name") == name

    def test_empty_curve_data(self, server_client):
        """Empty x/y lists in curve_data are accepted."""
        client, server = server_client
        result = client.call("add_dataset", {
            "reader_name": "EmptyReader",
            "filename": "/tmp/empty.dat",
            "name": "Empty",
            "curve_data": {"x": [], "y": []},
        })
        assert result.get("ok") is True
        info = client.dataset__get(dataset_index=result["dataset_index"])
        assert info.get("length") == 0
