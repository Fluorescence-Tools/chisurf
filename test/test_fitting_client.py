"""Tests for the FittingClient ZMQ adapter.

These tests verify that the FittingClient correctly wraps the
ChisurfClient and routes all operations through JSON-RPC.
They use a mock/stand-in server for round-trip tests and verify
error handling for invalid inputs.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

from chisurf.gui.widgets.fitting.fitting_client import (
    FittingClient,
    install_fitting_client,
    get_fitting_client,
    has_fitting_client,
)
from chisurf.core.api._client import ChisurfClient, RemoteError


class MockZmqClient:
    """A mock ZmqClient that simulates server responses."""

    def __init__(self, responses: Optional[Dict[str, Any]] = None):
        self.responses = responses or {}
        self.calls: list[tuple[str, Optional[Dict]]] = []
        self._sub_socket = None
        self._sub_thread = None

    def connect(self):
        pass

    def close(self):
        pass

    def call(self, method: str, params: Optional[Dict] = None) -> Dict:
        self.calls.append((method, params))
        if method in self.responses:
            return self.responses[method]
        return {"ok": True, "result": {}}

    def subscribe(self, topic, callback):
        return None

    def drain(self):
        pass


class MockChisurfClient:
    """A mock ChisurfClient for testing the FittingClient."""

    def __init__(self):
        self._client = MockZmqClient()
        self._responses: Dict[str, Any] = {}

    def set_response(self, method: str, response: Dict[str, Any]) -> None:
        self._responses[method] = response

    def call(self, method: str, params: Optional[Dict] = None) -> Dict:
        # Always record the call for param-verification tests
        recorded = self._client.call(method, params)
        # Return overridden response if registered
        key = f"{method}:{json.dumps(params or {}, sort_keys=True)}"
        if key in self._responses:
            return self._responses[key]
        if method in self._responses:
            return self._responses[method]
        return recorded

    def connect(self):
        pass

    def close(self):
        pass

    def subscribe(self, topic="", callback=None):
        return None

    def drain(self):
        pass


@pytest.fixture
def mock_client():
    return MockChisurfClient()


@pytest.fixture
def fc(mock_client):
    return FittingClient(mock_client)


# ── Fit CRUD ─────────────────────────────────────────────────────────

class TestListFits:
    def test_returns_empty_list(self, fc, mock_client):
        mock_client.set_response("fit.list", {"ok": True, "fits": []})
        result = fc.list_fits()
        assert result == []

    def test_returns_fit_list(self, fc, mock_client):
        mock_client.set_response("fit.list", {
            "ok": True,
            "fits": [
                {"uid": "abc", "index": 0, "name": "fit1"},
                {"uid": "def", "index": 1, "name": "fit2"},
            ],
        })
        result = fc.list_fits()
        assert len(result) == 2
        assert result[0]["name"] == "fit1"
        assert result[1]["uid"] == "def"

    def test_handles_missing_fits_key(self, fc, mock_client):
        mock_client.set_response("fit.list", {"ok": True})
        result = fc.list_fits()
        assert result == []


class TestGetFit:
    def test_by_uid(self, fc, mock_client):
        mock_client.set_response("fit.get", {
            "ok": True,
            "fit": {"uid": "abc", "name": "test_fit"},
        })
        result = fc.get_fit(fit_uid="abc")
        assert result["uid"] == "abc"
        assert result["name"] == "test_fit"

    def test_by_index(self, fc, mock_client):
        mock_client.set_response("fit.get", {
            "ok": True,
            "fit": {"index": 2, "name": "fit_by_index"},
        })
        result = fc.get_fit(fit_index=2)
        assert result["index"] == 2

    def test_not_found(self, fc, mock_client):
        mock_client.set_response("fit.get", {"ok": False, "error": "fit not found"})
        result = fc.get_fit(fit_uid="nonexistent")
        assert result == {}


class TestCreateFit:
    def test_creates_fit(self, fc, mock_client):
        mock_client.set_response("fit.create", {
            "ok": True,
            "uid": "new_uid",
            "fit_index": 0,
            "name": "new_fit",
        })
        result = fc.create_fit(dataset_indices=[0], model_name="TCSPC")
        assert result["uid"] == "new_uid"
        assert result["fit_index"] == 0


class TestRemoveFits:
    def test_remove_by_uid(self, fc, mock_client):
        mock_client.set_response("fit.remove", {
            "ok": True,
            "removed_count": 1,
            "remaining_count": 2,
        })
        result = fc.remove_fits(fit_uids=["abc"])
        assert result["removed_count"] == 1

    def test_remove_nonexistent(self, fc, mock_client):
        mock_client.set_response("fit.remove", {
            "ok": False,
            "error": "no fits specified for removal",
        })
        result = fc.remove_fits()
        assert result.get("ok") is False


class TestReorderFits:
    def test_reorder(self, fc, mock_client):
        mock_client.set_response("fit.reorder", {
            "ok": True,
            "count": 3,
        })
        result = fc.reorder_fits(["a", "b", "c"])
        assert result["count"] == 3


# ── Fit actions ──────────────────────────────────────────────────────

class TestRunFit:
    def test_calls_run(self, fc, mock_client):
        mock_client.set_response("fit.run", {
            "ok": True,
            "fit_index": 0,
            "chi2_before": 100.0,
            "chi2_after": 1.5,
        })
        result = fc.run_fit(fit_index=0)
        assert result["chi2_after"] == 1.5


class TestUpdateFit:
    def test_calls_update(self, fc, mock_client):
        mock_client.set_response("fit.update", {"ok": True})
        result = fc.update_fit(fit_uid="abc")
        assert result["ok"] is True


class TestSaveFit:
    def test_calls_save(self, fc, mock_client):
        mock_client.set_response("fit.save", {
            "ok": True,
            "saved_to": "/tmp/test.csv",
        })
        result = fc.save_fit(filename="/tmp/test.csv", fit_uid="abc")
        assert result["saved_to"] == "/tmp/test.csv"


# ── Fit range ────────────────────────────────────────────────────────

class TestSetFitRange:
    def test_sets_range(self, fc, mock_client):
        mock_client.set_response("fit.set_fit_range", {"ok": True})
        result = fc.set_fit_range(fit_uid="abc", xmin=10, xmax=100)
        assert result["ok"] is True


class TestAutoFitRange:
    def test_returns_range(self, fc, mock_client):
        mock_client.set_response("fit.range.auto", {
            "ok": True,
            "xmin": 0,
            "xmax": 1024,
        })
        result = fc.auto_fit_range(fit_uid="abc")
        assert result["xmin"] == 0
        assert result["xmax"] == 1024


class TestSetFitMask:
    def test_sets_mask(self, fc, mock_client):
        mock_client.set_response("fit.mask.set", {"ok": True})
        result = fc.set_fit_mask(mask=[1.0, 1.0, 0.0], fit_uid="abc")
        assert result["ok"] is True


# ── Parameter operations ─────────────────────────────────────────────

class TestGetParameter:
    def test_returns_parameter(self, fc, mock_client):
        mock_client.set_response("parameter.get", {
            "ok": True,
            "parameter": {
                "name": "tau1",
                "value": 3.5,
                "fixed": False,
                "bounds": [0.1, 10.0],
                "bounds_on": True,
                "error_estimate": 0.1,
            },
        })
        result = fc.get_parameter(parameter_name="tau1", fit_index=0)
        assert result["name"] == "tau1"
        assert result["value"] == 3.5


class TestSetParameterValue:
    def test_sets_value(self, fc, mock_client):
        mock_client.set_response("parameter.set_value", {"ok": True})
        result = fc.set_parameter_value("tau1", 4.0, fit_index=0)
        assert result["ok"] is True


class TestSetParameterFixed:
    def test_fixes_parameter(self, fc, mock_client):
        mock_client.set_response("parameter.set_fixed", {"ok": True})
        result = fc.set_parameter_fixed("tau1", True, fit_index=0)
        assert result["ok"] is True

    def test_frees_parameter(self, fc, mock_client):
        mock_client.set_response("parameter.set_fixed", {"ok": True})
        result = fc.set_parameter_fixed("tau1", False, fit_index=0)
        assert result["ok"] is True


class TestSetParameterBounds:
    def test_sets_bounds(self, fc, mock_client):
        mock_client.set_response("parameter.set_bounds", {"ok": True})
        result = fc.set_parameter_bounds("tau1", (0.1, 10.0), fit_index=0)
        assert result["ok"] is True


class TestSetParameterBoundsOn:
    def test_enables_bounds(self, fc, mock_client):
        mock_client.set_response("parameter.set_bounds_on", {"ok": True})
        result = fc.set_parameter_bounds_on("tau1", True, fit_index=0)
        assert result["ok"] is True


class TestLinkUnlink:
    def test_link_params(self, fc, mock_client):
        mock_client.set_response("parameter.link", {"ok": True})
        result = fc.link_parameters("tau1", "tau2", fit_index=0, target_fit_index=1)
        assert result["ok"] is True

    def test_unlink_param(self, fc, mock_client):
        mock_client.set_response("parameter.unlink", {"ok": True})
        result = fc.unlink_parameter("tau1", fit_index=0)
        assert result["ok"] is True


# ── Model operations ─────────────────────────────────────────────────

class TestModelComponent:
    def test_add_component(self, fc, mock_client):
        mock_client.set_response("model.component.add", {"ok": True})
        result = fc.model_add_component("lifetime", fit_uid="abc")
        assert result["ok"] is True

    def test_remove_component(self, fc, mock_client):
        mock_client.set_response("model.component.remove", {"ok": True})
        result = fc.model_remove_component(component_index=1, fit_uid="abc")
        assert result["ok"] is True


class TestModelState:
    def test_get_state(self, fc, mock_client):
        mock_client.set_response("model.state.get", {
            "ok": True,
            "state": {"convolve": True, "n_components": 3},
        })
        result = fc.model_get_state(fit_uid="abc")
        assert result["convolve"] is True
        assert result["n_components"] == 3

    def test_set_state(self, fc, mock_client):
        mock_client.set_response("model.state.set", {"ok": True})
        result = fc.model_set_state({"convolve": True}, fit_uid="abc")
        assert result["ok"] is True


# ── Fit selection & group ────────────────────────────────────────────

class TestFitSelect:
    def test_select_by_uid(self, fc, mock_client):
        mock_client.set_response("fit.select", {"ok": True, "fit": {"uid": "abc"}})
        result = fc.select_fit(fit_uid="abc")
        assert result.get("ok") is True

    def test_get_active(self, fc, mock_client):
        mock_client.set_response("fit.select", {"ok": True, "fit": {"uid": "active"}})
        result = fc.get_active_fit()
        assert result.get("uid") == "active"


class TestFitGroup:
    def test_select_member(self, fc, mock_client):
        mock_client.set_response("fit.group.select_member", {"ok": True})
        result = fc.group_select_member("group_uid", 0)
        assert result["ok"] is True

    def test_add_member(self, fc, mock_client):
        mock_client.set_response("fit.group.add_member", {"ok": True})
        result = fc.group_add_member("group_uid", "member_uid")
        assert result["ok"] is True

    def test_remove_member(self, fc, mock_client):
        mock_client.set_response("fit.group.remove_member", {"ok": True})
        result = fc.group_remove_member("group_uid", 0)
        assert result["ok"] is True

    def test_link_by_name(self, fc, mock_client):
        mock_client.set_response("fit.group.link_parameters_by_name", {"ok": True})
        result = fc.group_link_parameters_by_name("group_uid", "tau1")
        assert result["ok"] is True


# ── Sampling ─────────────────────────────────────────────────────────

class TestSampling:
    def test_start(self, fc, mock_client):
        mock_client.set_response("fit.sample.start", {
            "ok": True,
            "job_id": "job_123",
        })
        result = fc.start_sampling(fit_uid="abc", n_steps=1000, n_runs=2)
        assert result["job_id"] == "job_123"

    def test_status(self, fc, mock_client):
        mock_client.set_response("fit.sample.status", {
            "ok": True,
            "job_id": "job_123",
            "status": "running",
            "progress": 50,
        })
        result = fc.sampling_status("job_123")
        assert result["status"] == "running"
        assert result["progress"] == 50

    def test_cancel(self, fc, mock_client):
        mock_client.set_response("fit.sample.cancel", {"ok": True})
        result = fc.cancel_sampling("job_123")
        assert result["ok"] is True


# ── Parameter scan ───────────────────────────────────────────────────

class TestParameterScan:
    def test_start(self, fc, mock_client):
        mock_client.set_response("fit.parameter_scan.start", {
            "ok": True,
            "job_id": "scan_123",
        })
        result = fc.start_parameter_scan("tau1", fit_uid="abc")
        assert result["job_id"] == "scan_123"

    def test_result(self, fc, mock_client):
        mock_client.set_response("fit.parameter_scan.result", {
            "ok": True,
            "values": [1.0, 2.0, 3.0],
            "chi2": [100.0, 10.0, 100.0],
        })
        result = fc.parameter_scan_result("scan_123")
        assert len(result["values"]) == 3

    def test_cancel(self, fc, mock_client):
        mock_client.set_response("fit.parameter_scan.cancel", {"ok": True})
        result = fc.cancel_parameter_scan("scan_123")
        assert result["ok"] is True


# ── Plot data ────────────────────────────────────────────────────────

class TestPlotData:
    def test_get_plot_data(self, fc, mock_client):
        mock_client.set_response("plot.fit_data", {
            "ok": True,
            "plot": {
                "type": "fit_data",
                "curves": [{"label": "data", "x": [1, 2], "y": [3, 4]}],
            },
        })
        result = fc.get_plot_data(plot_type="fit_data", fit_uid="abc")
        assert result["type"] == "fit_data"
        assert len(result["curves"]) == 1


# ── Global adapter ───────────────────────────────────────────────────

class TestGlobalAdapter:
    def test_install_and_get(self):
        mock = MockChisurfClient()
        client = install_fitting_client(mock)
        assert client is not None
        assert has_fitting_client() is True
        assert get_fitting_client() is client

    def test_get_before_install(self):
        from chisurf.gui.widgets.fitting.fitting_client import (
            _FITTING_CLIENT,
        )
        _FITTING_CLIENT = None
        # Re-import won't help; just test get_fitting_client returns None
        # Actually we need to reset the module state
        import importlib
        import chisurf.gui.widgets.fitting.fitting_client as fcm
        importlib.reload(fcm)
        # After reload, should be None
        from chisurf.gui.widgets.fitting.fitting_client import (
            get_fitting_client,
            has_fitting_client,
        )
        # Note: after reload the module-level global is reset
        assert get_fitting_client() is None
        assert has_fitting_client() is False


# ── Fit count convenience ────────────────────────────────────────────

class TestFitCount:
    def test_fit_count(self, fc, mock_client):
        mock_client.set_response("fit.list", {
            "ok": True,
            "fits": [{"uid": "a"}, {"uid": "b"}, {"uid": "c"}],
        })
        assert fc.fit_count() == 3

    def test_parameter_dict(self, fc, mock_client):
        mock_client.set_response("fit.get", {
            "ok": True,
            "fit": {"parameters": {
                "tau1": {"value": 3.5},
                "tau2": {"value": 1.0},
            }},
        })
        params = fc.parameter_dict(fit_uid="abc")
        assert "tau1" in params
        assert params["tau1"]["value"] == 3.5


# ── Param verification ─────────────────────────────────────────────────

def _last_call(mock_client):
    """Return (method, params) of the last RPC call made to the mock."""
    calls = mock_client._client.calls
    if not calls:
        return None, None
    return calls[-1]


class TestParamVerification:
    """Verify the exact method names and parameters sent to the server.

    These tests catch silent protocol mismatches where a FittingClient
    method sends different params than what the server endpoint expects.
    """

    def test_list_fits(self, fc, mock_client):
        mock_client.set_response("fit.list", {"ok": True, "fits": []})
        fc.list_fits()
        method, params = _last_call(mock_client)
        assert method == "fit.list"
        assert params == {}

    def test_get_fit_by_uid(self, fc, mock_client):
        mock_client.set_response("fit.get", {"ok": True, "fit": {}})
        fc.get_fit(fit_uid="abc")
        method, params = _last_call(mock_client)
        assert method == "fit.get"
        assert params == {"fit_uid": "abc"}

    def test_get_fit_by_index(self, fc, mock_client):
        mock_client.set_response("fit.get", {"ok": True, "fit": {}})
        fc.get_fit(fit_index=3)
        method, params = _last_call(mock_client)
        assert method == "fit.get"
        assert params == {"fit_index": 3}

    def test_run_fit(self, fc, mock_client):
        mock_client.set_response("fit.run", {"ok": True})
        fc.run_fit(fit_uid="abc")
        method, params = _last_call(mock_client)
        assert method == "fit.run"
        assert params == {"fit_uid": "abc"}

    def test_update_fit(self, fc, mock_client):
        mock_client.set_response("fit.update", {"ok": True})
        fc.update_fit(fit_uid="abc")
        method, params = _last_call(mock_client)
        assert method == "fit.update"
        assert params == {"fit_uid": "abc"}

    def test_set_fit_range(self, fc, mock_client):
        mock_client.set_response("fit.set_fit_range", {"ok": True})
        fc.set_fit_range(fit_uid="abc", xmin=10, xmax=100)
        method, params = _last_call(mock_client)
        assert method == "fit.set_fit_range"
        assert params == {"fit_uid": "abc", "xmin": 10, "xmax": 100}

    def test_set_fit_mask(self, fc, mock_client):
        mock_client.set_response("fit.mask.set", {"ok": True})
        fc.set_fit_mask(mask=[1.0, 0.0], fit_uid="abc")
        method, params = _last_call(mock_client)
        assert method == "fit.mask.set"
        assert params == {"mask": [1.0, 0.0], "fit_uid": "abc"}

    def test_set_parameter_value(self, fc, mock_client):
        mock_client.set_response("parameter.set_value", {"ok": True})
        fc.set_parameter_value("tau1", 3.5, fit_uid="abc")
        method, params = _last_call(mock_client)
        assert method == "parameter.set_value"
        assert params == {"parameter_name": "tau1", "value": 3.5, "fit_uid": "abc"}

    def test_set_parameter_fixed(self, fc, mock_client):
        mock_client.set_response("parameter.set_fixed", {"ok": True})
        fc.set_parameter_fixed("tau1", True, fit_uid="abc")
        method, params = _last_call(mock_client)
        assert method == "parameter.set_fixed"
        assert params == {"parameter_name": "tau1", "fixed": True, "fit_uid": "abc"}

    def test_set_parameter_bounds(self, fc, mock_client):
        mock_client.set_response("parameter.set_bounds", {"ok": True})
        fc.set_parameter_bounds("tau1", (0.1, 10.0), fit_uid="abc")
        method, params = _last_call(mock_client)
        assert method == "parameter.set_bounds"
        assert params == {"parameter_name": "tau1", "bounds": [0.1, 10.0], "fit_uid": "abc"}

    def test_set_parameter_bounds_on(self, fc, mock_client):
        mock_client.set_response("parameter.set_bounds_on", {"ok": True})
        fc.set_parameter_bounds_on("tau1", True, fit_uid="abc")
        method, params = _last_call(mock_client)
        assert method == "parameter.set_bounds_on"
        assert params == {"parameter_name": "tau1", "bounds_on": True, "fit_uid": "abc"}

    def test_link_parameters(self, fc, mock_client):
        mock_client.set_response("parameter.link", {"ok": True})
        fc.link_parameters("tau1", "tau2", fit_uid="abc", target_fit_uid="def")
        method, params = _last_call(mock_client)
        assert method == "parameter.link"
        assert params == {
            "parameter_name": "tau1",
            "target_parameter_name": "tau2",
            "fit_uid": "abc",
            "target_fit_uid": "def",
        }

    def test_unlink_parameter(self, fc, mock_client):
        mock_client.set_response("parameter.unlink", {"ok": True})
        fc.unlink_parameter("tau1", fit_uid="abc")
        method, params = _last_call(mock_client)
        assert method == "parameter.unlink"
        assert params == {"parameter_name": "tau1", "fit_uid": "abc"}

    def test_model_finalize(self, fc, mock_client):
        mock_client.set_response("model.finalize", {"ok": True})
        fc.model_finalize(fit_uid="abc")
        method, params = _last_call(mock_client)
        assert method == "model.finalize"
        assert params == {"fit_uid": "abc"}

    def test_start_sampling(self, fc, mock_client):
        mock_client.set_response("fit.sample.start", {"ok": True, "job_id": "j1"})
        fc.start_sampling(fit_uid="abc", n_steps=1000, n_runs=2, target_directory="/tmp")
        method, params = _last_call(mock_client)
        assert method == "fit.sample.start"
        assert params == {
            "fit_uid": "abc",
            "n_steps": 1000,
            "n_runs": 2,
            "target_directory": "/tmp",
        }

    def test_start_parameter_scan(self, fc, mock_client):
        mock_client.set_response("fit.parameter_scan.start", {"ok": True, "job_id": "j1"})
        fc.start_parameter_scan("tau1", fit_uid="abc", n_steps=50, range_factor=2.0)
        method, params = _last_call(mock_client)
        assert method == "fit.parameter_scan.start"
        assert params == {
            "parameter_name": "tau1",
            "fit_uid": "abc",
            "n_steps": 50,
            "range_factor": 2.0,
        }

    def test_select_fit(self, fc, mock_client):
        mock_client.set_response("fit.select", {"ok": True, "fit": {"uid": "abc"}})
        fc.select_fit(fit_uid="abc")
        method, params = _last_call(mock_client)
        assert method == "fit.select"
        assert params == {"fit_uid": "abc"}

    def test_remove_fits(self, fc, mock_client):
        mock_client.set_response("fit.remove", {"ok": True})
        fc.remove_fits(fit_uids=["a", "b"])
        method, params = _last_call(mock_client)
        assert method == "fit.remove"
        assert params == {"fit_uids": ["a", "b"]}

    def test_reorder_fits(self, fc, mock_client):
        mock_client.set_response("fit.reorder", {"ok": True})
        fc.reorder_fits(["a", "b", "c"])
        method, params = _last_call(mock_client)
        assert method == "fit.reorder"
        assert params == {"fit_order": ["a", "b", "c"]}

    def test_group_select_member(self, fc, mock_client):
        mock_client.set_response("fit.group.select_member", {"ok": True})
        fc.group_select_member("group_uid", 0)
        method, params = _last_call(mock_client)
        assert method == "fit.group.select_member"
        assert params == {"fit_uid": "group_uid", "member_index": 0}

    def test_model_set_state(self, fc, mock_client):
        mock_client.set_response("model.state.set", {"ok": True})
        fc.model_set_state({"convolve": True}, fit_uid="abc")
        method, params = _last_call(mock_client)
        assert method == "model.state.set"
        assert params == {"state_data": {"convolve": True}, "fit_uid": "abc"}
