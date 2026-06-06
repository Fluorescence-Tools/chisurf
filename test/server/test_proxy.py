from __future__ import annotations

"""Tests for the clean typed proxy classes."""

from unittest.mock import MagicMock, patch
import pytest

from chisurf.core.api._client import ChisurfClient
from chisurf.core.api._proxies import (
    ProxyDatasetList, ProxyFitList,
    FitProxy, DatasetProxy, ParameterProxy, ModelProxy, DataProxy,
)


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def client():
    return MagicMock(spec=ChisurfClient)


# ── DataProxy tests ──────────────────────────────────────────────


class TestDataProxy:

    def test_properties(self):
        dp = DataProxy({"name": "MyData", "uid": "ds1", "filename": "/tmp/f.dat", "experiment": "TCSPC"})
        assert dp.name == "MyData"
        assert dp.uid == "ds1"
        assert dp.filename == "/tmp/f.dat"
        assert dp.experiment == "TCSPC"

    def test_empty_data(self):
        dp = DataProxy({})
        assert dp.name is None
        assert dp.uid is None
        assert dp.filename is None
        assert dp.experiment is None

    def test_repr(self):
        dp = DataProxy({"name": "MyData"})
        assert "MyData" in repr(dp)


# ── ParameterProxy tests ─────────────────────────────────────────


class TestParameterProxy:

    @pytest.fixture
    def param(self):
        return ParameterProxy({
            "name": "tau1", "fit_uid": "f1",
            "value": 3.5, "fixed": False, "bounds": (0.0, 10.0),
            "bounds_on": True, "is_linked": False, "linked_to": "",
            "error_estimate": 0.1,
        })

    def test_read_properties(self, param):
        assert param.name == "tau1"
        assert param.fit_uid == "f1"
        assert param.value == 3.5
        assert param.fixed is False
        assert param.bounds == (0.0, 10.0)
        assert param.bounds_on is True
        assert param.is_linked is False
        assert param.linked_to == ""
        assert param.error_estimate == 0.1

    def test_set_value_calls_server(self, client):
        p = ParameterProxy({"name": "tau1", "fit_uid": "f1"}, client=client)
        client.call.return_value = {"ok": True}
        p.set_value(4.0)
        client.call.assert_called_once_with("parameter.set_value", {
            "parameter_name": "tau1", "value": 4.0, "fit_uid": "f1",
        })

    def test_set_fixed_calls_server(self, client):
        p = ParameterProxy({"name": "tau1", "fit_uid": "f1"}, client=client)
        client.call.return_value = {"ok": True}
        p.set_fixed(True)
        client.call.assert_called_once_with("parameter.set_fixed", {
            "parameter_name": "tau1", "fixed": True, "fit_uid": "f1",
        })

    def test_set_bounds_calls_server(self, client):
        p = ParameterProxy({"name": "tau1", "fit_uid": "f1"}, client=client)
        client.call.return_value = {"ok": True}
        p.set_bounds((0.0, 5.0))
        client.call.assert_called_once_with("parameter.set_bounds", {
            "parameter_name": "tau1", "bounds": [0.0, 5.0], "fit_uid": "f1",
        })

    def test_set_bounds_on_calls_server(self, client):
        p = ParameterProxy({"name": "tau1", "fit_uid": "f1"}, client=client)
        client.call.return_value = {"ok": True}
        p.set_bounds_on(False)
        client.call.assert_called_once_with("parameter.set_bounds_on", {
            "parameter_name": "tau1", "bounds_on": False, "fit_uid": "f1",
        })

    def test_link_to_calls_server(self, client):
        p = ParameterProxy({"name": "tau1", "fit_uid": "f1"}, client=client)
        client.call.return_value = {"ok": True}
        p.link_to("tau2")
        client.call.assert_called_once_with("parameter.link", {
            "parameter_name": "tau1", "target_parameter_name": "tau2", "fit_uid": "f1",
        })

    def test_unlink_calls_server(self, client):
        p = ParameterProxy({"name": "tau1", "fit_uid": "f1"}, client=client)
        client.call.return_value = {"ok": True}
        p.unlink()
        client.call.assert_called_once_with("parameter.unlink", {
            "parameter_name": "tau1", "fit_uid": "f1",
        })

    def test_no_client_raises(self):
        p = ParameterProxy({"name": "tau1", "fit_uid": "f1"}, client=None)
        with pytest.raises(RuntimeError, match="No client"):
            p.set_value(1.0)
        with pytest.raises(RuntimeError, match="No client"):
            p.link_to("tau2")

    def test_repr(self, param):
        assert "tau1" in repr(param)


# ── ModelProxy tests ─────────────────────────────────────────────


class TestModelProxy:

    def test_properties(self):
        data = {
            "name": "Lifetime", "n_points": 100, "n_free": 3, "chi2r": 1.2,
            "parameters_all": [
                {"name": "tau1", "value": 3.5, "fit_uid": "f1"},
                {"name": "tau2", "value": 1.0, "fit_uid": "f1"},
            ],
        }
        m = ModelProxy(data)
        assert m.name == "Lifetime"
        assert m.n_points == 100
        assert m.n_free == 3
        assert m.chi2r == 1.2

    def test_parameters_all(self):
        data = {
            "parameters_all": [
                {"name": "tau1", "value": 3.5, "fit_uid": "f1"},
                {"name": "tau2", "value": 1.0, "fit_uid": "f1"},
            ],
        }
        m = ModelProxy(data)
        params = m.parameters_all
        assert len(params) == 2
        assert isinstance(params[0], ParameterProxy)
        assert params[0].name == "tau1"
        assert params[1].value == 1.0

    def test_parameters_all_dict(self):
        data = {
            "parameters_all": [
                {"name": "tau1", "value": 3.5, "fit_uid": "f1"},
            ],
        }
        m = ModelProxy(data)
        pdict = m.parameters_all_dict
        assert "tau1" in pdict
        assert isinstance(pdict["tau1"], ParameterProxy)
        assert pdict["tau1"].value == 3.5

    def test_empty_model(self):
        m = ModelProxy({})
        assert m.name is None
        assert m.parameters_all == []

    def test_repr(self):
        m = ModelProxy({"name": "MyModel"})
        assert "MyModel" in repr(m)


# ── FitProxy tests ───────────────────────────────────────────────


class TestFitProxy:

    @pytest.fixture
    def fit_data(self):
        return {
            "index": 0,
            "uid": "f1",
            "name": "TestFit",
            "type": "FitGroup",
            "chi2": 1.5,
            "chi2r": 1.2,
            "n_points": 100,
            "n_free": 3,
            "dataset_name": "TestData",
            "dataset_uid": "ds1",
            "model_name": "Lifetime",
            "data": {"name": "TestData", "uid": "ds1", "filename": "/tmp/f.dat", "experiment": "TCSPC"},
            "model": {
                "name": "Lifetime", "n_points": 100, "n_free": 3, "chi2r": 1.2,
                "parameters_all": [
                    {"name": "tau1", "value": 3.5, "fit_uid": "f1"},
                    {"name": "tau2", "value": 1.0, "fit_uid": "f1"},
                ],
            },
            "parameters": {"tau1": {"value": 3.5}},
        }

    def test_read_properties(self, fit_data):
        f = FitProxy(fit_data)
        assert f.uid == "f1"
        assert f.name == "TestFit"
        assert f.type == "FitGroup"
        assert f.index == 0
        assert f.chi2 == 1.5
        assert f.chi2r == 1.2
        assert f.n_points == 100
        assert f.n_free == 3
        assert f.dataset_name == "TestData"
        assert f.dataset_uid == "ds1"
        assert f.model_name == "Lifetime"

    def test_data_proxy(self, fit_data):
        f = FitProxy(fit_data)
        assert isinstance(f.data, DataProxy)
        assert f.data.name == "TestData"
        assert f.data.uid == "ds1"

    def test_no_data_when_missing(self):
        f = FitProxy({"uid": "f1", "name": "Test"})
        assert f.data is None

    def test_model_proxy(self, fit_data):
        f = FitProxy(fit_data)
        assert isinstance(f.model, ModelProxy)
        assert f.model.name == "Lifetime"
        assert len(f.model.parameters_all) == 2

    def test_no_model_when_missing(self):
        f = FitProxy({"uid": "f1", "name": "Test"})
        assert f.model is None

    def test_parameters_all_delegates_to_model(self, fit_data):
        f = FitProxy(fit_data)
        params = f.parameters_all
        assert len(params) == 2
        assert params[0].name == "tau1"

    def test_parameters_all_empty_no_model(self):
        f = FitProxy({"uid": "f1"})
        assert f.parameters_all == []

    def test_parameters_all_dict(self, fit_data):
        f = FitProxy(fit_data)
        pdict = f.parameters_all_dict
        assert "tau1" in pdict
        assert pdict["tau1"].value == 3.5

    def test_run_calls_server(self, client):
        f = FitProxy({"uid": "f1", "name": "Test"}, client=client)
        client.call.return_value = {"ok": True}
        f.run()
        client.call.assert_called_once_with("fit.run", {"fit_uid": "f1"})

    def test_save_calls_server(self, client):
        f = FitProxy({"uid": "f1"}, client=client)
        client.call.return_value = {"ok": True}
        f.save("/tmp/out", "csv")
        client.call.assert_called_once_with("fit.save", {
            "fit_uid": "f1", "filename": "/tmp/out", "file_type": "csv",
        })

    def test_save_with_kwargs(self, client):
        f = FitProxy({"uid": "f1"}, client=client)
        client.call.return_value = {"ok": True}
        f.save("/tmp/out", save_curves=True)
        client.call.assert_called_once_with("fit.save", {
            "fit_uid": "f1", "filename": "/tmp/out", "file_type": "csv", "save_curves": True,
        })

    def test_update_calls_server(self, client):
        f = FitProxy({"uid": "f1"}, client=client)
        client.call.return_value = {"ok": True}
        f.update()
        client.call.assert_called_once_with("fit.update", {"fit_uid": "f1"})

    def test_set_result_idx_calls_server(self, client):
        f = FitProxy({"uid": "f1"}, client=client)
        client.call.return_value = {"ok": True}
        f.set_result_idx(3)
        client.call.assert_called_once_with("fit.set_result_idx", {
            "fit_uid": "f1", "result_idx": 3,
        })

    def test_set_dataset_calls_server(self, client):
        f = FitProxy({"uid": "f1"}, client=client)
        client.call.return_value = {"ok": True}
        f.set_dataset(dataset_index=5)
        client.call.assert_called_once_with("fit.set_dataset", {
            "fit_uid": "f1", "dataset_index": 5, "dataset_uid": None,
        })

    def test_model_finalize_calls_server(self, client):
        f = FitProxy({"uid": "f1"}, client=client)
        client.call.return_value = {"ok": True}
        f.model_finalize()
        client.call.assert_called_once_with("model.finalize", {"fit_uid": "f1"})

    def test_model_set_parse_function(self, client):
        f = FitProxy({"uid": "f1"}, client=client)
        client.call.return_value = {"ok": True}
        f.model_set_parse_function("expr")
        client.call.assert_called_once_with("model.set_parse_function", {
            "fit_uid": "f1", "function_name": "expr",
        })

    def test_no_client_raises(self):
        f = FitProxy({"uid": "f1"}, client=None)
        with pytest.raises(RuntimeError, match="No client"):
            f.run()
        with pytest.raises(RuntimeError, match="No client"):
            f.save("/tmp/x")

    def test_repr(self, fit_data):
        f = FitProxy(fit_data)
        assert "TestFit" in repr(f)


# ── DatasetProxy tests ────────────────────────────────────────────


class TestDatasetProxy:

    def test_properties(self):
        ds = DatasetProxy({
            "index": 0, "uid": "ds1", "name": "TestData",
            "type": "DataCurve", "filename": "/tmp/f.dat",
            "experiment": "TCSPC", "length": 100,
        })
        assert ds.uid == "ds1"
        assert ds.name == "TestData"
        assert ds.type == "DataCurve"
        assert ds.index == 0
        assert ds.filename == "/tmp/f.dat"
        assert ds.experiment == "TCSPC"
        assert ds.length == 100

    def test_curve_data_calls_server(self, client):
        client.dataset__curve_data.return_value = {
            "ok": True, "x": [0.0, 1.0], "y": [2.0, 3.0],
        }
        ds = DatasetProxy({"uid": "ds1", "name": "Data"}, client=client)
        curve = ds.curve_data()
        assert curve["x"] == [0.0, 1.0]
        assert curve["y"] == [2.0, 3.0]
        client.dataset__curve_data.assert_called_once_with(dataset_uid="ds1")

    def test_curve_data_caches(self, client):
        client.dataset__curve_data.return_value = {"ok": True, "x": [0.0], "y": [1.0]}
        ds = DatasetProxy({"uid": "ds1"}, client=client)
        ds.curve_data()
        ds.curve_data()
        client.dataset__curve_data.assert_called_once()

    def test_no_client_curve_data_returns_empty(self):
        ds = DatasetProxy({"uid": "ds1"})
        assert ds.curve_data() == {}

    def test_repr(self):
        ds = DatasetProxy({"name": "MyData"})
        assert "MyData" in repr(ds)


# ── ProxyList base tests ──────────────────────────────────────────


class TestProxyFitList:

    def test_fetch_empty(self, client):
        client.fit__list.return_value = []
        pl = ProxyFitList(client)
        assert len(pl) == 0

    def test_fetch_one(self, client):
        client.fit__list.return_value = [{"uid": "f1", "name": "F1", "type": "FitGroup"}]
        pl = ProxyFitList(client)
        assert len(pl) == 1
        fit = pl[0]
        assert isinstance(fit, FitProxy)
        assert fit.name == "F1"

    def test_iter(self, client):
        client.fit__list.return_value = [
            {"uid": "f1", "name": "F1", "type": "FitGroup"},
            {"uid": "f2", "name": "F2", "type": "FitGroup"},
        ]
        pl = ProxyFitList(client)
        names = [f.name for f in pl]
        assert names == ["F1", "F2"]

    def test_pop(self, client):
        client.fit__list.return_value = [
            {"uid": "f1", "name": "F1", "type": "FitGroup"},
            {"uid": "f2", "name": "F2", "type": "FitGroup"},
        ]
        client.fit__remove.return_value = {"ok": True}
        pl = ProxyFitList(client)
        popped = pl.pop(0)
        assert popped.name == "F1"
        client.fit__remove.assert_called_once_with(fit_indices=[], fit_uids=["f1"])

    def test_clear(self, client):
        client.fit__list.return_value = [{"uid": "f1", "name": "F1", "type": "FitGroup"}]
        client.fit__clear.return_value = {"ok": True}
        pl = ProxyFitList(client)
        pl.clear()
        client.fit__clear.assert_called_once()

    def test_index(self, client):
        client.fit__list.return_value = [
            {"uid": "f1", "name": "F1", "type": "FitGroup"},
            {"uid": "f2", "name": "F2", "type": "FitGroup"},
        ]
        pl = ProxyFitList(client)
        target = FitProxy({"uid": "f2"})
        assert pl.index(target) == 1

    def test_contains_by_uid(self, client):
        client.fit__list.return_value = [
            {"uid": "f1", "name": "F1", "type": "FitGroup"},
        ]
        pl = ProxyFitList(client)
        target = FitProxy({"uid": "f1"})
        assert target in pl
        missing = FitProxy({"uid": "f99"})
        assert missing not in pl

    def test_fetch_always_calls_server(self, client):
        client.fit__list.return_value = [{"uid": "f1", "name": "F1", "type": "FitGroup"}]
        pl = ProxyFitList(client)
        assert len(pl) == 1
        client.fit__list.return_value = []
        # _fetch always calls server, so cache is never stale
        assert len(pl) == 0


class TestProxyDatasetList:

    def test_fetch_empty(self, client):
        client.dataset__list.return_value = []
        pl = ProxyDatasetList(client)
        assert len(pl) == 0

    def test_fetch_one(self, client):
        client.dataset__list.return_value = [{"uid": "ds1", "name": "DS1"}]
        pl = ProxyDatasetList(client)
        assert len(pl) == 1
        ds = pl[0]
        assert isinstance(ds, DatasetProxy)
        assert ds.name == "DS1"

    def test_remove(self, client):
        client.dataset__list.return_value = [{"uid": "ds1", "name": "DS1"}]
        client.dataset__remove.return_value = {"ok": True}
        pl = ProxyDatasetList(client)
        target = DatasetProxy({"uid": "ds1"})
        pl.remove(target)
        client.dataset__remove.assert_called_once_with(dataset_indices=[], dataset_uids=["ds1"])

    def test_clear(self, client):
        client.dataset__list.return_value = [{"uid": "ds1", "name": "DS1"}]
        client.dataset__clear.return_value = {"ok": True}
        pl = ProxyDatasetList(client)
        pl.clear()
        client.dataset__clear.assert_called_once()


# ── FitProxy RPC method completeness ─────────────────────────────


class TestFitProxyRpcMethods:

    def test_all_rpc_methods_covered(self):
        """FitProxy must have explicit methods for all documented RPC actions."""
        fit = FitProxy({"uid": "f1"}, client=MagicMock(spec=ChisurfClient))
        assert hasattr(fit, "run")
        assert hasattr(fit, "save")
        assert hasattr(fit, "update")
        assert hasattr(fit, "set_result_idx")
        assert hasattr(fit, "set_dataset")
        assert hasattr(fit, "model_finalize")
        assert hasattr(fit, "model_set_parse_function")


class TestDatasetProxyCurveData:

    def test_curve_data_via_explicit_method(self):
        """DatasetProxy.curve_data() is the only way to fetch arrays."""
        client = MagicMock(spec=ChisurfClient)
        client.dataset__curve_data.return_value = {
            "ok": True, "x": [0.0, 1.0], "y": [2.0, 3.0],
        }
        ds = DatasetProxy({"uid": "ds1", "name": "Data"}, client=client)
        curve = ds.curve_data()
        assert curve["ok"] is True
        assert curve["x"] == [0.0, 1.0]
        assert curve["y"] == [2.0, 3.0]
        client.dataset__curve_data.assert_called_once_with(dataset_uid="ds1")
