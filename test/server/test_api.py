from __future__ import annotations

import pytest

from chisurf.core.api import ChiSurfAPI
from chisurf.core.api.context import PluginContext


class DummyDataset:
    def __init__(self, name="test", uid="dummy-uid"):
        self.name = name
        self.unique_identifier = uid
        self.experiment = type("Exp", (), {"name": "TCSPC"})()


class DummyParameter:
    def __init__(self, name="tau", value=1.0):
        self.name = name
        self.value = value
        self.fixed = False
        self.bounds = (0.0, 10.0)
        self.bounds_on = True
        self.link = None
        self.error_estimate = 0.1


class DummyModel:
    def __init__(self):
        self.parameters_all_dict = {"tau": DummyParameter("tau", 3.8)}
        self.name = "TestModel"

    def update_model(self):
        pass

    def finalize(self):
        pass


class DummyFit:
    def __init__(self, name="Fit1", uid="fit-uid-1"):
        self.name = name
        self.unique_identifier = uid
        self.chi2 = 1.23
        self.data = DummyDataset(name="Data1", uid="data-uid-1")
        self.model = DummyModel()

    def run(self):
        self.chi2 = 0.95


class DummyClient:
    def __init__(self):
        self._calls = []

    def meta__ping(self):
        self._calls.append("meta__ping")
        return {"ok": True, "status": "alive"}

    def dataset__list(self):
        self._calls.append("dataset__list")
        return [
            {"index": 0, "uid": "ds-1", "name": "d1", "type": "DataCurve", "experiment": "TCSPC"},
        ]

    def dataset__get(self, dataset_index=None, dataset_uid=None):
        self._calls.append("dataset__get")
        return {"index": 0, "uid": "ds-1", "name": "d1", "type": "DataCurve", "experiment": "TCSPC"}

    def dataset__remove(self, dataset_indices=None, dataset_uids=None):
        self._calls.append("dataset__remove")
        return {"ok": True}

    def dataset__clear(self):
        self._calls.append("dataset__clear")
        return {"ok": True}

    def fit__list(self):
        self._calls.append("fit__list")
        return []

    def fit__get(self, fit_index=None, fit_uid=None):
        self._calls.append("fit__get")
        return {"uid": "f1", "name": "Fit1", "index": 0}

    def fit__run(self, fit_index=None, fit_uid=None):
        self._calls.append("fit__run")
        return {"ok": True}

    def fit__remove(self, fit_indices=None, fit_uids=None):
        self._calls.append("fit__remove")
        return {"ok": True}

    def fit__clear(self):
        self._calls.append("fit__clear")
        return {"ok": True}

    def parameter__get(self, parameter_name, fit_index=0, fit_uid=None):
        self._calls.append("parameter__get")
        return {"name": parameter_name, "value": 3.8}

    def parameter__set_value(self, parameter_name, value, fit_index=0, fit_uid=None):
        self._calls.append("parameter__set_value")
        return {"ok": True}

    def parameter__set_fixed(self, parameter_name, fixed, fit_index=0, fit_uid=None):
        self._calls.append("parameter__set_fixed")
        return {"ok": True}

    def parameter__set_bounds(self, parameter_name, lower, upper, fit_index=0, fit_uid=None):
        self._calls.append("parameter__set_bounds")
        return {"ok": True}

    def project__info(self):
        self._calls.append("project__info")
        return {"ok": True, "fit_count": 0, "dataset_count": 0}

    def project__save(self, target_path, project_name=None):
        self._calls.append("project__save")
        return {"ok": True}

    def project__load(self, project_path):
        self._calls.append("project__load")
        return {"ok": True}

    def call(self, method, params=None):
        self._calls.append(f"call:{method}")
        return {"ok": True}


class TestChiSurfAPI:

    def test_create_local_mode(self):
        api = ChiSurfAPI(mode="local")
        assert api.mode == "local"
        assert api.client is None

    def test_create_server_mode(self):
        client = DummyClient()
        api = ChiSurfAPI(client=client, mode="server")
        assert api.mode == "server"
        assert api.client is client

    def test_create_hybrid_mode(self):
        api = ChiSurfAPI(mode="hybrid")
        assert api.mode == "hybrid"

    def test_ping_no_client(self):
        api = ChiSurfAPI(mode="local")
        result = api.ping()
        assert result["ok"] is True

    def test_ping_with_client(self):
        client = DummyClient()
        api = ChiSurfAPI(client=client, mode="server")
        result = api.ping()
        assert result["ok"] is True
        assert "meta__ping" in client._calls

    def test_list_datasets_local(self):
        dataset = DummyDataset(name="test", uid="uid-1")
        import chisurf
        chisurf.imported_datasets.append(dataset)
        try:
            api = ChiSurfAPI(mode="local")
            result = api.list_datasets()
            assert isinstance(result, list)
            assert len(result) >= 1
        finally:
            chisurf.imported_datasets.clear()

    def test_list_datasets_server(self):
        client = DummyClient()
        api = ChiSurfAPI(client=client, mode="server")
        result = api.list_datasets()
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["name"] == "d1"

    def test_list_fits_local(self):
        fit = DummyFit(name="TestFit", uid="fit-uid")
        import chisurf
        chisurf.fits.append(fit)
        try:
            api = ChiSurfAPI(mode="local")
            result = api.list_fits()
            assert isinstance(result, list)
            assert len(result) >= 1
            assert result[0]["name"] == "TestFit"
            assert result[0]["chi2"] == 1.23
        finally:
            chisurf.fits.clear()

    def test_list_fits_server(self):
        client = DummyClient()
        api = ChiSurfAPI(client=client, mode="server")
        result = api.list_fits()
        assert isinstance(result, list)

    def test_run_fit_local(self):
        fit = DummyFit(name="MyFit", uid="run-uid")
        import chisurf
        chisurf.fits.append(fit)
        try:
            api = ChiSurfAPI(mode="local")
            result = api.run_fit(fit_index=0)
            assert result["ok"] is True
            assert result["fit_uid"] == "run-uid"
        finally:
            chisurf.fits.clear()

    def test_run_fit_server(self):
        client = DummyClient()
        api = ChiSurfAPI(client=client, mode="server")
        result = api.run_fit(fit_index=0)
        assert result["ok"] is True
        assert "fit__run" in client._calls

    def test_get_parameter_local(self):
        fit = DummyFit(name="Fit", uid="p-uid")
        import chisurf
        chisurf.fits.append(fit)
        try:
            api = ChiSurfAPI(mode="local")
            result = api.get_parameter("tau", fit_index=0)
            assert result["ok"] is True
            assert result["parameter"]["value"] == 3.8
        finally:
            chisurf.fits.clear()

    def test_set_parameter_value_local(self):
        fit = DummyFit(name="Fit", uid="set-uid")
        import chisurf
        chisurf.fits.append(fit)
        try:
            api = ChiSurfAPI(mode="local")
            result = api.set_parameter_value("tau", 5.0, fit_index=0)
            assert result["ok"] is True
        finally:
            chisurf.fits.clear()

    def test_get_project_info_local(self):
        api = ChiSurfAPI(mode="local")
        result = api.get_project_info()
        assert result["ok"] is True
        assert "fit_count" in result
        assert "dataset_count" in result

    def test_clear_datasets_server(self):
        client = DummyClient()
        api = ChiSurfAPI(client=client, mode="server")
        result = api.clear_datasets()
        assert result["ok"] is True
        assert "dataset__clear" in client._calls

    def test_clear_fits_local(self):
        fit = DummyFit()
        import chisurf
        chisurf.fits.append(fit)
        try:
            api = ChiSurfAPI(mode="local")
            result = api.clear_fits()
            assert result["ok"] is True
            assert len(chisurf.fits) == 0
        finally:
            chisurf.fits.clear()


class TestPluginContext:

    def test_creation(self):
        client = DummyClient()
        api = ChiSurfAPI(client=client, mode="server")
        ctx = PluginContext(api=api, client=client)
        assert ctx.api is api
        assert ctx.client is client

    def test_list_fits(self):
        client = DummyClient()
        api = ChiSurfAPI(client=client, mode="server")
        ctx = PluginContext(api=api, client=client)
        result = ctx.list_fits()
        assert isinstance(result, list)

    def test_ping(self):
        client = DummyClient()
        api = ChiSurfAPI(client=client, mode="server")
        ctx = PluginContext(api=api, client=client)
        result = ctx.ping()
        assert result["ok"] is True

    def test_no_api(self):
        ctx = PluginContext()
        result = ctx.list_fits()
        assert result == []

    def test_with_main_window(self):
        ctx = PluginContext(api=None, main_window="fake_window")
        assert ctx.main_window == "fake_window"