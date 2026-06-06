from __future__ import annotations

import pytest

from chisurf.core.api import ChiSurfAPI
from chisurf.core.api.adapters import FitListAdapter, DatasetListAdapter


class DummyClient:
    def dataset__list(self):
        return [
            {"index": 0, "uid": "ds-1", "name": "d1", "type": "DataCurve", "experiment": "TCSPC"},
            {"index": 1, "uid": "ds-2", "name": "d2", "type": "DataCurve", "experiment": "FCS"},
        ]

    def fit__list(self):
        return [
            {"index": 0, "uid": "f-1", "name": "Fit1", "type": "FitGroup", "chi2": 1.2,
             "dataset_uid": "ds-1", "dataset_name": "d1", "model_name": "LifetimeModel", "parameter_count": 4},
            {"index": 1, "uid": "f-2", "name": "Fit2", "type": "FitGroup", "chi2": 0.9,
             "dataset_uid": "ds-2", "dataset_name": "d2", "model_name": "AnisotropyModel", "parameter_count": 6},
        ]

    def call(self, method, params=None):
        return {"ok": True}


class TestFitListAdapter:
    def test_len(self):
        api = ChiSurfAPI(client=DummyClient(), mode="server")
        adapter = FitListAdapter(api)
        assert len(adapter) == 2

    def test_getitem(self):
        api = ChiSurfAPI(client=DummyClient(), mode="server")
        adapter = FitListAdapter(api)
        f = adapter[0]
        assert f["name"] == "Fit1"
        assert f["uid"] == "f-1"
        assert f["chi2"] == 1.2

    def test_getitem_slice(self):
        api = ChiSurfAPI(client=DummyClient(), mode="server")
        adapter = FitListAdapter(api)
        fits = adapter[0:1]
        assert len(fits) == 1
        assert fits[0]["uid"] == "f-1"

    def test_iteration(self):
        api = ChiSurfAPI(client=DummyClient(), mode="server")
        adapter = FitListAdapter(api)
        names = [f["name"] for f in adapter]
        assert names == ["Fit1", "Fit2"]

    def test_contains_by_uid(self):
        api = ChiSurfAPI(client=DummyClient(), mode="server")
        adapter = FitListAdapter(api)
        assert "f-1" in adapter
        assert "nonexistent" not in adapter

    def test_get_by_uid(self):
        api = ChiSurfAPI(client=DummyClient(), mode="server")
        adapter = FitListAdapter(api)
        f = adapter.get_by_uid("f-2")
        assert f is not None
        assert f["name"] == "Fit2"
        assert adapter.get_by_uid("nope") is None

    def test_fit_count(self):
        api = ChiSurfAPI(client=DummyClient(), mode="server")
        adapter = FitListAdapter(api)
        assert adapter.fit_count() == 2

    def test_refresh(self):
        api = ChiSurfAPI(client=DummyClient(), mode="server")
        adapter = FitListAdapter(api)
        result = adapter.refresh()
        assert len(result) == 2


class TestDatasetListAdapter:
    def test_len(self):
        api = ChiSurfAPI(client=DummyClient(), mode="server")
        adapter = DatasetListAdapter(api)
        assert len(adapter) == 2

    def test_getitem(self):
        api = ChiSurfAPI(client=DummyClient(), mode="server")
        adapter = DatasetListAdapter(api)
        d = adapter[0]
        assert d["name"] == "d1"
        assert d["uid"] == "ds-1"

    def test_iteration(self):
        api = ChiSurfAPI(client=DummyClient(), mode="server")
        adapter = DatasetListAdapter(api)
        names = [d["name"] for d in adapter]
        assert names == ["d1", "d2"]

    def test_contains_by_uid(self):
        api = ChiSurfAPI(client=DummyClient(), mode="server")
        adapter = DatasetListAdapter(api)
        assert "ds-1" in adapter
        assert "nope" not in adapter

    def test_get_by_uid(self):
        api = ChiSurfAPI(client=DummyClient(), mode="server")
        adapter = DatasetListAdapter(api)
        d = adapter.get_by_uid("ds-2")
        assert d is not None
        assert d["name"] == "d2"

    def test_dataset_count(self):
        api = ChiSurfAPI(client=DummyClient(), mode="server")
        adapter = DatasetListAdapter(api)
        assert adapter.dataset_count() == 2