from __future__ import annotations

"""Tests for chisurf.server.services.datasets.

All tests use a SessionState instead of patching chisurf globals.
"""

from unittest.mock import MagicMock

from chisurf.server.services.datasets import (
    add_dataset,
    remove_datasets,
    list_datasets,
    get_dataset_info,
)
from chisurf.server.session import SessionState


class DummyDataset:
    """Minimal dataset-like object that is not iterable."""
    def __init__(self, name="test", uid="dummy-uid"):
        self.name = name
        self.unique_identifier = uid
        self.experiment = MagicMock()
        self.experiment.name = "TCSPC"


class DummyReader:
    """Minimal experiment reader that returns a DummyDataset."""
    name = "DummyReader"
    filename = "/path/to/test.ptu"

    def read(self, name=None, **kw):
        return DummyDataset(name=name or "test", uid="dummy-uid")


class TestDatasetsService:

    def test_list_datasets_empty(self):
        state = SessionState()
        result = list_datasets(state)
        assert result["datasets"] == []

    def test_list_datasets(self):
        ds1 = DummyDataset(name="Data1", uid="uid-1")
        ds1.experiment.name = "FCS"

        ds2 = DummyDataset(name="Data2", uid="uid-2")

        state = SessionState(datasets=[ds1, ds2])
        result = list_datasets(state)
        assert len(result["datasets"]) == 2
        assert result["datasets"][0]["name"] == "Data1"
        assert result["datasets"][1]["name"] == "Data2"

    def test_get_dataset_info_found(self):
        ds = DummyDataset(name="MyData", uid="uid-42")
        ds.experiment.name = "FCS"

        state = SessionState(datasets=[ds])
        result = get_dataset_info(state, dataset_index=0)
        assert result["dataset"]["name"] == "MyData"
        assert result["dataset"]["experiment"] == "FCS"

    def test_get_dataset_info_not_found(self):
        state = SessionState()
        result = get_dataset_info(state, dataset_index=0)
        assert not result["ok"]
        assert "dataset not found" in result["error"]
        assert result["error_code"] == "NOT_FOUND"

    def test_add_dataset_calls_read(self):
        reader = DummyReader()
        state = SessionState()
        result = add_dataset(state, reader=reader)
        assert result["ok"]
        assert "uid" in result
        assert result["name"] == "DummyReader"
        assert len(state.datasets) == 1

    def test_add_dataset_uses_custom_name(self):
        reader = DummyReader()
        state = SessionState()
        result = add_dataset(state, reader=reader, name="custom-name")
        assert result["name"] == "custom-name"

    def test_remove_datasets_removes_by_index(self):
        ds = DummyDataset(name="Keep", uid="keep-uid")
        target = DummyDataset(name="Remove", uid="remove-uid")

        state = SessionState(datasets=[ds, target])
        result = remove_datasets(state, dataset_indices=[1])
        assert result["ok"]
        assert result["removed_count"] == 1
        assert len(state.datasets) == 1
        assert state.datasets[0].name == "Keep"

    def test_remove_datasets_protects_global_fit(self):
        global_fit = DummyDataset(name="Global-Fit", uid="gf-uid")

        state = SessionState(datasets=[global_fit])
        result = remove_datasets(state, dataset_indices=[0])
        assert result["removed_count"] == 0
        assert len(state.datasets) == 1

    def test_remove_datasets_checks_fit_dependency(self):
        ds = DummyDataset(name="DS", uid="ds-uid")

        fit = MagicMock()
        fit.data = ds

        state = SessionState(datasets=[ds], fits=[fit])
        result = remove_datasets(state, dataset_indices=[0])
        assert not result["ok"]
        assert "fit" in result.get("error", "")
