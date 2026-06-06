from __future__ import annotations

"""Tests for chisurf.server.session.SessionState."""

import pytest
from chisurf.server.session import SessionState


class TestSessionState:
    """SessionState is a mutable container for runtime state."""

    def test_create_empty(self):
        state = SessionState()
        assert state.datasets == []
        assert state.fits == []
        assert state.experiments == {}
        assert state.current_experiment is None
        assert state.current_setup is None
        assert state.current_fit_uid is None
        assert state.registry is not None

    def test_create_with_initial_values(self):
        state = SessionState(
            datasets=[1, 2],
            fits=[3, 4],
            experiments={"TCSPC": "exp1"},
            current_experiment="TCSPC",
            current_fit_uid="fit-1",
        )
        assert state.datasets == [1, 2]
        assert state.fits == [3, 4]
        assert state.experiments == {"TCSPC": "exp1"}
        assert state.current_experiment == "TCSPC"
        assert state.current_fit_uid == "fit-1"

    def test_add_dataset(self):
        state = SessionState()
        state.add_dataset("ds1")
        assert state.datasets == ["ds1"]

    def test_add_fit(self):
        state = SessionState()
        state.add_fit("fit1")
        assert state.fits == ["fit1"]

    def test_remove_dataset_by_index(self):
        state = SessionState(datasets=["a", "b", "c"])
        state.remove_dataset(1)
        assert state.datasets == ["a", "c"]

    def test_remove_dataset_by_uid(self):
        state = SessionState(datasets=[
            {"uid": "u1", "name": "A"},
            {"uid": "u2", "name": "B"},
        ])
        state.remove_dataset(uid="u1")
        assert len(state.datasets) == 1
        assert state.datasets[0]["uid"] == "u2"

    def test_remove_fit_by_index(self):
        state = SessionState(fits=["a", "b", "c"])
        state.remove_fit(1)
        assert state.fits == ["a", "c"]

    def test_remove_fit_by_uid(self):
        state = SessionState(fits=[
            {"uid": "u1", "name": "A"},
            {"uid": "u2", "name": "B"},
        ])
        state.remove_fit(uid="u1")
        assert len(state.fits) == 1
        assert state.fits[0]["uid"] == "u2"

    def test_clear(self):
        state = SessionState(
            datasets=[1],
            fits=[2],
            experiments={"X": "y"},
            current_experiment="X",
        )
        state.clear()
        assert state.datasets == []
        assert state.fits == []
        assert state.experiments == {}
        assert state.current_experiment is None
        assert state.current_fit_uid is None

    def test_to_dict(self):
        state = SessionState(
            datasets=["a"],
            fits=["b"],
            experiments={"TCSPC": "obj"},
            current_experiment="TCSPC",
        )
        d = state.to_dict()
        assert d["current_experiment"] == "TCSPC"
        assert d["dataset_count"] == 1
        assert d["fit_count"] == 1
        assert d["experiment_names"] == ["TCSPC"]

    def test_find_fit_by_uid(self):
        class Obj:
            def __init__(self, uid):
                self.unique_identifier = uid

        state = SessionState(fits=[Obj("u1"), Obj("u2"), Obj("u3")])
        found = state.find_fit_by_uid("u2")
        assert found is not None
        assert found.unique_identifier == "u2"

    def test_find_fit_by_uid_not_found(self):
        state = SessionState()
        assert state.find_fit_by_uid("nonexistent") is None

    def test_find_dataset_by_uid(self):
        class Obj:
            def __init__(self, uid):
                self.unique_identifier = uid

        state = SessionState(datasets=[Obj("d1"), Obj("d2")])
        found = state.find_dataset_by_uid("d1")
        assert found is not None
        assert found.unique_identifier == "d1"

    def test_find_dataset_by_uid_not_found(self):
        state = SessionState()
        assert state.find_dataset_by_uid("nonexistent") is None
