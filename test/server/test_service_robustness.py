from __future__ import annotations

"""Tests for service input robustness — edge cases for all service functions.

Covers: None values, missing keys, empty strings, type mismatches,
zero-length lists, boundary indices, and extreme numeric values.
"""

from unittest.mock import MagicMock

import pytest

from chisurf.server.services.datasets import (
    add_dataset,
    remove_datasets,
    clear_datasets,
    list_datasets,
    get_dataset_info,
    get_dataset_curve_data,
    dataset_rename,
    dataset_group,
    dataset_ungroup,
)
from chisurf.server.services.fits import (
    list_fits,
    get_fit_info,
    run_fit,
    fit_set_dataset,
    fit_set_result_idx,
    fit_set_fit_range,
    remove_fits,
    clear_fits,
    fit_save,
    fit_curve_data,
)
from chisurf.server.services.parameters import (
    get_parameter,
    set_parameter_value,
    set_parameter_fixed,
    set_parameter_bounds,
    set_parameter_bounds_on,
    parameter_link,
    parameter_unlink,
)
from chisurf.server.services.models import model_finalize, model_set_parse_function
from chisurf.server.services.projects import get_project_info, save_project, load_project
from chisurf.server.services.graph import build_fit_graph
from chisurf.server.session import SessionState


def _make_fit(**extra):
    fit = MagicMock()
    fit.unique_identifier = "fit-1"
    fit.name = "TestFit"
    fit.chi2 = 1.0
    fit.data = MagicMock()
    fit.data.name = "Data"
    fit.data.unique_identifier = "ds-1"
    fit.data.filename = "/tmp/test.dat"
    fit.model = MagicMock()
    fit.model.name = "Lifetime"
    fit.model.parameters_all_dict = {"tau": MagicMock()}
    fit.model.parameters_all = []
    fit.model.chi2r = 1.0
    for k, v in extra.items():
        setattr(fit, k, v)
    return fit


def _make_dataset(**extra):
    ds = MagicMock()
    ds.unique_identifier = "ds-1"
    ds.name = "TestData"
    ds.experiment = MagicMock()
    ds.experiment.name = "TCSPC"
    ds.filename = "/tmp/data.dat"
    ds.x = [0.0, 1.0, 2.0]
    ds.y = [3.0, 4.0, 5.0]
    for k, v in extra.items():
        setattr(ds, k, v)
    return ds


class TestInputRobustnessDatasets:

    def test_list_datasets_none_state(self):
        with pytest.raises(AttributeError):
            list_datasets(None)

    def test_add_dataset_no_reader_no_name(self):
        state = SessionState()
        result = add_dataset(state, reader=None, reader_name=None, filename=None)
        assert not result.get("ok")
        assert "no reader" in result.get("error", "")

    def test_add_dataset_with_curve_data_none_creates_empty(self):
        state = SessionState()
        result = add_dataset(state, reader_name="Test", filename="/tmp/f.dat", name="T", curve_data=None)
        assert result.get("ok") is True
        assert len(state.datasets) == 1

    def test_get_dataset_info_negative_index(self):
        ds = _make_dataset()
        state = SessionState(datasets=[ds])
        result = get_dataset_info(state, dataset_index=-1)
        assert not result.get("ok")

    def test_get_dataset_info_none_index(self):
        ds = _make_dataset()
        state = SessionState(datasets=[ds])
        result = get_dataset_info(state, dataset_index=None, dataset_uid=None)
        assert not result.get("ok")

    def test_get_dataset_curve_data_nonexistent(self):
        state = SessionState()
        result = get_dataset_curve_data(state, dataset_index=0)
        assert not result.get("ok")

    def test_dataset_rename_empty_name(self):
        ds = _make_dataset()
        state = SessionState(datasets=[ds])
        result = dataset_rename(state, new_name="", dataset_index=0)
        assert result.get("ok")
        assert ds.name == ""

    def test_remove_datasets_no_indices(self):
        ds = _make_dataset()
        state = SessionState(datasets=[ds])
        result = remove_datasets(state, dataset_indices=None, dataset_uids=None)
        assert not result.get("ok")
        assert "no datasets" in result.get("error", "")

    def test_remove_datasets_empty_list(self):
        ds = _make_dataset()
        state = SessionState(datasets=[ds])
        result = remove_datasets(state, dataset_indices=[], dataset_uids=[])
        assert not result.get("ok")

    def test_clear_datasets_empty(self):
        state = SessionState()
        result = clear_datasets(state)
        assert result.get("ok")
        assert result.get("cleared_count") == 0

    def test_dataset_group_out_of_range(self):
        state = SessionState()
        result = dataset_group(state, dataset_indices=[0])
        assert not result.get("ok")

    def test_dataset_ungroup_out_of_range(self):
        state = SessionState()
        result = dataset_ungroup(state, dataset_indices=[0])
        assert not result.get("ok")


class TestInputRobustnessFits:

    def test_list_fits_none_state(self):
        with pytest.raises(AttributeError):
            list_fits(None)

    def test_get_fit_info_negative_index(self):
        fit = _make_fit()
        state = SessionState(fits=[fit])
        result = get_fit_info(state, fit_index=-1)
        assert not result.get("ok")

    def test_get_fit_info_none_index(self):
        state = SessionState()
        result = get_fit_info(state, fit_index=None, fit_uid=None)
        assert not result.get("ok")

    def test_run_fit_negative_index(self):
        state = SessionState()
        result = run_fit(state, fit_index=-1)
        assert not result.get("ok")

    def test_remove_fits_no_indices(self):
        fit = _make_fit()
        state = SessionState(fits=[fit])
        result = remove_fits(state, fit_indices=None, fit_uids=None)
        assert not result.get("ok")
        assert "no fits" in result.get("error", "")

    def test_remove_fits_empty_lists(self):
        fit = _make_fit()
        state = SessionState(fits=[fit])
        result = remove_fits(state, fit_indices=[], fit_uids=[])
        assert not result.get("ok")

    def test_clear_fits_empty(self):
        state = SessionState()
        result = clear_fits(state)
        assert result.get("ok")
        assert result.get("cleared_count") == 0

    def test_fit_set_dataset_nonexistent_fit(self):
        state = SessionState()
        result = fit_set_dataset(state, fit_index=0, dataset_index=0)
        assert not result.get("ok")

    def test_fit_set_result_idx_negative(self):
        state = SessionState()
        result = fit_set_result_idx(state, fit_index=-1, result_idx=1)
        assert not result.get("ok")

    def test_fit_set_fit_range_swapped(self):
        fit = _make_fit()
        state = SessionState(fits=[fit])
        result = fit_set_fit_range(state, fit_index=0, xmin=100, xmax=10)
        assert result.get("ok")  # fit itself may accept swapped ranges

    def test_fit_save_no_filename_saves_to_empty(self):
        fit = _make_fit()
        state = SessionState(fits=[fit])
        result = fit_save(state, filename="", fit_index=0)
        # Service accepts empty filename and returns ok with empty saved_to path
        assert result.get("ok") is True
        assert "saved_to" in result

    def test_fit_curve_data_nonexistent(self):
        state = SessionState()
        result = fit_curve_data(state, fit_index=0)
        assert not result.get("ok")


class TestInputRobustnessParameters:

    def test_get_parameter_empty_name(self):
        fit = _make_fit()
        state = SessionState(fits=[fit])
        result = get_parameter(state, parameter_name="", fit_index=0)
        assert not result.get("ok")

    def test_get_parameter_nonexistent_fit(self):
        state = SessionState()
        result = get_parameter(state, parameter_name="tau", fit_index=0)
        assert not result.get("ok")

    def test_set_parameter_value_infinity(self):
        p = MagicMock()
        model = MagicMock()
        model.parameters_all_dict = {"tau": p}
        fit = _make_fit(model=model)
        state = SessionState(fits=[fit])
        result = set_parameter_value(state, parameter_name="tau", value=float('inf'), fit_index=0)
        assert result.get("ok")
        assert p.value == float('inf')

    def test_set_parameter_value_nan(self):
        p = MagicMock()
        model = MagicMock()
        model.parameters_all_dict = {"tau": p}
        fit = _make_fit(model=model)
        state = SessionState(fits=[fit])
        result = set_parameter_value(state, parameter_name="tau", value=float('nan'), fit_index=0)
        assert result.get("ok")

    def test_set_parameter_fixed_string(self):
        p = MagicMock()
        model = MagicMock()
        model.parameters_all_dict = {"tau": p}
        fit = _make_fit(model=model)
        state = SessionState(fits=[fit])
        result = set_parameter_fixed(state, parameter_name="tau", fixed="yes", fit_index=0)
        assert result.get("ok")
        assert p.fixed is True

    def test_set_parameter_bounds_nan(self):
        p = MagicMock()
        model = MagicMock()
        model.parameters_all_dict = {"tau": p}
        fit = _make_fit(model=model)
        state = SessionState(fits=[fit])
        result = set_parameter_bounds(state, parameter_name="tau", bounds=(float('nan'), float('inf')), fit_index=0)
        assert result.get("ok")

    def test_set_parameter_bounds_on_no_bounds(self):
        p = MagicMock()
        p.bounds_on = False
        model = MagicMock()
        model.parameters_all_dict = {"tau": p}
        fit = _make_fit(model=model)
        state = SessionState(fits=[fit])
        result = set_parameter_bounds_on(state, parameter_name="tau", bounds_on=True, fit_index=0)
        assert result.get("ok")

    def test_parameter_link_nonexistent_target(self):
        p = MagicMock()
        p.name = "tau1"
        model = MagicMock()
        model.parameters_all_dict = {"tau1": p}
        fit = _make_fit(model=model)
        state = SessionState(fits=[fit])
        result = parameter_link(state, parameter_name="tau1", target_parameter_name="nonexistent", fit_index=0)
        assert not result.get("ok")

    def test_parameter_unlink_not_linked(self):
        p = MagicMock()
        p.name = "tau1"
        p.link = None
        model = MagicMock()
        model.parameters_all_dict = {"tau1": p}
        fit = _make_fit(model=model)
        state = SessionState(fits=[fit])
        result = parameter_unlink(state, parameter_name="tau1", fit_index=0)
        assert result.get("ok")


class TestInputRobustnessModels:

    def test_model_finalize_nonexistent(self):
        state = SessionState()
        result = model_finalize(state, fit_index=0)
        assert not result.get("ok")

    def test_model_set_parse_function_nonexistent(self):
        state = SessionState()
        result = model_set_parse_function(state, parse_function="y=x", fit_index=0)
        assert not result.get("ok")


class TestInputRobustnessProjects:

    def test_get_project_info_empty(self):
        state = SessionState()
        result = get_project_info(state)
        assert result.get("ok")
        assert result.get("fit_count") == 0

    def test_save_project_no_path_defaults(self):
        state = SessionState()
        result = save_project(state, target_path="")
        # Empty path defaults to "project.json"
        assert result.get("ok") is True

    def test_load_project_nonexistent(self):
        state = SessionState()
        result = load_project(state, project_path="/nonexistent/file.csp")
        assert not result.get("ok")


class TestInputRobustnessGraph:

    def test_build_graph_nonexistent_fit_indices(self):
        state = SessionState()
        result = build_fit_graph(state, fit_indices=[0, 1])
        assert result.get("ok")
        assert len(result["graph"]["nodes"]) == 0


class TestStressLargeState:

    def test_one_hundred_datasets(self):
        state = SessionState()
        for i in range(100):
            ds = _make_dataset(name=f"DS{i}", uid=f"ds-{i}")
            state.add_dataset(ds)
        assert len(state.datasets) == 100
        result = list_datasets(state)
        assert len(result["datasets"]) == 100
        assert result["datasets"][0]["name"] == "DS0"
        assert result["datasets"][99]["name"] == "DS99"

    def test_one_hundred_fits(self):
        state = SessionState()
        for i in range(100):
            fit = _make_fit(name=f"Fit{i}")
            fit.unique_identifier = f"fit-{i}"
            state.add_fit(fit)
        assert len(state.fits) == 100
        result = list_fits(state)
        assert len(result["fits"]) == 100

    def test_mixed_large_state(self):
        state = SessionState()
        for i in range(50):
            ds = _make_dataset(name=f"DS{i}", uid=f"ds-{i}")
            state.add_dataset(ds)
        for i in range(50):
            fit = _make_fit(name=f"Fit{i}")
            fit.unique_identifier = f"fit-{i}"
            state.add_fit(fit)
        ds_result = list_datasets(state)
        fit_result = list_fits(state)
        assert len(ds_result["datasets"]) == 50
        assert len(fit_result["fits"]) == 50

    def test_clear_large_state(self):
        state = SessionState()
        for i in range(100):
            state.add_dataset(_make_dataset(uid=f"ds-{i}"))
        result = clear_datasets(state)
        assert result.get("cleared_count") == 100
        assert len(state.datasets) == 0


class TestIdempotency:
    """Services tolerate repeated calls safely."""

    def test_double_clear_datasets(self):
        state = SessionState()
        r1 = clear_datasets(state)
        assert r1.get("ok")
        assert r1.get("cleared_count") == 0
        r2 = clear_datasets(state)
        assert r2.get("ok")

    def test_double_clear_fits(self):
        state = SessionState()
        r1 = clear_fits(state)
        assert r1.get("ok")
        r2 = clear_fits(state)
        assert r2.get("ok")

    def test_double_remove_nonexistent(self):
        state = SessionState()
        r1 = remove_datasets(state, dataset_indices=[0])
        assert not r1.get("ok")
        r2 = remove_datasets(state, dataset_indices=[0])
        assert not r2.get("ok")
        # Same error both times, no crash

    def test_double_remove_fits_nonexistent(self):
        state = SessionState()
        r1 = remove_fits(state, fit_indices=[0])
        assert not r1.get("ok")
        r2 = remove_fits(state, fit_indices=[0])
        assert not r2.get("ok")

    def test_double_rename_same_dataset(self):
        ds = _make_dataset(name="Original")
        state = SessionState(datasets=[ds])
        r1 = dataset_rename(state, new_name="Renamed", dataset_index=0)
        assert r1.get("ok")
        r2 = dataset_rename(state, new_name="RenamedAgain", dataset_index=0)
        assert r2.get("ok")
        assert ds.name == "RenamedAgain"

    def test_double_run_fit_same_index(self):
        fit = _make_fit()
        state = SessionState(fits=[fit])
        r1 = run_fit(state, fit_index=0)
        r2 = run_fit(state, fit_index=0)
        assert isinstance(r1, dict)
        assert isinstance(r2, dict)

    def test_clear_then_remove(self):
        state = SessionState()
        clear_datasets(state)
        result = remove_datasets(state, dataset_indices=[0])
        assert not result.get("ok")


class TestFromControllerFlag:
    """The _from_controller flag suppresses event publishing in services."""

    def test_add_dataset_from_controller_no_event(self):
        state = SessionState()
        events = []
        from chisurf.server.eventbus import InProcessEventBus
        bus = InProcessEventBus()
        bus.subscribe("dataset.added", lambda e: events.append(e))
        result = add_dataset(
            state, reader_name="TestReader", filename="/tmp/f.dat",
            name="NoEvent", curve_data={"x": [], "y": []},
            _from_controller=True, event_bus=bus,
        )
        assert result.get("ok") is True
        assert len(events) == 0

    def test_add_dataset_no_flag_publishes_event(self):
        state = SessionState()
        events = []
        from chisurf.server.eventbus import InProcessEventBus
        bus = InProcessEventBus()
        bus.subscribe("dataset.added", lambda e: events.append(e))
        result = add_dataset(
            state, reader_name="TestReader", filename="/tmp/f2.dat",
            name="WithEvent", curve_data={"x": [], "y": []},
            event_bus=bus,
        )
        assert result.get("ok") is True
        assert len(events) == 1

    def test_remove_datasets_from_controller_no_event(self):
        ds = _make_dataset(name="Test", uid="ds-1")
        state = SessionState(datasets=[ds])
        events = []
        from chisurf.server.eventbus import InProcessEventBus
        bus = InProcessEventBus()
        bus.subscribe("dataset.removed", lambda e: events.append(e))
        result = remove_datasets(
            state, dataset_indices=[0],
            _from_controller=True, event_bus=bus,
        )
        assert result.get("ok") is True
        assert len(events) == 0


class TestEdgeCaseBugs:
    """Regression tests for edge-case bugs discovered during code review."""

    # --- session_restore ---

    def test_session_restore_with_project_and_event_bus(self):
        """session_restore must NOT pass event_bus to load_project (it doesn't accept it)."""
        from chisurf.server.services.session_svc import session_restore
        state = SessionState()
        event_bus = MagicMock()
        # project_path points to nonexistent dir -> load_project returns error,
        # but the call itself must not crash with TypeError
        result = session_restore(state, project_path="/nonexistent/project", event_bus=event_bus)
        # load_project should fail gracefully
        assert result.get("ok") is False
        event_bus.publish.assert_called_once()

    def test_session_restore_without_project_clears(self):
        """session_restore with no project_path just clears the session."""
        from chisurf.server.services.session_svc import session_restore
        state = SessionState()
        state.current_experiment = "TCSPC"
        result = session_restore(state, project_path=None)
        assert result.get("ok") is True
        assert state.current_experiment is None

    def test_session_restore_clear_publishes_event(self):
        """session_restore with no project_path publishes session.restored."""
        from chisurf.server.services.session_svc import session_restore
        state = SessionState()
        events = []
        from chisurf.server.eventbus import InProcessEventBus
        bus = InProcessEventBus()
        bus.subscribe("session.restored", lambda e: events.append(e))
        result = session_restore(state, event_bus=bus)
        assert result.get("ok") is True
        assert len(events) == 1

    # --- _resolve_fit in parameters.py ---

    def test_get_parameter_with_none_fit_index(self):
        """get_parameter must not crash when fit_index is None."""
        from chisurf.server.services.parameters import get_parameter
        state = SessionState()
        # No fits exist, fit_index=None should not crash the _resolve_fit helper
        result = get_parameter(state, "tau", fit_index=None)
        assert result.get("ok") is False  # fit not found (expected)

    def test_set_parameter_value_with_none_fit_index(self):
        """set_parameter_value must not crash when fit_index is None."""
        from chisurf.server.services.parameters import set_parameter_value
        state = SessionState()
        result = set_parameter_value(state, "tau", 1.0, fit_index=None)
        assert result.get("ok") is False

    # --- dataset_group / dataset_ungroup ---

    def test_dataset_group_single_int_crashes(self):
        """dataset_group must reject non-iterable dataset_indices."""
        state = SessionState()
        # Passing a single int (not a list) should return an error, not crash
        result = dataset_group(state, dataset_indices=0)
        assert result.get("ok") is False

    def test_dataset_group_empty_list_returns_error(self):
        """dataset_group with empty dataset_indices returns an error."""
        state = SessionState()
        result = dataset_group(state, dataset_indices=[])
        assert result.get("ok") is False

    def test_dataset_ungroup_single_int_crashes(self):
        """dataset_ungroup must reject non-iterable dataset_indices."""
        state = SessionState()
        result = dataset_ungroup(state, dataset_indices=0)
        assert result.get("ok") is False

    def test_dataset_ungroup_empty_list_returns_error(self):
        """dataset_ungroup with empty dataset_indices returns an error."""
        state = SessionState()
        result = dataset_ungroup(state, dataset_indices=[])
        assert result.get("ok") is False

    def test_dataset_ungroup_ungrouped_count_zero(self):
        """dataset_ungroup must return ungrouped_count=0 when nothing was ungrouped."""
        from chisurf.core.data import DataCurve
        ds = DataCurve(name="Regular", x=[1.0], y=[2.0])
        state = SessionState(datasets=[ds])
        # Ungroup a non-grouped dataset -> nothing expands
        result = dataset_ungroup(state, dataset_indices=[0])
        assert result.get("ok") is True
        assert result.get("ungrouped_count") == 0
        assert result.get("dataset_count") == 1  # dataset stays as-is

    # --- fit_create with empty dataset_indices ---

    def test_fit_create_empty_dataset_indices(self):
        """fit_create must treat dataset_indices=[] as empty list, not fallback to [0]."""
        from chisurf.server.services.fits import fit_create
        # Need at least one dataset to avoid the "no datasets available" early return
        ds = _make_dataset(name="TestData")
        state = SessionState(datasets=[ds])
        result = fit_create(state, dataset_indices=[], model_name="NonExistent")
        # Should fail because dataset_indices is empty, not silently use dataset_index=0
        assert result.get("ok") is False

    def test_fit_create_empty_dataset_indices_no_datasets(self):
        """fit_create with dataset_indices=[] and no datasets returns error."""
        from chisurf.server.services.fits import fit_create
        state = SessionState()
        result = fit_create(state, dataset_indices=[])
        assert result.get("ok") is False

    # --- NaN in curve_data arrays ---

    def test_curve_data_nan_not_crash_serialization(self):
        """get_dataset_curve_data must handle NaN/Inf without crashing JSON serialization."""
        import numpy as np
        ds = _make_dataset(
            x=[0.0, 1.0, 2.0],
            y=[float('nan'), float('inf'), 3.0],
        )
        state = SessionState(datasets=[ds])
        result = get_dataset_curve_data(state, dataset_index=0)
        assert result.get("ok") is True
        # NaN/Inf should be replaced with None
        y = result.get("y", [])
        assert y is not None
        assert len(y) == 3
        assert y[0] is None  # NaN -> None
        assert y[1] is None  # Inf -> None
        assert y[2] == 3.0

    def test_fit_curve_data_nan_not_crash(self):
        """fit_curve_data must handle NaN/Inf without crashing JSON serialization."""
        import numpy as np
        # Create a fit whose data/model contains NaN
        data = MagicMock()
        data.x = [0.0, 1.0]
        data.y = [float('nan'), float('inf')]
        data.name = "NaNData"
        data.unique_identifier = "ds-nan"
        data.filename = "/tmp/nan.dat"
        data.experiment = MagicMock()
        data.experiment.name = "Test"

        model = MagicMock()
        model.name = "TestModel"
        model.x = [0.0, 1.0]
        model.y = [float('nan'), 2.0]
        model.residuals = [float('inf'), 0.5]

        fit = _make_fit(data=data, model=model)

        from chisurf.server.services.fits import fit_curve_data
        state = SessionState(fits=[fit])
        result = fit_curve_data(state, fit_index=0)
        assert result.get("ok") is True
        # Check NaN values are replaced with None
        for key in ("y", "fy", "residuals"):
            vals = result.get(key, [])
            assert vals is not None, f"{key} should not be None"
            assert None in vals, f"{key} should contain None for NaN/Inf: {vals}"
