from __future__ import annotations

"""Tests for chisurf.server.services.parameters."""

from unittest.mock import MagicMock

from chisurf.server.services.parameters import (
    get_parameter,
    set_parameter_value,
    set_parameter_fixed,
    set_parameter_bounds,
    set_parameter_bounds_on,
    parameter_link,
    parameter_unlink,
)
from chisurf.server.session import SessionState


class TestParametersService:

    def test_get_parameter(self):
        p = MagicMock()
        p.value = 1.5
        p.fixed = False
        p.bounds = (0.0, 10.0)
        p.bounds_on = True
        p.error_estimate = 0.1
        p.link = None

        model = MagicMock()
        model.parameters_all_dict = {"tau": p}

        fit = MagicMock()
        fit.model = model
        fit.unique_identifier = "fit-uid"

        state = SessionState(fits=[fit])
        result = get_parameter(state, parameter_name="tau", fit_index=0)
        assert result["ok"]
        assert result["parameter"]["value"] == 1.5
        assert result["parameter"]["bounds"] == (0.0, 10.0)

    def test_get_parameter_unknown(self):
        model = MagicMock()
        model.parameters_all_dict = {}
        fit = MagicMock()
        fit.model = model
        fit.unique_identifier = "fit-uid"

        state = SessionState(fits=[fit])
        result = get_parameter(state, parameter_name="nonexistent", fit_index=0)
        assert not result["ok"]

    def test_set_parameter_value(self):
        p = MagicMock()
        p.value = 1.0

        model = MagicMock()
        model.parameters_all_dict = {"tau": p}
        model.update_model = MagicMock()
        model.finalize = MagicMock()

        fit = MagicMock()
        fit.model = model
        fit.unique_identifier = "fit-uid"

        state = SessionState(fits=[fit])
        result = set_parameter_value(state, parameter_name="tau", value=2.5, fit_index=0)
        assert result["ok"]
        assert p.value == 2.5
        model.update_model.assert_called_once()

    def test_set_parameter_fixed(self):
        p = MagicMock()
        p.fixed = False

        model = MagicMock()
        model.parameters_all_dict = {"tau": p}
        model.finalize = MagicMock()

        fit = MagicMock()
        fit.model = model
        fit.unique_identifier = "fit-uid"

        state = SessionState(fits=[fit])
        result = set_parameter_fixed(state, parameter_name="tau", fixed=True, fit_index=0)
        assert result["ok"]
        assert p.fixed is True

    def test_set_parameter_bounds(self):
        p = MagicMock()
        p.bounds = (0.0, 10.0)

        model = MagicMock()
        model.parameters_all_dict = {"tau": p}
        model.finalize = MagicMock()

        fit = MagicMock()
        fit.model = model
        fit.unique_identifier = "fit-uid"

        state = SessionState(fits=[fit])
        result = set_parameter_bounds(state, parameter_name="tau", bounds=(0.0, 20.0), fit_index=0)
        assert result["ok"]
        assert p.bounds == (0.0, 20.0)

    def test_set_parameter_bounds_on(self):
        p = MagicMock()
        p.bounds_on = False
        model = MagicMock()
        model.parameters_all_dict = {"tau": p}
        fit = MagicMock()
        fit.model = model
        fit.unique_identifier = "fit-uid"
        state = SessionState(fits=[fit])
        result = set_parameter_bounds_on(state, parameter_name="tau", bounds_on=True, fit_index=0)
        assert result["ok"]
        assert p.bounds_on is True

    def test_parameter_link(self):
        p_source = MagicMock()
        p_source.name = "tau1"
        p_source.link = None
        p_target = MagicMock()
        p_target.name = "tau2"
        model = MagicMock()
        model.parameters_all_dict = {"tau1": p_source, "tau2": p_target}
        fit = MagicMock()
        fit.model = model
        fit.unique_identifier = "fit-uid"
        state = SessionState(fits=[fit])
        result = parameter_link(state, parameter_name="tau1", target_parameter_name="tau2", fit_index=0)
        assert result["ok"]
        assert p_source.link is p_target

    def test_parameter_link_cross_fit(self):
        p_source = MagicMock()
        p_source.name = "tau1"
        p_source.link = None
        p_target = MagicMock()
        p_target.name = "tau2"
        model1 = MagicMock()
        model1.parameters_all_dict = {"tau1": p_source}
        model2 = MagicMock()
        model2.parameters_all_dict = {"tau2": p_target}
        fit1 = MagicMock()
        fit1.model = model1
        fit1.unique_identifier = "fit-1"
        fit2 = MagicMock()
        fit2.model = model2
        fit2.unique_identifier = "fit-2"
        state = SessionState(fits=[fit1, fit2])
        result = parameter_link(
            state, parameter_name="tau1", target_parameter_name="tau2",
            fit_index=0, target_fit_index=1,
        )
        assert result["ok"]
        assert p_source.link is p_target

    def test_parameter_link_not_found(self):
        model = MagicMock()
        model.parameters_all_dict = {"tau1": MagicMock()}
        fit = MagicMock()
        fit.model = model
        fit.unique_identifier = "fit-uid"
        state = SessionState(fits=[fit])
        result = parameter_link(state, parameter_name="nonexistent", target_parameter_name="tau2", fit_index=0)
        assert not result["ok"]

    def test_parameter_unlink(self):
        p = MagicMock()
        p.name = "tau1"
        p.link = "some_other_param"
        model = MagicMock()
        model.parameters_all_dict = {"tau1": p}
        fit = MagicMock()
        fit.model = model
        fit.unique_identifier = "fit-uid"
        state = SessionState(fits=[fit])
        result = parameter_unlink(state, parameter_name="tau1", fit_index=0)
        assert result["ok"]
        assert p.link is None

    def test_parameter_unlink_not_found(self):
        state = SessionState()
        result = parameter_unlink(state, parameter_name="nonexistent", fit_index=0)
        assert not result["ok"]

    def test_get_parameter_link_info(self):
        p = MagicMock()
        p.value = 1.5
        p.fixed = False
        p.bounds = (0.0, 10.0)
        p.bounds_on = True
        p.error_estimate = 0.1
        target = MagicMock()
        target.name = "master_tau"
        p.link = target
        model = MagicMock()
        model.parameters_all_dict = {"tau": p}
        fit = MagicMock()
        fit.model = model
        fit.unique_identifier = "fit-uid"
        state = SessionState(fits=[fit])
        result = get_parameter(state, parameter_name="tau", fit_index=0)
        assert result["ok"]
        assert result["parameter"]["linked_to"] == "master_tau"

    def test_parameter_service_fit_not_found(self):
        state = SessionState()
        result = set_parameter_value(state, parameter_name="tau", value=1.0, fit_index=0)
        assert not result["ok"]
        assert "fit not found" in result.get("error", "")
