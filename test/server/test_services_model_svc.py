from __future__ import annotations

"""Tests for chisurf.server.services.model_svc."""

from unittest.mock import MagicMock

from chisurf.server.services.model_svc import model_component_remove
from chisurf.server.session import SessionState


class TestModelComponentRemove:

    def test_remove_fit_not_found(self):
        state = SessionState()
        result = model_component_remove(state, component_index=0)
        assert not result["ok"]
        assert "fit not found" in result.get("error", "")

    def test_remove_fit_no_model(self):
        fit = MagicMock()
        fit.model = None
        state = SessionState(fits=[fit])
        result = model_component_remove(state, component_index=0, fit_index=0)
        assert not result["ok"]
        assert "no model" in result.get("error", "")

    def test_remove_via_remove_component_method(self):
        model = MagicMock()
        model.remove_component = MagicMock()
        fit = MagicMock()
        fit.model = model
        state = SessionState(fits=[fit])
        result = model_component_remove(state, component_index=2, fit_index=0)
        assert result["ok"]
        model.remove_component.assert_called_once_with(2)

    def test_remove_via_pop_fallback(self):
        model = MagicMock()
        model.remove_component = None
        lifetimes = MagicMock()
        lifetimes.pop = MagicMock(return_value="popped")
        model.lifetimes = lifetimes
        fit = MagicMock()
        fit.model = model
        state = SessionState(fits=[fit])
        result = model_component_remove(state, component_index=0, fit_index=0)
        assert result["ok"]
        lifetimes.pop.assert_called_once()

    def test_remove_no_method_available(self):
        model = MagicMock()
        model.remove_component = None
        model.lifetimes = None
        model.species = None
        model.rotations = None
        model.distances = None
        model.gaussians = None
        fit = MagicMock()
        fit.model = model
        state = SessionState(fits=[fit])
        result = model_component_remove(state, component_index=0, fit_index=0)
        assert not result["ok"]

    def test_remove_no_nameerror_on_fallback(self):
        """Regression test for BUG-03: undefined component_type in fallback tuple."""
        model = MagicMock()
        model.remove_component = None
        model.lifetimes = None
        model.species = None
        model.rotations = None
        model.distances = None
        model.gaussians = None
        fit = MagicMock()
        fit.model = model
        state = SessionState(fits=[fit])
        try:
            result = model_component_remove(state, component_index=0, fit_index=0)
            assert not result["ok"]
        except NameError:
            assert False, "model_component_remove raised NameError (BUG-03)"
