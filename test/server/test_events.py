from __future__ import annotations

"""Tests for server event subscription / publication."""

from unittest.mock import MagicMock
import pytest

from chisurf.server.eventbus import InProcessEventBus
from chisurf.server.dispatcher import ServiceDispatcher
from chisurf.server.session import SessionState


class TestEventBusIntegration:

    @pytest.fixture
    def event_bus(self):
        return InProcessEventBus()

    @pytest.fixture
    def dispatcher(self, event_bus):
        state = SessionState()
        d = ServiceDispatcher(state, event_bus=event_bus)
        d._build_default_registry()
        return d

    def test_fit_create_publishes_event(self, event_bus, dispatcher):
        events = []
        event_bus.subscribe("fit.created", lambda e: events.append(e))

        result = dispatcher.dispatch("fit.create", {
            "dataset_index": 0,
            "model_name": "Lifetime fit",
        })
        # Should fail because no datasets, but event should not be published
        assert not result["ok"]
        assert len(events) == 0

    def test_fit_clear_publishes_event(self, event_bus, dispatcher):
        events = []
        event_bus.subscribe("fit.cleared", lambda e: events.append(e))

        state = dispatcher._state
        fit = MagicMock()
        fit.unique_identifier = "f1"
        state.add_fit(fit)

        result = dispatcher.dispatch("fit.clear", {})
        assert result["ok"]
        assert len(events) == 1
        assert events[0]["cleared_count"] == 1

    def test_fit_remove_publishes_event(self, event_bus, dispatcher):
        events = []
        event_bus.subscribe("fit.removed", lambda e: events.append(e))

        state = dispatcher._state
        fit = MagicMock()
        fit.unique_identifier = "f1"
        state.add_fit(fit)

        result = dispatcher.dispatch("fit.remove", {"fit_indices": [0]})
        assert result["ok"]
        assert len(events) == 1
        assert events[0]["removed_count"] == 1

    def test_dataset_clear_publishes_event(self, event_bus, dispatcher):
        events = []
        event_bus.subscribe("dataset.cleared", lambda e: events.append(e))

        state = dispatcher._state
        ds = MagicMock()
        ds.unique_identifier = "ds1"
        state.add_dataset(ds)

        result = dispatcher.dispatch("dataset.clear", {})
        assert result["ok"]
        assert len(events) == 1
        assert events[0]["cleared_count"] == 1

    def test_fit_run_publishes_event(self, event_bus, dispatcher):
        events = []
        event_bus.subscribe("fit.ran", lambda e: events.append(e))

        state = dispatcher._state
        fit = MagicMock()
        fit.unique_identifier = "f1"
        fit.chi2 = 3.0
        fit.run = MagicMock()
        fit.model = MagicMock()
        fit.model.parameters_all_dict = {}
        state.add_fit(fit)

        result = dispatcher.dispatch("fit.run", {"fit_index": 0})
        assert result["ok"]
        assert len(events) == 1
        assert events[0]["fit_index"] == 0

    def test_wildcard_subscription(self, event_bus, dispatcher):
        events = []
        event_bus.subscribe("fit.*", lambda e: events.append(e))

        state = dispatcher._state
        fit = MagicMock()
        fit.unique_identifier = "f1"
        state.add_fit(fit)

        dispatcher.dispatch("fit.clear", {})
        assert len(events) == 1
        assert "cleared_count" in events[0]

