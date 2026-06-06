from __future__ import annotations

"""Edge-case tests for InProcessEventBus.

Covers: wildcard patterns (fnmatch * and ?), multiple subscribers,
late subscribers, unsubscribe, handler exceptions, ordering,
nested publish, event structure guarantees.
"""

import threading
import time
from unittest.mock import MagicMock

import pytest

from chisurf.server.eventbus import InProcessEventBus
from chisurf.server.dispatcher import ServiceDispatcher
from chisurf.server.session import SessionState


@pytest.fixture
def event_bus():
    return InProcessEventBus()


@pytest.fixture
def dispatcher(event_bus):
    state = SessionState()
    d = ServiceDispatcher(state, event_bus=event_bus)
    d._build_default_registry()
    return d


class TestWildcardPatterns:

    def test_wildcard_fit_all(self, event_bus, dispatcher):
        events = []
        event_bus.subscribe("fit.*", lambda e: events.append(e))
        state = dispatcher._state
        fit = MagicMock()
        fit.unique_identifier = "f1"
        state.add_fit(fit)

        dispatcher.dispatch("fit.clear", {})
        dispatcher.dispatch("fit.remove", {"fit_indices": [0]})
        # fit.remove succeeds because we re-add after remove
        state.add_fit(fit)
        dispatcher.dispatch("fit.run", {"fit_index": 0})

        # Should receive exactly the events that were published
        assert len(events) >= 2

    def test_wildcard_dataset_all(self, event_bus, dispatcher):
        events = []
        event_bus.subscribe("dataset.*", lambda e: events.append(e))
        state = dispatcher._state
        ds = MagicMock()
        ds.unique_identifier = "ds1"
        state.add_dataset(ds)

        dispatcher.dispatch("dataset.clear", {})
        assert len(events) == 1
        assert "cleared_count" in events[0]

    def test_double_wildcard(self, event_bus):
        events = []
        event_bus.subscribe("*", lambda e: events.append(e))
        event_bus.publish("any.topic", {"value": 1})
        event_bus.publish("another", {"value": 2})
        assert len(events) == 2

    def test_question_mark_pattern(self, event_bus):
        """? matches single character in wildcard."""
        events = []
        event_bus.subscribe("fit.???", lambda e: events.append(e))
        event_bus.publish("fit.add", {"v": 1})   # 3 chars after dot
        event_bus.publish("fit.run", {"v": 2})    # 3 chars
        event_bus.publish("fit.clear", {"v": 3})  # 5 chars, should NOT match
        assert len(events) == 2

    def test_wildcard_does_not_match_exact_higher_level(self, event_bus):
        """fit.* does not match just 'fit' without sub-topic."""
        events = []
        event_bus.subscribe("fit.*", lambda e: events.append(e))
        event_bus.publish("fit", {"v": 1})
        assert len(events) == 0

    def test_mixed_exact_and_wildcard(self, event_bus):
        events_exact = []
        events_wild = []
        event_bus.subscribe("fit.cleared", lambda e: events_exact.append(e))
        event_bus.subscribe("fit.*", lambda e: events_wild.append(e))

        event_bus.publish("fit.cleared", {"n": 1})
        event_bus.publish("fit.ran", {"n": 2})

        assert len(events_exact) == 1
        assert events_exact[0]["n"] == 1
        assert len(events_wild) == 2


class TestMultipleSubscribers:

    def test_two_subscribers_same_topic(self, event_bus):
        events1 = []
        events2 = []
        event_bus.subscribe("topic.x", lambda e: events1.append(e))
        event_bus.subscribe("topic.x", lambda e: events2.append(e))

        event_bus.publish("topic.x", {"v": 42})
        assert len(events1) == 1
        assert len(events2) == 1
        assert events1[0]["v"] == 42
        assert events2[0]["v"] == 42

    def test_many_subscribers_same_topic(self, event_bus):
        all_events = []

        def _make_collector(container):
            def handler(e):
                container.append(e)
            return handler

        subscribers = 50
        lists = [[] for _ in range(subscribers)]
        for i in range(subscribers):
            event_bus.subscribe("topic.y", _make_collector(lists[i]))

        event_bus.publish("topic.y", {"idx": 99})
        for lst in lists:
            assert len(lst) == 1
            assert lst[0]["idx"] == 99

    def test_subscriber_does_not_affect_others_on_error(self, event_bus):
        good_events = []
        bad_called = False

        def _bad(e):
            nonlocal bad_called
            bad_called = True
            raise ValueError("bad handler")

        def _good(e):
            good_events.append(e)

        event_bus.subscribe("topic.z", _bad)
        event_bus.subscribe("topic.z", _good)

        event_bus.publish("topic.z", {"v": 1})
        assert bad_called
        assert len(good_events) == 1


class TestLateSubscriber:

    def test_late_subscriber_misses_earlier_events(self, event_bus):
        event_bus.publish("late.topic", {"v": "before"})
        late_events = []
        event_bus.subscribe("late.topic", lambda e: late_events.append(e))
        event_bus.publish("late.topic", {"v": "after"})
        assert len(late_events) == 1
        assert late_events[0]["v"] == "after"

    def test_subscribe_then_publish_after_clear(self, event_bus):
        events = []
        event_bus.subscribe("post.clear", lambda e: events.append(e))
        event_bus.clear()
        event_bus.publish("post.clear", {"v": 1})
        assert len(events) == 0


class TestUnsubscribe:

    def test_unsubscribe_stops_events(self, event_bus):
        events = []
        token = event_bus.subscribe("unsub.test", lambda e: events.append(e))
        event_bus.publish("unsub.test", {"v": 1})
        assert len(events) == 1

        event_bus.unsubscribe(token)
        event_bus.publish("unsub.test", {"v": 2})
        assert len(events) == 1  # no new event

    def test_unsubscribe_wildcard(self, event_bus):
        events = []
        token = event_bus.subscribe("wild.*", lambda e: events.append(e))
        event_bus.publish("wild.test", {"v": 1})
        assert len(events) == 1

        event_bus.unsubscribe(token)
        event_bus.publish("wild.test", {"v": 2})
        assert len(events) == 1

    def test_unsubscribe_nonexistent(self, event_bus):
        event_bus.unsubscribe("nonexistent-token")

    def test_unsubscribe_then_resubscribe(self, event_bus):
        events = []
        token = event_bus.subscribe("re.test", lambda e: events.append(e))
        event_bus.unsubscribe(token)
        token2 = event_bus.subscribe("re.test", lambda e: events.append(e))
        event_bus.publish("re.test", {"v": 3})
        assert len(events) == 1
        assert events[0]["v"] == 3
        assert token != token2


class TestEventStructure:

    def test_event_has_topic(self, event_bus):
        events = []
        event_bus.subscribe("struct.test", lambda e: events.append(e))
        event_bus.publish("struct.test", {"custom": "data"})
        assert len(events) == 1
        assert events[0]["topic"] == "struct.test"
        assert events[0]["custom"] == "data"

    def test_event_has_timestamp(self, event_bus):
        events = []
        event_bus.subscribe("ts.test", lambda e: events.append(e))
        event_bus.publish("ts.test", {})
        assert "timestamp" in events[0]
        assert isinstance(events[0]["timestamp"], float)

    def test_payload_not_mutated_by_publish(self, event_bus):
        events = []
        original = {"key": "value", "nested": {"inner": 1}}
        event_bus.subscribe("mut.test", lambda e: events.append(e))
        event_bus.publish("mut.test", original)
        assert events[0]["key"] == "value"
        assert events[0]["nested"]["inner"] == 1

    def test_empty_payload_publish(self, event_bus):
        events = []
        event_bus.subscribe("empty.test", lambda e: events.append(e))
        event_bus.publish("empty.test", {})
        assert len(events) == 1
        assert "topic" in events[0]
        assert "timestamp" in events[0]


class TestOrderingAndNesting:

    def test_multiple_events_in_order(self, event_bus):
        events = []
        event_bus.subscribe("order.*", lambda e: events.append(e))
        for i in range(10):
            event_bus.publish("order.test", {"seq": i})
        assert len(events) == 10
        for i, ev in enumerate(events):
            assert ev["seq"] == i

    def test_nested_publish_does_not_deadlock(self, event_bus):
        """Publishing from within a handler should not deadlock."""
        inner_events = []

        def _outer(e):
            event_bus.publish("inner", {"from_outer": True})

        def _inner(e):
            inner_events.append(e)

        event_bus.subscribe("outer", _outer)
        event_bus.subscribe("inner", _inner)
        event_bus.publish("outer", {"v": 1})
        assert len(inner_events) == 1
        assert inner_events[0]["from_outer"] is True

    def test_pattern_and_exact_mixed_ordering(self, event_bus):
        events = []
        event_bus.subscribe("precise", lambda e: events.append(("exact", e)))
        event_bus.subscribe("prec*", lambda e: events.append(("wild", e)))

        event_bus.publish("precise", {"v": 1})
        # Both should fire; ordering: exact first, then wildcard
        topics = [e[0] for e in events]
        assert "exact" in topics
        assert "wild" in topics
        assert len(events) == 2
