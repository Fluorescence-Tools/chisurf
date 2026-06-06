from __future__ import annotations

import threading
from chisurf.server.eventbus import EventBus, InProcessEventBus


class TestInProcessEventBus:
    """In-memory pub/sub for testing and single-process use."""

    def test_subscribe_and_publish(self):
        bus = InProcessEventBus()
        received = []

        bus.subscribe("dataset.added", lambda e: received.append(e))
        bus.publish("dataset.added", {"uid": "abc"})

        assert len(received) == 1
        assert received[0]["uid"] == "abc"

    def test_subscribe_with_pattern(self):
        bus = InProcessEventBus()
        received = []

        bus.subscribe("fit.*", lambda e: received.append(e))
        bus.publish("fit.added", {"i": 1})
        bus.publish("fit.removed", {"i": 2})
        bus.publish("dataset.added", {"i": 3})

        assert len(received) == 2

    def test_unsubscribe(self):
        bus = InProcessEventBus()

        def handler(e):
            pass

        token = bus.subscribe("test", handler)
        bus.unsubscribe(token)

        assert len(bus._subscriptions) == 0

    def test_multiple_subscribers_same_topic(self):
        bus = InProcessEventBus()
        r1, r2 = [], []

        bus.subscribe("evt", lambda e: r1.append(1))
        bus.subscribe("evt", lambda e: r2.append(2))
        bus.publish("evt", {})

        assert r1 == [1]
        assert r2 == [2]

    def test_publish_unknown_topic_no_error(self):
        bus = InProcessEventBus()
        bus.publish("nonexistent", {})  # must not raise

    def test_subscribe_unknown_topic(self):
        bus = InProcessEventBus()
        bus.subscribe("unknown", lambda e: None)
        bus.publish("unknown", {})  # must not raise

    def test_handler_error_does_not_crash_bus(self):
        bus = InProcessEventBus()

        def broken(e):
            raise RuntimeError("boom")

        bus.subscribe("topic", broken)
        bus.subscribe("topic", lambda e: ok.append(1))

        ok = []
        bus.publish("topic", {})  # must not raise
        assert ok == [1]

    def test_clear_removes_all_subscribers(self):
        bus = InProcessEventBus()
        bus.subscribe("a", lambda e: None)
        bus.subscribe("b", lambda e: None)
        bus.clear()
        assert len(bus._subscriptions) == 0
        assert len(bus._pattern_subscriptions) == 0

    def test_event_has_timestamp(self):
        bus = InProcessEventBus()
        received = []

        bus.subscribe("evt", lambda e: received.append(e))
        bus.publish("evt", {"foo": "bar"})

        assert "timestamp" in received[0]
        assert "topic" in received[0]
        assert received[0]["foo"] == "bar"
        assert received[0]["topic"] == "evt"

    def test_concurrent_publish(self):
        """Publishing from multiple threads is safe."""
        bus = InProcessEventBus()
        n = 50
        results = []
        lock = threading.Lock()
        ready = threading.Event()

        def handler(e):
            with lock:
                results.append(e["i"])

        bus.subscribe("evt", handler)

        def publisher():
            ready.wait()
            for i in range(n):
                bus.publish("evt", {"i": i})

        t = threading.Thread(target=publisher)
        t.start()
        ready.set()
        t.join()

        assert len(results) == n

    def test_event_bus_type(self):
        bus = EventBus()
        assert isinstance(bus, EventBus)
