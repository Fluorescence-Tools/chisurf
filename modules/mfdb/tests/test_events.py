"""PRD-21 Task 3: the post-commit MFDB event bus.

Asserts the bus contract (subscribe/publish, ordering, best-effort isolation) and
that registration publishes ``artifact.registered`` / ``operation.succeeded`` after
the transaction commits — with a subscriber that consumes them and one that raises
without breaking registration.
"""

from __future__ import annotations

import os

import pytest

from mfdb.lifecycle.events import (
    EVENT_ARTIFACT_REGISTERED,
    EVENT_OPERATION_SUCCEEDED,
    Event,
    EventBus,
    get_event_bus,
)
from mfdb.repository import MFDatabase
from mfdb.provenance.result_registry import (
    register_operation,
    register_raw_measurement,
    register_result,
    set_global_db,
)


# -- bus unit behaviour ------------------------------------------------------


def test_subscribe_and_publish_delivers_payload():
    bus = EventBus()
    seen: list[Event] = []
    bus.subscribe("x.happened", seen.append)
    bus.publish("x.happened", a=1, b="two")
    assert len(seen) == 1
    assert seen[0].name == "x.happened"
    assert seen[0]["a"] == 1 and seen[0].get("b") == "two"
    assert seen[0].timestamp  # stamped


def test_only_matching_and_global_subscribers_run():
    bus = EventBus()
    matched, other, every = [], [], []
    bus.subscribe("a", matched.append)
    bus.subscribe("b", other.append)
    bus.subscribe_all(every.append)
    bus.publish("a", n=1)
    assert len(matched) == 1 and other == [] and len(every) == 1


def test_failing_handler_is_isolated():
    bus = EventBus()
    ran = []

    def boom(_event):
        raise RuntimeError("handler error")

    bus.subscribe("e", boom)
    bus.subscribe("e", ran.append)
    # publish must not raise even though the first handler does
    bus.publish("e", k="v")
    assert len(ran) == 1  # the good handler still ran


def test_unsubscribe_and_clear():
    bus = EventBus()
    seen = []
    h = bus.subscribe("e", seen.append)
    bus.unsubscribe("e", h)
    bus.publish("e")
    assert seen == []
    bus.subscribe("e", seen.append)
    bus.clear()
    bus.publish("e")
    assert seen == []


# -- wired into registration (post-commit) -----------------------------------


@pytest.fixture
def captured_bus():
    """Capture events on the default bus; restore subscribers afterwards."""
    bus = get_event_bus()
    events: list[Event] = []
    handler = bus.subscribe_all(events.append)
    try:
        yield events
    finally:
        bus.unsubscribe_all(handler)


def test_register_result_publishes_artifact_registered(tmp_path, captured_bus):
    db = MFDatabase(os.path.join(tmp_path, "ev.db"))
    try:
        f = tmp_path / "m.ptu"
        f.write_bytes(b"\x00\x01")
        raw = register_raw_measurement(str(f), db=db)
        names = [(e.name, e.get("artifact_id")) for e in captured_bus]
        assert (EVENT_ARTIFACT_REGISTERED, raw) in names
    finally:
        set_global_db(None)
        db.close()


def test_register_operation_publishes_operation_succeeded(tmp_path, captured_bus):
    db = MFDatabase(os.path.join(tmp_path, "ev.db"))
    try:
        f = tmp_path / "m.ptu"
        f.write_bytes(b"\x00\x01")
        raw = register_raw_measurement(str(f), db=db)
        captured_bus.clear()
        op = register_operation(
            operation_type="microtime_shift",
            inputs=[raw],
            outputs=[],
            parameters={"global_shift": 1},
            db=db,
        )
        op_events = [e for e in captured_bus if e.name == EVENT_OPERATION_SUCCEEDED]
        assert len(op_events) == 1
        assert op_events[0]["operation_id"] == op
        assert op_events[0]["inputs"] == [raw]
    finally:
        set_global_db(None)
        db.close()


def test_subscriber_failure_does_not_break_registration(tmp_path):
    bus = get_event_bus()

    def boom(_event):
        raise RuntimeError("subscriber blew up")

    handler = bus.subscribe(EVENT_ARTIFACT_REGISTERED, boom)
    db = MFDatabase(os.path.join(tmp_path, "ev.db"))
    try:
        f = tmp_path / "m.ptu"
        f.write_bytes(b"\x00\x01")
        # registration must still succeed despite the raising subscriber
        raw = register_raw_measurement(str(f), db=db)
        assert raw
        assert db.get_artifact(raw) is not None
    finally:
        bus.unsubscribe(EVENT_ARTIFACT_REGISTERED, handler)
        set_global_db(None)
        db.close()
