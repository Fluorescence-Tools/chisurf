"""PRD-12 Increment 2: lifecycle transition API.

`transition_state` validates against the seeded rules, is idempotent, records an
ordered history, and publishes PRD-21's `state.changed` event post-commit.
"""

from __future__ import annotations

import os

import pytest

from chisurf.core.mfdb.events import EVENT_STATE_CHANGED, get_event_bus
from chisurf.core.mfdb.lifecycle import StateTransitionError
from chisurf.core.mfdb.repository import MFDatabase


@pytest.fixture
def db(tmp_path):
    database = MFDatabase(os.path.join(tmp_path, "lifecycle_api.db"))
    try:
        yield database
    finally:
        database.close()


def test_initial_and_advancing_transitions_recorded(db):
    assert db.get_state("sample", "s1") is None
    assert db.transition_state("sample", "s1", "registered") is True
    assert db.get_state("sample", "s1") == "registered"
    assert db.transition_state("sample", "s1", "measured", reason="ran acquisition") is True
    assert db.get_state("sample", "s1") == "measured"


def test_illegal_transition_rejected(db):
    db.transition_state("sample", "s2", "registered")
    # registered -> archived is not a declared rule
    with pytest.raises(StateTransitionError):
        db.transition_state("sample", "s2", "archived")
    # state is unchanged after the rejected jump
    assert db.get_state("sample", "s2") == "registered"


def test_illegal_initial_state_rejected(db):
    # "measured" is not an allowed initial state (only "registered" is)
    with pytest.raises(StateTransitionError):
        db.transition_state("sample", "s3", "measured")
    assert db.get_state("sample", "s3") is None


def test_idempotent_re_transition_is_noop(db):
    assert db.transition_state("artifact", "a1", "registered") is True
    # already registered -> no-op, returns False, no new row
    assert db.transition_state("artifact", "a1", "registered") is False
    assert len(db.get_state_history("artifact", "a1")) == 1


def test_history_is_ordered_with_from_states(db):
    db.transition_state("sample", "s4", "registered")
    db.transition_state("sample", "s4", "measured")
    db.transition_state("sample", "s4", "processed")
    history = db.get_state_history("sample", "s4")
    assert [h["to_state"] for h in history] == ["registered", "measured", "processed"]
    assert [h["from_state"] for h in history] == [None, "registered", "measured"]
    assert history[1]["reason"] is None or isinstance(history[1]["reason"], str)


def test_branching_operation_lifecycle(db):
    db.transition_state("operation", "op1", "pending")
    db.transition_state("operation", "op1", "running")
    db.transition_state("operation", "op1", "failed")
    assert db.get_state("operation", "op1") == "failed"
    # running -> failed was legal; succeeded -> running is not
    db.transition_state("operation", "op2", "pending")
    db.transition_state("operation", "op2", "running")
    db.transition_state("operation", "op2", "succeeded")
    with pytest.raises(StateTransitionError):
        db.transition_state("operation", "op2", "running")


def test_state_changed_event_published(db):
    seen = []
    bus = get_event_bus()

    def handler(event):
        seen.append(event)

    bus.subscribe(EVENT_STATE_CHANGED, handler)
    try:
        db.transition_state("sample", "s5", "registered", operator_user_id="u1")
    finally:
        bus.unsubscribe(EVENT_STATE_CHANGED, handler)

    assert len(seen) == 1
    payload = seen[0]
    assert payload["entity_type"] == "sample"
    assert payload["entity_id"] == "s5"
    assert payload["from_state"] is None
    assert payload["to_state"] == "registered"
    assert payload["operator_user_id"] == "u1"


def test_idempotent_noop_publishes_nothing(db):
    db.transition_state("artifact", "a2", "registered")
    seen = []
    bus = get_event_bus()
    bus.subscribe(EVENT_STATE_CHANGED, lambda e: seen.append(e))
    try:
        # already in state -> no-op -> no event
        assert db.transition_state("artifact", "a2", "registered") is False
    finally:
        bus.clear()
    assert seen == []


def test_audit_log_records_transition(db):
    db.transition_state("sample", "s6", "registered", operator_user_id="u2")
    logs = db.get_audit_logs(action="transition", target_type="state:sample", target_id="s6")
    assert logs
    assert logs[0]["target_id"] == "s6"
