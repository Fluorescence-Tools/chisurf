"""PRD-12 Increment 4: mfdb-admin lifecycle RPC handlers.

The lifecycle state machine is reachable through the admin backend handlers (and thus
the MFDBClient) so the GUI view is a thin caller. Illegal transitions come back as an
``error`` field rather than an exception across the RPC boundary.
"""

from __future__ import annotations

from mfdb.admin.backend.services import (
    lifecycle_definitions_handler,
    lifecycle_history_handler,
    lifecycle_state_handler,
    lifecycle_transition_handler,
)

from .conftest import patch_db


def test_state_and_transition_handlers(db):
    with patch_db(db):
        assert lifecycle_state_handler("sample", "s1")["state"] is None
        res = lifecycle_transition_handler("sample", "s1", "registered")
        assert res["changed"] is True
        assert res["state"] == "registered"
        # idempotent no-op
        res2 = lifecycle_transition_handler("sample", "s1", "registered")
        assert res2["changed"] is False
        assert lifecycle_state_handler("sample", "s1")["state"] == "registered"


def test_illegal_transition_returns_error_not_exception(db):
    with patch_db(db):
        lifecycle_transition_handler("sample", "s2", "registered")
        res = lifecycle_transition_handler("sample", "s2", "archived")
        assert res["changed"] is False
        assert "error" in res
        # state unchanged
        assert res["state"] == "registered"


def test_history_handler_is_ordered(db):
    with patch_db(db):
        lifecycle_transition_handler("sample", "s3", "registered")
        lifecycle_transition_handler("sample", "s3", "measured")
        history = lifecycle_history_handler("sample", "s3")["history"]
    assert [h["to_state"] for h in history] == ["registered", "measured"]


def test_definitions_handler_exposes_lifecycles(db):
    with patch_db(db):
        defs = lifecycle_definitions_handler()["definitions"]
    assert "sample" in defs and "operation" in defs
    assert "registered" in defs["sample"]["states"]
    assert [None, "registered"] in defs["sample"]["transitions"]


def test_via_inprocess_client(db):
    """End to end through the MFDBClient RPC layer (InProcessClient)."""
    with patch_db(db):
        from mfdb.admin.gui.client import MFDBClient

        client = MFDBClient(inprocess=True)
        assert client.lifecycle_state("artifact", "a1") is None
        out = client.lifecycle_transition("artifact", "a1", "registered")
        assert out["changed"] is True
        assert client.lifecycle_state("artifact", "a1") == "registered"
        hist = client.lifecycle_history("artifact", "a1")
        assert [h["to_state"] for h in hist] == ["registered"]
