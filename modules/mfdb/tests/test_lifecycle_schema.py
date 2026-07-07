"""PRD-12 Increment 1: lifecycle schema foundation.

A fresh MFDatabase gets the `.dic`-declared transition tables (created by
reconcile_schema) plus the seeded per-entity-type state vocabularies and transition
rules from the authored source. No transition API yet (that is Increment 2) — this
asserts the foundation is in place and idempotent.
"""

from __future__ import annotations

import os

import pytest

from mfdb.lifecycle import (
    bootstrap_lifecycle_defs,
    get_lifecycle_def,
    load_lifecycle_defs,
    state_field,
)
from mfdb.repository import MFDatabase


@pytest.fixture
def db(tmp_path):
    database = MFDatabase(os.path.join(tmp_path, "lifecycle.db"))
    try:
        yield database
    finally:
        database.close()


def test_transition_tables_exist(db):
    tables = {
        r[0]
        for r in db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "mfdb_state_transition" in tables
    assert "mfdb_state_transition_rule" in tables


def test_transition_log_has_expected_columns(db):
    cols = {r[1] for r in db.conn.execute("PRAGMA table_info(mfdb_state_transition)")}
    assert {
        "transition_id", "entity_type", "entity_id", "from_state", "to_state",
        "reason", "operator_user_id", "created_at", "updated_at", "deleted_at",
    } <= cols


def test_state_vocabularies_seeded(db):
    for entity_type, ld in load_lifecycle_defs().items():
        rows = {
            r[0]
            for r in db.conn.execute(
                "SELECT value FROM mfdb_vocabulary WHERE field_name = ?",
                (state_field(entity_type),),
            ).fetchall()
        }
        assert set(ld.states) <= rows, f"missing states for {entity_type}: {set(ld.states) - rows}"


def test_transition_rules_seeded(db):
    sample = get_lifecycle_def("sample")
    assert sample is not None
    rows = {
        (r[0], r[1])
        for r in db.conn.execute(
            "SELECT from_state, to_state FROM mfdb_state_transition_rule "
            "WHERE entity_type = 'sample'"
        ).fetchall()
    }
    # the authored initial + advancing transitions are present
    assert (None, "registered") in rows
    assert ("registered", "measured") in rows
    assert ("validated", "archived") in rows
    # an illegal jump is NOT a declared rule
    assert ("registered", "archived") not in rows


def test_bootstrap_is_idempotent(db):
    before = db.conn.execute(
        "SELECT COUNT(*) FROM mfdb_state_transition_rule"
    ).fetchone()[0]
    bootstrap_lifecycle_defs(db.conn)
    bootstrap_lifecycle_defs(db.conn)
    after = db.conn.execute(
        "SELECT COUNT(*) FROM mfdb_state_transition_rule"
    ).fetchone()[0]
    assert before == after and before > 0


def test_lifecycle_def_allows_matches_rules(db):
    op = get_lifecycle_def("operation")
    assert op.allows(None, "pending")
    assert op.allows("running", "failed")
    assert not op.allows("succeeded", "running")
    assert op.initial_states == ("pending",)
