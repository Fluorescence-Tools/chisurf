"""PRD-26 Task 1: the dictionary-/schema-driven DAO core.

Exercises generic parameterised CRUD against a real dictionary-declared table
(``flr_sample``) so the DAO is proven against the live schema, not a mock.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from chisurf.core.mfdb.dao import (
    DictionaryDao,
    UnknownColumnError,
    UnknownTableError,
)
from chisurf.core.mfdb.repository import MFDatabase


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as tmpdir:
        database = MFDatabase(os.path.join(tmpdir, "test.db"))
        try:
            yield database
        finally:
            database.close()


def _dao(db) -> DictionaryDao:
    return DictionaryDao.from_connection(db.conn)


def test_primary_key_and_columns_from_live_schema(db):
    dao = _dao(db)
    assert dao.has_table("flr_sample")
    assert dao.primary_key("flr_sample") == "sample_id"
    cols = dao.columns("flr_sample")
    assert {"sample_id", "description"} <= cols


def test_crud_round_trip_with_soft_delete(db):
    dao = _dao(db)
    pk = dao.insert("flr_sample", {"sample_id": "s1", "description": "DNA"})
    assert pk == "s1"

    row = dao.get("flr_sample", "s1")
    assert row is not None and row["description"] == "DNA"

    assert dao.update("flr_sample", "s1", {"description": "DNA duplex"}) == 1
    assert dao.get("flr_sample", "s1")["description"] == "DNA duplex"

    listed = dao.list("flr_sample", filters={"sample_id": "s1"})
    assert len(listed) == 1 and listed[0]["sample_id"] == "s1"

    # Soft-delete hides the row from default reads but keeps it retrievable.
    assert dao.soft_delete("flr_sample", "s1") == 1
    assert dao.get("flr_sample", "s1") is None
    assert dao.list("flr_sample", filters={"sample_id": "s1"}) == []
    deleted = dao.get("flr_sample", "s1", include_deleted=True)
    assert deleted is not None and deleted["deleted_at"] is not None


def test_soft_delete_accepts_explicit_deleted_at_value(db):
    """An explicit deleted_at value is stored verbatim (repo passes _utc_now())."""
    dao = _dao(db)
    dao.insert("flr_sample", {"sample_id": "sd", "description": "x"})
    marker = "2024-01-02T03:04:05+00:00"
    assert dao.soft_delete("flr_sample", "sd", deleted_at=marker) == 1
    row = dao.get("flr_sample", "sd", include_deleted=True)
    assert row["deleted_at"] == marker
    # Idempotent: re-deleting an already-deleted row affects no rows.
    assert dao.soft_delete("flr_sample", "sd", deleted_at=marker) == 0


def test_delete_setup_migrated_to_dao(db):
    """The DAO-backed delete_setup soft-deletes the setup (sets deleted_at)."""
    db.add_setup("setup-1", name="S1")
    dao = _dao(db)
    assert dao.get("mfdb_setup", "setup-1")["deleted_at"] is None
    db.delete_setup("setup-1")
    # Hidden from default reads; the deleted_at marker is set.
    assert dao.get("mfdb_setup", "setup-1") is None
    assert dao.get("mfdb_setup", "setup-1", include_deleted=True)["deleted_at"] is not None


def test_unknown_table_and_column_are_rejected(db):
    dao = _dao(db)
    with pytest.raises(UnknownTableError):
        dao.insert("not_a_table", {"x": 1})
    with pytest.raises(UnknownColumnError):
        dao.insert("flr_sample", {"sample_id": "s2", "bogus_col": 1})
    with pytest.raises(UnknownColumnError):
        dao.list("flr_sample", filters={"bogus_col": 1})
    with pytest.raises(UnknownColumnError):
        dao.update("flr_sample", "s2", {"bogus_col": 1})


def test_primary_key_cannot_be_updated(db):
    dao = _dao(db)
    dao.insert("flr_sample", {"sample_id": "s3", "description": "x"})
    with pytest.raises(Exception):
        dao.update("flr_sample", "s3", {"sample_id": "s3b"})


def test_get_artifact_migrated_to_dao_returns_soft_deleted(db):
    """The DAO-backed ``get_artifact`` keeps returning soft-deleted artifacts.

    ``get_artifact`` historically returned rows regardless of ``deleted_at``;
    migrating it onto ``DictionaryDao.get(..., include_deleted=True)`` must
    preserve that (callers that want live-only filter explicitly).
    """
    dao = _dao(db)
    dao.insert(
        "mfdb_artifact",
        {"artifact_id": "a1", "artifact_kind": "burst_table", "storage_mode": "embedded"},
    )
    fetched = db.get_artifact("a1")
    assert fetched is not None and fetched["artifact_kind"] == "burst_table"

    dao.soft_delete("mfdb_artifact", "a1")
    still = db.get_artifact("a1")
    assert still is not None, "get_artifact must still return soft-deleted artifacts"
    assert still["deleted_at"] is not None
    assert db.get_artifact("does-not-exist") is None


def test_values_are_parameterised_no_injection(db):
    """A SQL-injection-shaped value is stored verbatim; the table survives."""
    dao = _dao(db)
    payload = "'); DROP TABLE flr_sample;-- O'Brien"
    dao.insert("flr_sample", {"sample_id": "s4", "description": payload})
    row = dao.get("flr_sample", "s4")
    assert row["description"] == payload
    # The table still exists and is queryable (injection did not execute).
    assert dao.get("flr_sample", "s4") is not None
    assert dao.has_table("flr_sample")
