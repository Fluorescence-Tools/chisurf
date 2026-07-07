"""PRD-26 Task 1: the dictionary-/schema-driven DAO core.

Exercises generic parameterised CRUD against a real dictionary-declared table
(``flr_sample``) so the DAO is proven against the live schema, not a mock.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from mfdb.dao import (
    DictionaryDao,
    UnknownColumnError,
    UnknownTableError,
)
from mfdb.repository import MFDatabase


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


def test_get_sample_migrated_to_dao_returns_dict(db):
    """The DAO-backed ``get_sample`` returns a dict (not a sqlite3.Row).

    Callers index by key and some already use dict ``.get(...)``; the dict must
    cover the former column subset and ``None`` for a missing sample.
    """
    db.add_sample("samp", description="hello", num_of_probes=1)
    row = db.get_sample("samp")
    assert isinstance(row, dict)
    # superset of the former subset; key access (and .get) works
    assert row.get("sample_uuid")
    assert row["description"] == "hello"
    assert db.get_sample("missing") is None


def test_get_sample_returns_soft_deleted(db):
    """``get_sample`` had no deleted_at filter; migration keeps that semantics."""
    db.add_sample("samp2", description="x", num_of_probes=1)
    db.dao.soft_delete("flr_sample", "samp2", pk_column="sample_id")
    row = db.get_sample("samp2")
    assert row is not None and row["deleted_at"] is not None


def test_delete_parameter_migrated_to_dao_soft_deletes_by_either_key(db):
    """delete_parameter soft-deletes by parameter_uuid *or* parameter_id."""
    dao = _dao(db)
    dao.insert(
        "mfdb_parameter",
        {"parameter_id": 101, "parameter_uuid": "p-uuid", "name": "k", "value": 1.0},
    )
    # delete by the uuid key
    db.delete_parameter("p-uuid")
    assert dao.get("mfdb_parameter", 101, pk_column="parameter_id") is None
    assert (
        dao.get("mfdb_parameter", 101, pk_column="parameter_id", include_deleted=True)[
            "deleted_at"
        ]
        is not None
    )
    # a second row deleted by the integer key
    dao.insert(
        "mfdb_parameter",
        {"parameter_id": 102, "parameter_uuid": "p-uuid-2", "name": "k", "value": 1.0},
    )
    db.delete_parameter(102)
    assert dao.get("mfdb_parameter", 102, pk_column="parameter_id") is None


def test_upsert_inserts_then_updates_in_place(db):
    """upsert inserts a new row, then updates the same primary key in place."""
    dao = _dao(db)
    pk = dao.upsert("flr_sample", {"sample_id": "u1", "description": "first"})
    assert pk == "u1"
    assert dao.get("flr_sample", "u1")["description"] == "first"

    # Second upsert on the same PK updates rather than raising on conflict.
    dao.upsert("flr_sample", {"sample_id": "u1", "description": "second"})
    rows = dao.list("flr_sample", filters={"sample_id": "u1"})
    assert len(rows) == 1, "upsert must not create a duplicate row"
    assert rows[0]["description"] == "second"


def test_upsert_preserves_unspecified_columns(db):
    """Unlike INSERT OR REPLACE, upsert leaves columns absent from values intact."""
    dao = _dao(db)
    dao.upsert("flr_sample", {"sample_id": "u2", "description": "d", "details": "keepme"})
    # Update only description; details must survive (INSERT OR REPLACE would null it).
    dao.upsert("flr_sample", {"sample_id": "u2", "description": "d2"})
    row = dao.get("flr_sample", "u2")
    assert row["description"] == "d2"
    assert row["details"] == "keepme"


def test_upsert_touches_updated_at(db):
    """The update branch refreshes updated_at when the table declares it."""
    dao = _dao(db)
    dao.upsert(
        "flr_sample",
        {"sample_id": "u3", "description": "d", "updated_at": "2000-01-01T00:00:00+00:00"},
    )
    dao.upsert("flr_sample", {"sample_id": "u3", "description": "d2"})
    row = dao.get("flr_sample", "u3")
    assert row["updated_at"] != "2000-01-01T00:00:00+00:00"


def test_upsert_on_natural_key_preserves_identity_pk(db):
    """upsert on a UNIQUE natural key keeps the auto-assigned primary key stable.

    This is the FK-preserving identity-row property: repeated writes reuse the
    same ``type_id`` (INSERT OR REPLACE would delete-and-reinsert, changing it).
    """
    dao = _dao(db)
    dao.upsert("probe_types", {"type_name": "dye", "display_name": "Dye"}, conflict="type_name")
    first = dao.list("probe_types", filters={"type_name": "dye"})[0]
    type_id = first["type_id"]

    dao.upsert("probe_types", {"type_name": "dye", "display_name": "Dye v2"}, conflict="type_name")
    rows = dao.list("probe_types", filters={"type_name": "dye"})
    assert len(rows) == 1
    assert rows[0]["type_id"] == type_id, "identity PK must be preserved across upserts"
    assert rows[0]["display_name"] == "Dye v2"


def test_upsert_rejects_unknown_table_and_column(db):
    dao = _dao(db)
    with pytest.raises(UnknownTableError):
        dao.upsert("not_a_table", {"x": 1})
    with pytest.raises(UnknownColumnError):
        dao.upsert("flr_sample", {"sample_id": "u4", "bogus_col": 1})
