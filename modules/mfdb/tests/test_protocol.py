"""PRD-14 Increment 1: protocol entity — schema + versioned CRUD.

`mfdb_protocol` is `.dic`-declared (created by reconcile_schema); operations carry
`protocol_id`/`protocol_version` columns. Protocols are append-only by version and
scoped own/public; their parameter schema is the operation_type's PRD-11 schema (no
forked stack).
"""

from __future__ import annotations

import os

import pytest

from mfdb.repository import MFDatabase


@pytest.fixture
def db(tmp_path):
    database = MFDatabase(os.path.join(tmp_path, "protocol.db"))
    try:
        yield database
    finally:
        database.close()


def test_protocol_table_and_operation_columns_exist(db):
    tables = {r[0] for r in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "mfdb_protocol" in tables
    op_cols = {r[1] for r in db.conn.execute("PRAGMA table_info(mfdb_operation)")}
    assert {"protocol_id", "protocol_version"} <= op_cols


def test_create_protocol_returns_id_and_version(db):
    pid, version = db.create_protocol(
        "smFRET acquisition", "measurement", operation_type="measurement_import"
    )
    assert pid and version == 1
    row = db.get_protocol_by_id(pid)
    assert row["name"] == "smFRET acquisition"
    assert row["category"] == "measurement"
    assert row["operation_type"] == "measurement_import"


def test_unknown_category_rejected(db):
    with pytest.raises(ValueError):
        db.create_protocol("bad", "not_a_category")


def test_editing_creates_new_version_append_only(db):
    pid1, v1 = db.create_protocol("burst pipeline", "processing", operation_type="burst_selection")
    pid2, v2 = db.create_protocol("burst pipeline", "processing", operation_type="burst_selection",
                                  description="tightened min_photons")
    assert v1 == 1 and v2 == 2
    assert pid1 != pid2
    # the old version row is untouched (append-only)
    assert db.get_protocol("burst pipeline", version=1)["protocol_id"] == pid1
    assert db.get_protocol("burst pipeline")["protocol_id"] == pid2  # latest
    assert [p["version"] for p in db.list_protocol_versions("burst pipeline")] == [1, 2]


def test_list_protocols_returns_latest_per_name_scoped(db):
    db.ensure_user("u1")
    db.ensure_user("u2")
    db.create_protocol("A", "processing", created_by_user_id="u1")
    db.create_protocol("A", "processing", created_by_user_id="u1")  # v2 latest
    db.create_protocol("B", "analysis", is_public=True, created_by_user_id="u2")
    db.create_protocol("C", "analysis", created_by_user_id="u2")  # private, not u1

    mine = db.list_protocols(scope="own", owner_id="u1")
    assert {p["name"] for p in mine} == {"A"}
    assert mine[0]["version"] == 2  # latest only

    public = db.list_protocols(scope="public")
    assert {p["name"] for p in public} == {"B"}

    all_for_u1 = db.list_protocols(scope="all", owner_id="u1")
    assert {p["name"] for p in all_for_u1} == {"A", "B"}  # own + public, not C


def test_parameter_schema_is_operation_type_schema(db):
    # burst_selection has a declared .dic parameter schema (PRD-11)
    pid, _ = db.create_protocol("burst", "processing", operation_type="burst_selection")
    protocol = db.get_protocol_by_id(pid)
    schema = db.get_protocol_parameter_schema(protocol)
    assert "min_photons" in schema  # reuses mfdb_operation_parameter_def, no forked stack


def test_protocol_with_no_operation_type_has_empty_schema(db):
    pid, _ = db.create_protocol("freeform", "analysis")
    assert db.get_protocol_parameter_schema(db.get_protocol_by_id(pid)) == {}


# -- Increment 2: operations reference the protocol they ran -----------------


def _set_global(db):
    from mfdb.provenance.result_registry import set_global_db

    set_global_db(db)


def test_register_operation_records_protocol_ref(db, tmp_path):
    from mfdb.provenance.result_registry import register_operation, set_global_db

    pid, version = db.create_protocol(
        "shift v", "processing", operation_type="microtime_shift"
    )
    try:
        op_id = register_operation(
            operation_type="microtime_shift",
            parameters={"global_shift": 2},
            protocol_id=pid,
            db=db,
        )
    finally:
        set_global_db(None)
    row = db.conn.execute(
        "SELECT protocol_id, protocol_version FROM mfdb_operation WHERE operation_id = ?",
        (op_id,),
    ).fetchone()
    assert row[0] == pid
    # protocol_version defaults to the referenced protocol's version
    assert row[1] == version


def test_register_operation_rejects_unknown_protocol(db):
    from mfdb.provenance.result_registry import register_operation, set_global_db

    try:
        with pytest.raises(ValueError):
            register_operation(
                operation_type="microtime_shift",
                protocol_id="does-not-exist",
                db=db,
            )
    finally:
        set_global_db(None)


def test_register_operation_rejects_operation_type_mismatch(db):
    from mfdb.provenance.result_registry import register_operation, set_global_db

    pid, _ = db.create_protocol("burst proc", "processing", operation_type="burst_selection")
    try:
        with pytest.raises(ValueError):
            # protocol realizes burst_selection, but the operation is microtime_shift
            register_operation(
                operation_type="microtime_shift",
                protocol_id=pid,
                db=db,
            )
    finally:
        set_global_db(None)


def test_reproduce_run_by_protocol_and_version(db):
    """The recorded operation pins exactly which protocol version produced it."""
    from mfdb.provenance.result_registry import register_operation, set_global_db

    pid_v1, v1 = db.create_protocol("pipe", "processing", operation_type="microtime_shift")
    try:
        op = register_operation(
            operation_type="microtime_shift",
            parameters={"global_shift": 5},
            protocol_id=pid_v1,
            db=db,
        )
        # a later protocol edit (v2) does not change the recorded operation's ref
        db.create_protocol("pipe", "processing", operation_type="microtime_shift")
    finally:
        set_global_db(None)
    row = db.conn.execute(
        "SELECT protocol_id, protocol_version FROM mfdb_operation WHERE operation_id = ?",
        (op,),
    ).fetchone()
    assert (row[0], row[1]) == (pid_v1, v1)
    assert db.get_protocol("pipe")["version"] == 2  # latest advanced, the run did not
