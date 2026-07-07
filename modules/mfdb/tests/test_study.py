"""PRD-13 Increment 1: study/project entity — schema + CRUD + membership + fields.

`mfdb_study` / `mfdb_study_member` / `mfdb_study_key_value` are `.dic`-declared (created
by reconcile_schema). Studies are user-scoped (own + public); samples/artifacts are
many-to-many members; per-study configurable fields reuse the key-value pattern.
"""

from __future__ import annotations

import os

import pytest

from mfdb.repository import MFDatabase


@pytest.fixture
def db(tmp_path):
    database = MFDatabase(os.path.join(tmp_path, "study.db"))
    try:
        yield database
    finally:
        database.close()


def test_study_tables_exist(db):
    tables = {r[0] for r in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"mfdb_study", "mfdb_study_member", "mfdb_study_key_value"} <= tables


def test_create_and_get_study(db):
    sid = db.create_study("FRET screen", "DNA hairpins", created_by_user_id=None)
    row = db.get_study(sid)
    assert row["name"] == "FRET screen"
    assert row["description"] == "DNA hairpins"


def test_list_studies_scoped(db):
    db.ensure_user("u1")
    db.ensure_user("u2")
    db.create_study("A", created_by_user_id="u1")
    db.create_study("B", is_public=True, created_by_user_id="u2")
    db.create_study("C", created_by_user_id="u2")  # private, not u1

    assert {s["name"] for s in db.list_studies(scope="mine", owner_id="u1")} == {"A"}
    assert {s["name"] for s in db.list_studies(scope="public")} == {"B"}
    # own + public, not another user's private C
    assert {s["name"] for s in db.list_studies(scope="all", owner_id="u1")} == {"A", "B"}


def test_membership_is_many_to_many_and_idempotent(db):
    s1 = db.create_study("S1")
    s2 = db.create_study("S2")
    db.add_study_member(s1, "sample", "samp_1")
    db.add_study_member(s1, "artifact", "art_1")
    db.add_study_member(s2, "sample", "samp_1")  # same sample, second study
    db.add_study_member(s1, "sample", "samp_1")  # duplicate -> no-op

    members = db.list_study_members(s1)
    assert {(m["member_type"], m["member_id"]) for m in members} == {
        ("sample", "samp_1"), ("artifact", "art_1")
    }
    # a sample can belong to several studies
    assert set(db.list_studies_for_member("sample", "samp_1")) == {s1, s2}


def test_unknown_member_type_rejected(db):
    s = db.create_study("S")
    with pytest.raises(ValueError):
        db.add_study_member(s, "not_a_type", "x")


def test_configurable_fields_upsert(db):
    s = db.create_study("S")
    db.set_study_field(s, "grant", "NIH-123")
    db.set_study_field(s, "stage", "pilot")
    db.set_study_field(s, "stage", "production")  # upsert
    assert db.get_study_fields(s) == {"grant": "NIH-123", "stage": "production"}


def test_browse_datasets_filters_by_study(db, tmp_path):
    """browse_datasets gains a study facet: direct-member artifacts and artifacts
    whose linked sample is a member."""
    from mfdb.result_registry import (
        register_raw_measurement,
        register_result,
        set_global_db,
    )

    db.add_sample("samp_s")
    study = db.create_study("MyStudy")
    other = db.create_study("Other")
    f = tmp_path / "m.ptu"
    f.write_bytes(b"\x00\x01\x02")
    try:
        # artifact A: linked to a sample that is a study member
        a_sample = register_raw_measurement(str(f), sample_id="samp_s", db=db)
        # artifact B: a direct artifact member
        b_direct = register_result(
            kind="processed_data", data={"x": [1]}, operation_type="microtime_shift",
            parameters={"global_shift": 1}, db=db,
        )
        # artifact C: in neither study
        c_out = register_result(
            kind="processed_data", data={"x": [2]}, operation_type="microtime_shift",
            parameters={"global_shift": 2}, db=db,
        )
    finally:
        set_global_db(None)

    db.add_study_member(study, "sample", "samp_s")   # pulls in A
    db.add_study_member(study, "artifact", b_direct)  # pulls in B
    db.add_study_member(other, "artifact", c_out)

    from mfdb.session import configured_default_user_id

    owner = configured_default_user_id()
    ids = {
        d["artifact_id"]
        for d in db.browse_datasets(scope="own", owner_id=owner, study_id=study)["datasets"]
    }
    assert a_sample in ids
    assert b_direct in ids
    assert c_out not in ids
    # without the facet all three are visible to the owner
    all_ids = {
        d["artifact_id"]
        for d in db.browse_datasets(scope="own", owner_id=owner)["datasets"]
    }
    assert {a_sample, b_direct, c_out} <= all_ids


def test_backfill_studies_from_project_ids(db):
    db.add_sample("s1", num_of_probes=1)
    db.add_sample("s2", num_of_probes=1)
    db.add_sample("s3", num_of_probes=1)
    db.conn.execute("UPDATE flr_sample SET project_id = 'projA' WHERE sample_id IN ('s1','s2')")
    db.conn.execute("UPDATE flr_sample SET project_id = 'projB' WHERE sample_id = 's3'")
    db.conn.commit()

    report = db.backfill_studies_from_project_ids()
    assert report["studies_created"] == 2
    assert report["members_added"] == 3

    studies = {s["name"]: s["study_id"] for s in db.list_studies(scope="all")}
    assert {"projA", "projB"} <= set(studies)
    members_a = {m["member_id"] for m in db.list_study_members(studies["projA"])}
    assert members_a == {"s1", "s2"}

    # idempotent: a second run creates nothing new
    again = db.backfill_studies_from_project_ids()
    assert again == {"studies_created": 0, "members_added": 0}
