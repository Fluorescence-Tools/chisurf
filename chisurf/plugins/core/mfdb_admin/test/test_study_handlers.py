"""PRD-13 Increment 3: mfdb-admin study RPC handlers.

Studies are reachable through the admin backend (and thus MFDBClient): list/get/create
+ membership + configurable fields. A bad member_type / missing name comes back as an
``error`` field, not an exception.
"""

from __future__ import annotations

from mfdb.admin.backend.services import (
    add_study_member_handler,
    create_study_handler,
    get_study_handler,
    list_studies_handler,
    set_study_field_handler,
)

from .conftest import patch_db


def test_create_list_get_study(db):
    with patch_db(db):
        sid = create_study_handler("Screen A", "desc")["study_id"]
        studies = list_studies_handler(scope="all")["studies"]
        got = get_study_handler(sid)
    assert any(s["study_id"] == sid for s in studies)
    assert got["study"]["name"] == "Screen A"
    assert got["members"] == []
    assert got["fields"] == {}


def test_missing_name_returns_error(db):
    with patch_db(db):
        res = create_study_handler("")
    assert "error" in res


def test_membership_and_fields(db):
    with patch_db(db):
        sid = create_study_handler("S")["study_id"]
        add_study_member_handler(sid, "sample", "samp1")
        add_study_member_handler(sid, "artifact", "art1")
        set_study_field_handler(sid, "grant", "NIH-1")
        got = get_study_handler(sid)
    member_keys = {(m["member_type"], m["member_id"]) for m in got["members"]}
    assert member_keys == {("sample", "samp1"), ("artifact", "art1")}
    assert got["fields"] == {"grant": "NIH-1"}


def test_bad_member_type_returns_error(db):
    with patch_db(db):
        sid = create_study_handler("S")["study_id"]
        res = add_study_member_handler(sid, "not_a_type", "x")
    assert "error" in res


def test_via_inprocess_client(db):
    with patch_db(db):
        from mfdb.admin.gui.client import MFDBClient

        client = MFDBClient(inprocess=True)
        sid = client.create_study("C")["study_id"]
        client.add_study_member(sid, "sample", "s1")
        got = client.get_study(sid)
    assert got["study"]["name"] == "C"
    assert got["members"][0]["member_id"] == "s1"
