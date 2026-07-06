"""PRD-14 Increment 3: mfdb-admin protocol RPC handlers.

Protocols are reachable through the admin backend (and thus MFDBClient) so the GUI
view is a thin caller: list/get/versions/create + the per-operation protocol provenance.
An invalid category comes back as an ``error`` field, not an exception.
"""

from __future__ import annotations

from mfdb.admin.backend.services import (
    create_protocol_handler,
    get_protocol_handler,
    list_protocol_versions_handler,
    list_protocols_handler,
    protocol_for_operation_handler,
)

from .conftest import patch_db


def test_create_and_get_protocol(db):
    with patch_db(db):
        res = create_protocol_handler("acq", "measurement", operation_type="measurement_import")
        assert res["version"] == 1
        got = get_protocol_handler("acq")
        assert got["protocol"]["name"] == "acq"
        assert got["protocol"]["category"] == "measurement"


def test_create_invalid_category_returns_error(db):
    with patch_db(db):
        res = create_protocol_handler("bad", "nope")
    assert "error" in res


def test_versions_and_list(db):
    with patch_db(db):
        create_protocol_handler("p", "processing", operation_type="burst_selection")
        create_protocol_handler("p", "processing", operation_type="burst_selection")
        versions = list_protocol_versions_handler("p")["versions"]
        protocols = list_protocols_handler(scope="all")["protocols"]
    assert [v["version"] for v in versions] == [1, 2]
    # list returns the latest per name
    assert [p for p in protocols if p["name"] == "p"][0]["version"] == 2


def test_get_protocol_exposes_parameter_schema(db):
    with patch_db(db):
        create_protocol_handler("burst", "processing", operation_type="burst_selection")
        got = get_protocol_handler("burst")
    names = {p["name"] for p in got["parameter_schema"]}
    assert "min_photons" in names  # the operation_type schema (PRD-11), no forked stack


def test_protocol_for_operation_provenance(db):
    from mfdb.result_registry import register_operation, set_global_db

    with patch_db(db):
        res = create_protocol_handler("shift", "processing", operation_type="microtime_shift")
        # create_protocol_handler opened its own db; fetch the id via get
        got = get_protocol_handler("shift")
        pid = got["protocol"]["protocol_id"]
        try:
            op = register_operation(
                operation_type="microtime_shift",
                parameters={"global_shift": 1},
                protocol_id=pid,
                db=db,
            )
        finally:
            set_global_db(None)
        prov = protocol_for_operation_handler(op)
    assert prov["protocol"]["protocol_id"] == pid
    assert prov["protocol_version"] == res["version"]


def test_via_inprocess_client(db):
    with patch_db(db):
        from mfdb.admin.gui.client import MFDBClient

        client = MFDBClient(inprocess=True)
        out = client.create_protocol("c", "analysis")
        assert out["version"] == 1
        assert any(p["name"] == "c" for p in client.list_protocols())
