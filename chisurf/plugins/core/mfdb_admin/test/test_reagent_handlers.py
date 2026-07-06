"""PRD-15: mfdb-admin reagent RPC handlers.

Reagent lots are reachable through the admin backend (and thus MFDBClient): list/create
+ expiry + usage links. A bad kind / target_type comes back as an ``error`` field, not
an exception.
"""

from __future__ import annotations

from mfdb.admin.backend.services import (
    add_reagent_usage_handler,
    create_reagent_lot_handler,
    expired_reagent_lots_handler,
    list_reagent_lots_handler,
    list_reagent_usage_handler,
)

from .conftest import patch_db


def test_create_and_list_lot(db):
    with patch_db(db):
        lot_id = create_reagent_lot_handler(
            kind="fluorophore", name="Alexa 488", lot_number="A488-42", vendor="Thermo"
        )["lot_id"]
        lots = list_reagent_lots_handler(kind="fluorophore")["lots"]
    assert any(l["lot_id"] == lot_id and l["lot_number"] == "A488-42" for l in lots)


def test_bad_kind_returns_error(db):
    with patch_db(db):
        res = create_reagent_lot_handler(kind="nonsense", name="x")
    assert "error" in res


def test_expiry_filter_and_expired_handler(db):
    with patch_db(db):
        create_reagent_lot_handler(kind="buffer", name="fresh", expiry="2999-01-01")
        create_reagent_lot_handler(kind="buffer", name="old", expiry="2000-01-01")
        visible = {l["name"] for l in list_reagent_lots_handler(kind="buffer")["lots"]}
        all_lots = {l["name"] for l in list_reagent_lots_handler(
            kind="buffer", include_expired=True)["lots"]}
        expired = {l["name"] for l in expired_reagent_lots_handler()["lots"]}
    assert visible == {"fresh"}
    assert all_lots == {"fresh", "old"}
    assert expired == {"old"}


def test_usage_link_and_list(db):
    with patch_db(db):
        lot_id = create_reagent_lot_handler(kind="fluorophore", name="Cy3")["lot_id"]
        add_reagent_usage_handler(lot_id, "operation", "op-1", role="label")
        used = list_reagent_usage_handler("operation", "op-1")["lots"]
    assert [l["lot_id"] for l in used] == [lot_id]


def test_bad_target_type_returns_error(db):
    with patch_db(db):
        lot_id = create_reagent_lot_handler(kind="fluorophore", name="Cy5")["lot_id"]
        res = add_reagent_usage_handler(lot_id, "instrument", "x")
    assert "error" in res


def test_via_inprocess_client(db):
    with patch_db(db):
        from mfdb.admin.gui.client import MFDBClient

        client = MFDBClient(inprocess=True)
        lot_id = client.create_reagent_lot("buffer", "PBS", lot_number="L1")["lot_id"]
        client.link_reagent(lot_id, "sample", "s1")
        lots = client.list_reagent_lots(kind="buffer")
        used = client.list_reagents_for("sample", "s1")
    assert any(l["lot_id"] == lot_id for l in lots)
    assert used[0]["lot_id"] == lot_id
