"""PRD-15: lightweight reagent / consumable inventory.

Lots carry lot_number/expiry and link many-to-many to operations/setups/samples through
``mfdb_reagent_usage`` (no columns added to those tables). The "what was used" query and
expiry checks support reproducibility and QC.
"""

from __future__ import annotations

import os

import pytest

from mfdb.reagents import (
    add_reagent_lot,
    expired_lots,
    link_reagent,
    list_lots,
    list_reagents_for,
)
from mfdb.repository import MFDatabase
from mfdb.result_registry import set_global_db


@pytest.fixture
def db(tmp_path):
    database = MFDatabase(os.path.join(tmp_path, "reagents.db"))
    try:
        yield database
    finally:
        set_global_db(None)
        database.close()


def test_add_lot_and_list(db):
    lid = add_reagent_lot(
        db, kind="fluorophore", name="Alexa 488", lot_number="A488-42",
        vendor="Thermo", expiry="2030-01-01",
    )
    assert lid
    lots = list_lots(db, kind="fluorophore")
    assert len(lots) == 1
    assert lots[0]["lot_id"] == lid
    assert lots[0]["lot_number"] == "A488-42"


def test_unknown_kind_is_rejected(db):
    with pytest.raises(ValueError, match="unknown reagent kind"):
        add_reagent_lot(db, kind="nonsense", name="x")


def test_usage_many_to_many_without_schema_pollution(db):
    dye = add_reagent_lot(db, kind="fluorophore", name="Alexa 647")
    buf = add_reagent_lot(db, kind="buffer", name="PBS pH7.4")
    # one operation used both a dye and a buffer
    link_reagent(db, dye, "operation", "op-1", role="label")
    link_reagent(db, buf, "operation", "op-1", role="medium")
    # the dye was also used on a sample
    link_reagent(db, dye, "sample", "sample-9")

    used_by_op = {r["lot_id"] for r in list_reagents_for(db, "operation", "op-1")}
    assert used_by_op == {dye, buf}
    used_by_sample = [r["lot_id"] for r in list_reagents_for(db, "sample", "sample-9")]
    assert used_by_sample == [dye]
    # no columns were added to operation/sample tables — the link is orthogonal
    op_cols = {r[1] for r in db.conn.execute("PRAGMA table_info(mfdb_operation)").fetchall()}
    assert "lot_id" not in op_cols


def test_link_reagent_is_idempotent(db):
    dye = add_reagent_lot(db, kind="fluorophore", name="Cy3")
    link_reagent(db, dye, "operation", "op-7")
    link_reagent(db, dye, "operation", "op-7")
    assert len(list_reagents_for(db, "operation", "op-7")) == 1


def test_unknown_target_type_is_rejected(db):
    dye = add_reagent_lot(db, kind="fluorophore", name="Cy5")
    with pytest.raises(ValueError, match="unknown target_type"):
        link_reagent(db, dye, "instrument", "x")


def test_expiry_query_and_default_exclusion(db):
    fresh = add_reagent_lot(db, kind="buffer", name="fresh", expiry="2999-01-01")
    old = add_reagent_lot(db, kind="buffer", name="old", expiry="2000-01-01")
    no_expiry = add_reagent_lot(db, kind="buffer", name="stable")

    expired = {r["lot_id"] for r in expired_lots(db)}
    assert expired == {old}

    # list_lots excludes expired by default, keeps fresh + no-expiry
    visible = {r["lot_id"] for r in list_lots(db, kind="buffer")}
    assert visible == {fresh, no_expiry}
    # include_expired returns all three
    assert len(list_lots(db, kind="buffer", include_expired=True)) == 3
    # as_of lets you ask "what was expired at a past date"
    assert expired_lots(db, as_of="1999-01-01") == []
