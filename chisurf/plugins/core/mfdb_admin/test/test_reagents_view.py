"""PRD-15: ReagentLotsView Qt widget (smoke).

Drives the thin view over a real in-process MFDBClient RPC layer: list lots, filter by
kind, toggle show-expired, show the selected lot's fields, and create a lot via the form.
"""

from __future__ import annotations

import pytest

pytest.importorskip("qtpy")

from chisurf.plugins.core.mfdb_admin.gui.client import MFDBClient
from chisurf.plugins.core.mfdb_admin.gui.reagents_view import ReagentLotsView

from .conftest import patch_db


@pytest.fixture
def qapp():
    from qtpy import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _lot_names(view) -> set[str]:
    return {
        view.lot_table.item(r, 1).text() for r in range(view.lot_table.rowCount())
    }


def test_view_lists_lots(db, qapp):
    with patch_db(db):
        client = MFDBClient(inprocess=True)
        client.create_reagent_lot("fluorophore", "Alexa 488")
        client.create_reagent_lot("buffer", "PBS")
        view = ReagentLotsView(client)
        assert _lot_names(view) == {"Alexa 488", "PBS"}


def test_view_kind_filter(db, qapp):
    with patch_db(db):
        client = MFDBClient(inprocess=True)
        client.create_reagent_lot("fluorophore", "Alexa 488")
        client.create_reagent_lot("buffer", "PBS")
        view = ReagentLotsView(client)
        view.kind_filter_combo.setCurrentText("buffer")
        assert _lot_names(view) == {"PBS"}


def test_view_show_expired_toggle(db, qapp):
    with patch_db(db):
        client = MFDBClient(inprocess=True)
        client.create_reagent_lot("buffer", "fresh", expiry="2999-01-01")
        client.create_reagent_lot("buffer", "old", expiry="2000-01-01")
        view = ReagentLotsView(client)
        assert _lot_names(view) == {"fresh"}  # expired hidden by default
        view.show_expired_check.setChecked(True)
        assert _lot_names(view) == {"fresh", "old"}


def test_view_select_shows_detail(db, qapp):
    with patch_db(db):
        client = MFDBClient(inprocess=True)
        client.create_reagent_lot("fluorophore", "Cy3", lot_number="L-7", vendor="GE")
        view = ReagentLotsView(client)
        view.lot_table.selectRow(0)
        detail = {
            view.detail_table.item(r, 0).text(): view.detail_table.item(r, 1).text()
            for r in range(view.detail_table.rowCount())
        }
    assert detail["name"] == "Cy3"
    assert detail["lot_number"] == "L-7"
    assert detail["vendor"] == "GE"


def test_view_create_adds_lot(db, qapp):
    with patch_db(db):
        client = MFDBClient(inprocess=True)
        view = ReagentLotsView(client)
        view.new_kind_combo.setCurrentText("kit")
        view.new_name_edit.setText("Click-iT kit")
        view.new_lot_number_edit.setText("K-99")
        view.create_lot()
        assert "Click-iT kit" in _lot_names(view)
        assert "Created lot" in view.message_label.text()


def test_view_create_rejects_blank_name(db, qapp):
    with patch_db(db):
        client = MFDBClient(inprocess=True)
        view = ReagentLotsView(client)
        view.new_name_edit.setText("   ")
        view.create_lot()
    assert "Enter a lot name" in view.message_label.text()
    assert _lot_names(view) == set()
