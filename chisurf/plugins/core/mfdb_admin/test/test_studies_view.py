"""PRD-13 Increment 3b: the StudiesView Qt widget (smoke).

Drives the thin view against the real InProcessClient RPC layer: create studies, list
them, and show members + fields on selection.
"""

from __future__ import annotations

import pytest

pytest.importorskip("qtpy")

from chisurf.plugins.core.mfdb_admin.gui.client import MFDBClient
from chisurf.plugins.core.mfdb_admin.gui.studies_view import StudiesView

from .conftest import patch_db


@pytest.fixture
def qapp():
    from qtpy import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def test_view_lists_studies(db, qapp):
    with patch_db(db):
        client = MFDBClient(inprocess=True)
        client.create_study("Alpha")
        client.create_study("Beta")
        view = StudiesView(client)
    names = {view.study_table.item(r, 0).text() for r in range(view.study_table.rowCount())}
    assert {"Alpha", "Beta"} <= names


def test_view_shows_members_and_fields_on_select(db, qapp):
    with patch_db(db):
        client = MFDBClient(inprocess=True)
        sid = client.create_study("Gamma")["study_id"]
        client.add_study_member(sid, "sample", "s1")
        client.set_study_field(sid, "grant", "NIH-9")
        view = StudiesView(client)
        # select the Gamma row
        for r in range(view.study_table.rowCount()):
            if view.study_table.item(r, 2).text() == sid:
                view.study_table.selectRow(r)
                break
    assert view.member_table.rowCount() == 1
    assert view.member_table.item(0, 1).text() == "s1"
    assert view.field_table.rowCount() == 1
    assert view.field_table.item(0, 0).text() == "grant"


def test_view_create_adds_study(db, qapp):
    with patch_db(db):
        client = MFDBClient(inprocess=True)
        view = StudiesView(client)
        view.new_name_edit.setText("Delta")
        view.create_study()
    names = {view.study_table.item(r, 0).text() for r in range(view.study_table.rowCount())}
    assert "Delta" in names
    assert "Created study Delta" in view.message_label.text()
