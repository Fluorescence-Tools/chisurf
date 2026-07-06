"""PRD-14 Increment 3b: the ProtocolsView Qt widget (smoke).

Drives the thin view against the real InProcessClient RPC layer: create protocols,
list them (latest per name), show version history + the operation_type parameter schema,
and surface an invalid category.
"""

from __future__ import annotations

import pytest

pytest.importorskip("qtpy")

from mfdb.admin.gui.client import MFDBClient
from mfdb.admin.gui.protocols_view import ProtocolsView

from .conftest import patch_db


@pytest.fixture
def qapp():
    from qtpy import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def test_view_lists_latest_protocols(db, qapp):
    with patch_db(db):
        client = MFDBClient(inprocess=True)
        client.create_protocol("acq", "measurement", operation_type="measurement_import")
        client.create_protocol("acq", "measurement", operation_type="measurement_import")
        view = ProtocolsView(client)
    assert view.protocol_table.rowCount() == 1  # latest per name
    assert view.protocol_table.item(0, 0).text() == "acq"
    assert view.protocol_table.item(0, 1).text() == "2"


def test_view_shows_versions_and_schema_on_select(db, qapp):
    with patch_db(db):
        client = MFDBClient(inprocess=True)
        client.create_protocol("burst", "processing", operation_type="burst_selection")
        client.create_protocol("burst", "processing", operation_type="burst_selection")
        view = ProtocolsView(client)
        view.protocol_table.selectRow(0)
    assert view.version_table.rowCount() == 2
    schema_names = {
        view.schema_table.item(r, 0).text() for r in range(view.schema_table.rowCount())
    }
    assert "min_photons" in schema_names


def test_view_create_adds_protocol(db, qapp):
    with patch_db(db):
        client = MFDBClient(inprocess=True)
        view = ProtocolsView(client)
        view.new_name_edit.setText("new_proto")
        view.new_category_combo.setCurrentText("analysis")
        view.create_protocol()
    assert "Created new_proto v1" in view.message_label.text()
    names = {
        view.protocol_table.item(r, 0).text() for r in range(view.protocol_table.rowCount())
    }
    assert "new_proto" in names
