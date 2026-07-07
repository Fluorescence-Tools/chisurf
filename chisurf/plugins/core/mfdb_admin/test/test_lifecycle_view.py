"""PRD-12 Increment 4b: the LifecycleView Qt widget.

Smoke-tests the thin view against the real InProcessClient RPC layer: load definitions,
display state + history, drive a transition, and surface an illegal jump — all without
the full mfdb-admin tool.
"""

from __future__ import annotations

import pytest

pytest.importorskip("qtpy")

from chisurf.plugins.core.mfdb_admin.gui.client import MFDBClient
from chisurf.plugins.core.mfdb_admin.gui.lifecycle_view import LifecycleView

from .conftest import patch_db


@pytest.fixture
def qapp():
    from qtpy import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def test_view_constructs_and_loads_definitions(db, qapp):
    with patch_db(db):
        view = LifecycleView(MFDBClient(inprocess=True))
        types = {view.entity_type_combo.itemText(i) for i in range(view.entity_type_combo.count())}
    assert {"sample", "artifact", "operation"} <= types


def test_view_refresh_shows_state_and_next_states(db, qapp):
    with patch_db(db):
        client = MFDBClient(inprocess=True)
        client.lifecycle_transition("sample", "s1", "registered")
        view = LifecycleView(client)
        view.set_entity("sample", "s1")
        view.refresh()
        next_states = {view.to_state_combo.itemText(i) for i in range(view.to_state_combo.count())}
    assert "Current state: registered" == view.state_label.text()
    # registered -> measured is the legal next step
    assert "measured" in next_states
    assert view.history_table.rowCount() == 1


def test_view_apply_transition_advances_state(db, qapp):
    with patch_db(db):
        client = MFDBClient(inprocess=True)
        client.lifecycle_transition("sample", "s2", "registered")
        view = LifecycleView(client)
        view.set_entity("sample", "s2")
        view.refresh()
        # the combo now offers "measured"; select and apply
        idx = view.to_state_combo.findText("measured")
        view.to_state_combo.setCurrentIndex(idx)
        view.apply_transition()
    assert view.state_label.text() == "Current state: measured"
    assert view.history_table.rowCount() == 2


def test_view_surfaces_illegal_transition(db, qapp):
    with patch_db(db):
        client = MFDBClient(inprocess=True)
        client.lifecycle_transition("sample", "s3", "registered")
        view = LifecycleView(client)
        view.set_entity("sample", "s3")
        view.refresh()
        # force an illegal target that the combo would not normally offer
        view.to_state_combo.addItem("archived")
        view.to_state_combo.setCurrentText("archived")
        view.apply_transition()
    assert "Rejected" in view.message_label.text()
    assert view.state_label.text() == "Current state: registered"
