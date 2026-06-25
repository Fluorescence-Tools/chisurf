"""PRD-05: CalibrationsView Qt widget (smoke).

Drives the thin view over a real in-process MFDBClient: list calibrations, register a
literature value via the form, and show stale uses.
"""

from __future__ import annotations

import pytest

pytest.importorskip("qtpy")

from chisurf.core.mfdb.staleness import record_calibration_use
from chisurf.plugins.core.mfdb_admin.gui.calibrations_view import CalibrationsView
from chisurf.plugins.core.mfdb_admin.gui.client import MFDBClient

from .conftest import patch_db


@pytest.fixture
def qapp():
    from qtpy import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _cal_types(view) -> set[str]:
    return {
        view.cal_table.item(r, 0).text() for r in range(view.cal_table.rowCount())
    }


def test_view_lists_calibrations(db, qapp):
    with patch_db(db):
        client = MFDBClient(inprocess=True)
        client.create_calibration("g_factor", 1.02)
        client.create_calibration("gamma", 0.9)
        view = CalibrationsView(client)
        assert _cal_types(view) == {"g_factor", "gamma"}


def test_view_create_adds_calibration(db, qapp):
    with patch_db(db):
        client = MFDBClient(inprocess=True)
        view = CalibrationsView(client)
        view.new_type_combo.setCurrentText("forster_radius")
        view.new_value_edit.setText("54.0")
        view.new_notes_edit.setText("Hellenkamp 2018")
        view.create_calibration()
        assert "forster_radius" in _cal_types(view)
        assert "Registered" in view.message_label.text()


def test_view_rejects_non_numeric_value(db, qapp):
    with patch_db(db):
        client = MFDBClient(inprocess=True)
        view = CalibrationsView(client)
        view.new_value_edit.setText("not-a-number")
        view.create_calibration()
    assert "numeric" in view.message_label.text()
    assert _cal_types(view) == set()


def test_view_shows_stale_uses(db, qapp):
    with patch_db(db):
        client = MFDBClient(inprocess=True)
        old = client.create_calibration("g_factor", 1.02)["artifact_id"]
        record_calibration_use(db, used_by_id="fit-1", calibration_artifact_id=old,
                               used_by_type="artifact")
        client.create_calibration("g_factor", 1.05)  # supersedes
        view = CalibrationsView(client)
        assert view.stale_table.rowCount() == 1
        assert view.stale_table.item(0, 0).text() == "fit-1"
        assert view.stale_table.item(0, 1).text() == "g_factor"
