"""mfdb-admin uses the shared NavigationPanelTool (left-nav / right-panel) shell.

Drives the full MFDBWidget against the real in-process RPC layer and an empty
temp DB: it must be a NavigationPanelTool, expose the flattened nav panels, and
build every panel (entity docks + the workflow views) without raising.
"""
from __future__ import annotations

import os
from unittest import mock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

from chisurf.gui.widgets.navigation import NavigationPanelTool
from chisurf.plugins.core.mfdb_admin.gui.client import MFDBClient

from .conftest import patch_db


@pytest.fixture
def qapp():
    from qtpy import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _make_widget(db):
    from chisurf.plugins.core.mfdb_admin.gui.tool import MFDBWidget

    with patch_db(db):
        client = MFDBClient(inprocess=True)
        with mock.patch.object(MFDBWidget, "_verify_admin_access", lambda s: None), \
             mock.patch.object(MFDBWidget, "_ensure_authenticated", lambda s: None):
            return MFDBWidget(client=client)


def test_widget_is_navigation_panel_tool(db, qapp):
    w = _make_widget(db)
    assert isinstance(w, NavigationPanelTool)
    # left nav + right stack from the shell
    assert hasattr(w, "nav_list") and hasattr(w, "stacked_widget")


def test_panels_flattened_with_separators_and_views(db, qapp):
    w = _make_widget(db)
    names = [p.get("name") for p in w.panels]
    # aggregate panels
    for n in ("Overview", "All items", "Measurements"):
        assert n in names
    # separator group headers
    assert sum(1 for p in w.panels if p.get("separator")) >= 4
    # entity panels are flattened in
    assert sum(1 for p in w.panels if p.get("entity_key")) >= 15
    # the previously-orphaned workflow views are now reachable
    for n in ("Studies", "Protocols", "Lifecycle", "Calibrations", "Reagent Lots", "Pipelines"):
        assert n in names


def test_every_panel_builds(db, qapp):
    from qtpy import QtWidgets

    w = _make_widget(db)
    for i, panel in enumerate(w.panels):
        if panel.get("separator"):
            continue
        w.nav_list.setCurrentRow(i)
        assert isinstance(panel.get("instance"), QtWidgets.QWidget), panel.get("name")


def test_entity_dock_uses_autoform(db, qapp):
    from chisurf.plugins.core.mfdb_admin.gui.autoform_entity_form import EntityForm

    w = _make_widget(db)
    w.nav_list.setCurrentRow(w._row_by_entity["sample"])
    dock = w._entity_docks.get("sample")
    assert dock is not None
    assert isinstance(dock._form, EntityForm)
