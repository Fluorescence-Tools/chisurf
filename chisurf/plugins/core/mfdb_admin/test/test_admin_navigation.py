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
from mfdb.admin.gui.client import MFDBClient

from .conftest import patch_db


@pytest.fixture
def qapp():
    from qtpy import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


# Keep constructed widgets alive for the whole module: each MFDBWidget builds
# docks that schedule QTimer.singleShot(0, refresh); if a widget is GC'd while a
# timer is pending, the callback hits a deleted C++ object. Holding references
# avoids that cross-test flake (the widgets are torn down at interpreter exit).
_WIDGETS: list = []


def _make_widget(db):
    from mfdb.admin.gui.tool import MFDBWidget

    with patch_db(db):
        client = MFDBClient(inprocess=True)
        with mock.patch.object(MFDBWidget, "_verify_admin_access", lambda s: None), \
             mock.patch.object(MFDBWidget, "_ensure_authenticated", lambda s: None):
            w = MFDBWidget(client=client)
    _WIDGETS.append(w)
    return w


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
    # the spectra/optical component curation view is integrated from the optical_components module
    assert "Spectra" in names
    # nav items carry emoji icons
    assert any((p.get("icon") or "") for p in w.panels if p.get("entity_key"))


def test_fluorophore_panel_and_rpc_integrated(db, qapp):
    from mfdb.admin.gui.optical_components import OpticalComponentDock

    w = _make_widget(db)
    row = w._row_by_name["Spectra"]
    w.nav_list.setCurrentRow(row)
    inst = w._unwrap(w.panels[row]["instance"])
    assert isinstance(inst, OpticalComponentDock)
    # fluorophores.* RPC handlers are registered with the admin dispatcher
    res = w.client._call("fluorophores.list", {"limit": 1})
    assert "probes" in res and "total" in res


def test_every_panel_builds(db, qapp):
    from qtpy import QtWidgets

    w = _make_widget(db)
    for i, panel in enumerate(w.panels):
        if panel.get("separator"):
            continue
        w.nav_list.setCurrentRow(i)
        assert isinstance(panel.get("instance"), QtWidgets.QWidget), panel.get("name")


def test_entity_dock_uses_autoform(db, qapp):
    from mfdb.admin.gui.autoform_entity_form import EntityForm

    w = _make_widget(db)
    w.nav_list.setCurrentRow(w._row_by_entity["sample"])
    dock = w._entity_docks.get("sample")
    assert dock is not None
    assert isinstance(dock._form, EntityForm)
