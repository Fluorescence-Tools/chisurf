"""Headless test for the spectra-downloader staging data browser."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")


@pytest.fixture
def qapp():
    from qtpy import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _staging_db():
    from chisurf.plugins._dev.fluorophore_db.mfdb_adapter import FluorophoreDatabase

    path = os.path.join(tempfile.mkdtemp(), "staging.db")
    db = FluorophoreDatabase(path)
    db.connect()
    wl = np.linspace(400, 700, 8)
    with db:
        db.register_component(name="EGFP", source="fpbase", kind="fluorescent_protein",
                              spectra={"absorption": (wl, np.ones_like(wl)),
                                       "emission": (wl, np.ones_like(wl))})
        db.register_component(name="Thorlabs SPCMxxA", source="fpbase", kind="detector",
                              source_ref="9342",
                              spectra={"quantum_efficiency": (wl, np.linspace(0, 1, 8))})
        db.conn.commit()
    return db


def test_browser_lists_and_shows_detail_on_selection(qapp):
    from chisurf.plugins.spectra_downloader.browser import SpectraBrowserWidget

    db = _staging_db()
    w = SpectraBrowserWidget(db)
    # both components listed; category filter populated
    assert len(w._rows) == 2
    cats = {w._category.itemText(i) for i in range(w._category.count())}
    assert {"detector", "protein"} <= cats

    # filter to the detector and select it → detail + properties + spectrum update
    w._search.setText("SPCM")
    qapp.processEvents()
    assert w._table.rowCount() == 1
    w._table.selectRow(0)
    qapp.processEvents()
    assert getattr(w._detail._model, "chromophore_name") == "Thorlabs SPCMxxA"
    assert getattr(w._detail._model, "category") == "detector"
    # the raw optical-properties table got the granular component_kind
    prop_names = {w._props.item(i, 0).text() for i in range(w._props.rowCount())}
    assert "component_kind" in prop_names

    # category filter narrows the list
    w._search.setText("")
    w._category.setCurrentText("protein")
    qapp.processEvents()
    assert w._table.rowCount() == 1

    # JSON metadata view is populated for the selection
    w._category.setCurrentText("All")
    w._search.setText("SPCM")
    qapp.processEvents()
    w._table.selectRow(0)
    qapp.processEvents()
    import json
    meta = json.loads(w._metadata.toPlainText())
    assert meta["probe"]["chromophore_name"] == "Thorlabs SPCMxxA"
    assert any(s["type"] == "quantum_efficiency" for s in meta["spectra"])
    db.close()


def test_spectra_tool_navigation_panels_build(qapp):
    """The Spectra tool is a NavigationPanelTool whose panels all build."""
    from qtpy import QtWidgets

    from chisurf.gui.widgets.navigation import NavigationPanelTool
    from chisurf.plugins.spectra_downloader.gui.tool import SpectraTool

    db = _staging_db()
    tool = SpectraTool(db)
    assert isinstance(tool, NavigationPanelTool)
    names = [p.get("name") for p in tool.panels]
    assert names == ["Overview", "Browse", "Download", "Add to MFDB"]
    for i in range(len(tool.panels)):
        tool.nav_list.setCurrentRow(i)
        qapp.processEvents()
        assert isinstance(tool.panels[i].get("instance"), QtWidgets.QWidget)
    db.close()


def test_add_to_mfdb_panel_local(qapp):
    import tempfile, sqlite3

    from chisurf.plugins.spectra_downloader.gui.add_to_mfdb_panel import AddToMfdbPanel

    db = _staging_db()  # EGFP (protein) + SPCMxxA (detector)
    panel = AddToMfdbPanel(db)
    # AutoForm-backed endpoint/auth model with sane local defaults
    assert panel._model.mode == "local"
    assert panel._model.cmd_port == 8765
    # add into a fresh local MFDB
    mfdb = tempfile.mktemp(suffix=".mfdb")
    panel._model.db_path = mfdb
    panel._add_all()
    qapp.processEvents()
    cats = {
        r[0]
        for r in sqlite3.connect(mfdb).execute(
            "SELECT category FROM probes WHERE deleted_at IS NULL"
        )
    }
    assert {"protein", "detector"} <= cats
    db.close()


def test_add_to_mfdb_session_admin_gate(qapp):
    """Admins add without a login; non-admins are refused (session-first)."""
    import tempfile, sqlite3

    from chisurf.core.mfdb.repository import MFDatabase
    from chisurf.plugins.spectra_downloader.gui.add_to_mfdb_panel import AddToMfdbPanel

    db = _staging_db()
    panel = AddToMfdbPanel(db)

    # target MFDB with an admin user_default and a non-admin guest
    mfdb = tempfile.mktemp(suffix=".mfdb")
    with MFDatabase(mfdb) as d:
        d.add_user("user_default", "Default User", is_admin=1)
        d.add_user("guest", "Guest", is_admin=0)
    panel._model.db_path = mfdb
    panel._model.password = ""  # no password — rely on the session

    # admin → add succeeds with no login
    panel._model.user = "user_default"
    panel._add_all()
    qapp.processEvents()
    assert sqlite3.connect(mfdb).execute(
        "SELECT COUNT(*) FROM probes WHERE deleted_at IS NULL"
    ).fetchone()[0] >= 2

    # non-admin → refused
    panel._model.user = "guest"
    panel._add_all()
    qapp.processEvents()
    assert "not an administrator" in panel._log.toPlainText()
    db.close()


def test_autoform_password_kind_is_masked(qapp):
    """The password ValueSection renders a masked QLineEdit."""
    from qtpy import QtWidgets

    from chisurf.core.dataspec import ValueSection, ModelView
    from chisurf.gui.autoform import AutoForm

    class _M:
        secret = "hunter2"

        def view_spec(self):
            return ModelView(sections=[ValueSection(attr="secret", label="Secret", kind="password")])

    form = AutoForm(_M())
    edits = form.findChildren(QtWidgets.QLineEdit)
    assert any(e.echoMode() == QtWidgets.QLineEdit.Password for e in edits)


def test_overview_panel_counts(qapp):
    from chisurf.plugins.spectra_downloader.gui.overview_panel import OverviewPanel

    db = _staging_db()  # EGFP (protein) + SPCMxxA (detector)
    panel = OverviewPanel(db)
    import json
    blob = json.loads(panel._json.toPlainText())
    assert blob["by_category"].get("protein") == 1
    assert blob["by_category"].get("detector") == 1
    assert getattr(panel._form._model, "detectors") == "1"
    db.close()


def test_source_filter_and_push(qapp):
    import tempfile

    from chisurf.plugins.spectra_downloader.browser import SpectraBrowserWidget
    from chisurf.plugins.spectra_downloader.download.merge import push_staging_to_mfdb

    db = _staging_db()  # EGFP (fpbase), SPCMxxA (fpbase) — both source fpbase
    # add a second source
    wl = np.linspace(400, 700, 8)
    with db:
        db.register_component(name="FB340-10", source="thorlabs", kind="bandpass",
                              spectra={"transmission": (wl, np.ones_like(wl))})
        db.conn.commit()

    w = SpectraBrowserWidget(db, initial_source="thorlabs")
    qapp.processEvents()
    assert w._source.currentText() == "thorlabs"
    assert w._table.rowCount() == 1  # only the thorlabs component

    # push the selected (thorlabs) component into a fresh MFDB
    w._table.selectRow(0)
    qapp.processEvents()
    mfdb = tempfile.mktemp(suffix=".mfdb")
    summary = push_staging_to_mfdb(str(db.db_path), probe_ids=w._selected_probe_ids(), mfdb_path=mfdb)
    assert summary["merged"] == 1

    import sqlite3
    rows = sqlite3.connect(mfdb).execute(
        "SELECT chromophore_name, category, source FROM probes WHERE deleted_at IS NULL"
    ).fetchall()
    assert rows == [("FB340-10", "filter", "thorlabs")]

    # push all → the other two components arrive too
    push_staging_to_mfdb(str(db.db_path), mfdb_path=mfdb)
    total = sqlite3.connect(mfdb).execute(
        "SELECT COUNT(*) FROM probes WHERE deleted_at IS NULL"
    ).fetchone()[0]
    assert total == 3
    db.close()
