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
    db.close()
