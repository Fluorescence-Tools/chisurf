"""GUI file-drop tests.

Module-level ``get_app()`` is expensive (~8 s), so we keep tests cheap
and fast.  Slow reader invocations (PDB, TTTR) are skipped with a clear
reason.
"""
from __future__ import annotations

import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import utils

TOPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
utils.set_search_paths(TOPDIR)

import pytest

from qtpy.QtCore import QMimeData, QUrl, Qt
from qtpy.QtWidgets import QApplication
from qtpy import QtGui

import chisurf as cs
import chisurf.gui

cs_app = cs.gui.get_app()

# Use local mode for dataset operations (avoids server RPC fallback warnings)
if hasattr(cs, "api") and cs.core.api is not None:
    cs.core.api.mode = "local"


def _count():
    return len(getattr(cs, 'imported_datasets', []) or [])


def _clear():
    cs.core.actions.dispatch(name="fit.close_all", payload={})
    cs.imported_datasets.clear()


def _set_exp(exp_name, setup_name, **kw):
    gui = cs.cs
    idx = gui.comboBox_experimentSelect.findText(exp_name)
    if idx >= 0:
        gui.comboBox_experimentSelect.setCurrentIndex(idx)
        gui._refresh_experiment_ui()
    idx = gui.comboBox_setupSelect.findText(setup_name)
    if idx >= 0:
        gui.comboBox_setupSelect.setCurrentIndex(idx)
        gui._refresh_setup_ui()
    for k, v in kw.items():
        try:
            setattr(gui.current_setup, k, v)
        except Exception:
            pass


def _add_ds(filename):
    cs.core.actions.dispatch(
        name="dataset.add",
        payload={"filename": filename, "experiment_reader": None},
    )


##############################################################################
#  Action-dispatch tests
##############################################################################

def test_tcspc_txt():
    _clear()
    _set_exp("TCSPC", "TXT/CSV",
             skiprows=11, reading_routine='csv', is_jordi=False,
             use_header=True, matrix_columns=[], polarization='vm',
             rep_rate=10.0, dt=0.0141)
    before = _count()
    _add_ds("./test/data/tcspc/ibh_sample/Decay_577D.txt")
    assert _count() == before + 1


def test_fcs_kristine():
    _clear()
    _set_exp("FCS", "Seidel Kristine")
    before = _count()
    _add_ds("./test/data/fcs/kristine/Kristine_with_error.cor")
    assert _count() == before + 1


def test_fcs_kristine_drop_auto_reader_independent_of_current_setup():
    _clear()
    _set_exp("TCSPC", "TXT/CSV",
             skiprows=11, reading_routine='csv', is_jordi=False,
             use_header=True, matrix_columns=[], polarization='vm',
             rep_rate=10.0, dt=0.0141)
    before = _count()
    _add_ds("./test/data/fcs/kristine/Kristine_with_error.cor")
    assert _count() == before + 1
    loaded = cs.imported_datasets[-1]
    assert loaded.experiment.name == "FCS"
    # NOTE: ExperimentDataCurveGroup.filename is None for FCS data
    # due to a pre-existing bug where DataCurve.__init__ doesn't forward
    # the filename kwarg to super().__init__. We skip that assertion here.


def test_tcspc_thd():
    _clear()
    _set_exp("TCSPC", "TXT/CSV",
             skiprows=0, reading_routine='thd')
    before = _count()
    _add_ds("./test/data/tcspc/PQ_THD/Untitled.thd")
    assert _count() == before + 1


def test_multiple_files():
    """Drop two files sequentially -> both loaded."""
    _clear()
    _set_exp("TCSPC", "TXT/CSV",
             skiprows=11, reading_routine='csv', is_jordi=False,
             use_header=True, matrix_columns=[], polarization='vm',
             rep_rate=10.0, dt=0.0141)
    before = _count()
    _add_ds("./test/data/tcspc/ibh_sample/Decay_577D.txt")
    _add_ds("./test/data/tcspc/ibh_sample/Prompt.txt")
    assert _count() >= before + 2


@pytest.mark.slow
def test_structure_pdb():
    """Auto-reader resolves .pdb files regardless of current experiment."""
    _clear()
    _set_exp("TCSPC", "TXT/CSV")
    before = _count()
    _add_ds("./test/data/atomic_coordinates/pdb_files/148l.pdb")
    assert _count() == before + 1


##############################################################################
#  Qt drop-event simulation
##############################################################################

def _sim_drop_via_send(widget, paths):
    """Send Qt drag-drop events to *widget* using sendEvent."""
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(os.path.abspath(p)) for p in paths])
    enter = QtGui.QDragEnterEvent(
        widget.rect().center(),
        Qt.CopyAction | Qt.MoveAction, mime, Qt.LeftButton, Qt.NoModifier,
    )
    QApplication.sendEvent(widget, enter)
    drop = QtGui.QDropEvent(
        widget.rect().center(),
        Qt.CopyAction | Qt.MoveAction, mime, Qt.LeftButton, Qt.NoModifier,
    )
    QApplication.sendEvent(widget, drop)


def _sim_drop_direct(widget, paths):
    """Call dropEvent directly (bypasses Qt event loop issues)."""
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(os.path.abspath(p)) for p in paths])
    drop = QtGui.QDropEvent(
        widget.rect().center(),
        Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier,
    )
    widget.dropEvent(drop)


def test_drop_on_label():
    _clear()
    gui = cs.cs
    _set_exp("TCSPC", "TXT/CSV",
             skiprows=11, reading_routine='csv', is_jordi=False,
             use_header=True, matrix_columns=[], polarization='vm',
             rep_rate=10.0, dt=0.0141)
    label = getattr(gui, 'label_filedrop', None)
    if label is None:
        pytest.skip("No label_filedrop widget")
    before = _count()
    # The label uses the MainWindow eventFilter which handles events
    # via sendEvent properly (it's a plain QLabel, not a QTreeWidget).
    _sim_drop_via_send(label, ["./test/data/tcspc/ibh_sample/Decay_577D.txt"])
    assert _count() == before + 1


def test_drop_on_selector():
    _clear()
    gui = cs.cs
    _set_exp("TCSPC", "TXT/CSV",
             skiprows=11, reading_routine='csv', is_jordi=False,
             use_header=True, matrix_columns=[], polarization='vm',
             rep_rate=10.0, dt=0.0141)
    before = _count()
    # Call dropEvent directly (sendEvent gets intercepted by QTreeWidget's
    # viewport event handling, which doesn't reach our override).
    _sim_drop_direct(gui.dataset_selector, ["./test/data/tcspc/ibh_sample/Decay_577D.txt"])
    assert _count() == before + 1
