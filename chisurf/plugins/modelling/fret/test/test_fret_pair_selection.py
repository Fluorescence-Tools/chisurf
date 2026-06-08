import pytest
from qtpy import QtWidgets


def test_fret_pair_selection_window_creation(qapp, qtbot):
    pytest.importorskip("pyqtgraph")
    pytest.importorskip("mdtraj")
    from chisurf.plugins.modelling.fret.gui.pair_selection_wizard import FRETPairSelectionWindow
    widget = FRETPairSelectionWindow()
    qtbot.addWidget(widget)
    assert isinstance(widget, QtWidgets.QMainWindow)
    assert "FRET" in widget.windowTitle() or "Pair" in widget.windowTitle()
    assert hasattr(widget, "traj_edit")
    assert hasattr(widget, "run_btn")
    assert hasattr(widget, "table")
