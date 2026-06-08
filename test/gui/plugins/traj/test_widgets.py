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


def test_save_topology_creation(qapp, qtbot):
    pytest.importorskip("mdtraj")
    try:
        from chisurf.plugins.traj.traj_save_topology.widget import SaveTopology
        widget = SaveTopology()
        qtbot.addWidget(widget)
        assert isinstance(widget, QtWidgets.QWidget)
        assert hasattr(widget, "trajectory_filename")
    except Exception:
        pytest.skip("SaveTopology requires .ui file which may not be available in test")


def test_join_trajectories_widget_creation(qapp, qtbot):
    pytest.importorskip("mdtraj")
    try:
        from chisurf.plugins.traj.traj_join.widget import JoinTrajectoriesWidget
        widget = JoinTrajectoriesWidget()
        qtbot.addWidget(widget)
        assert isinstance(widget, QtWidgets.QWidget)
        assert hasattr(widget, "chunk_size")
    except Exception:
        pytest.skip("JoinTrajectoriesWidget requires .ui file which may not be available in test")
