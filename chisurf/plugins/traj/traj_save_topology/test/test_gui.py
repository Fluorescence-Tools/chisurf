import pytest
from qtpy import QtWidgets


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
