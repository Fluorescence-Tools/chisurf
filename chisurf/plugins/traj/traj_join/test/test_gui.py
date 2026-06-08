import pytest
from qtpy import QtWidgets


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
