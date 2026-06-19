import pytest
from qtpy import QtWidgets


def test_irf_estimator_tool_creation(qapp, qtbot):
    pytest.importorskip("pyqtgraph")
    from chisurf.plugins.fluorescence_decay.irf_estimator import IRFEstimatorTool
    widget = IRFEstimatorTool()
    qtbot.addWidget(widget)
    assert isinstance(widget, QtWidgets.QMainWindow)
    assert widget.windowTitle() == "IRF Estimator - Blind IRF Estimation"
    assert hasattr(widget, "estimate_button")
    assert hasattr(widget, "main_plot")
    assert hasattr(widget, "save_action")
    assert hasattr(widget, "transfer_action")
