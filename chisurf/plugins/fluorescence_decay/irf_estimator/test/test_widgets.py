import pytest
from qtpy import QtWidgets


def test_irf_estimator_plugin_creation(qapp, qtbot):
    pytest.importorskip("pyqtgraph")
    from chisurf.plugins.fluorescence_decay.irf_estimator import IRFEstimatorPlugin
    widget = IRFEstimatorPlugin()
    qtbot.addWidget(widget)
    assert isinstance(widget, QtWidgets.QWidget)
    assert widget.windowTitle() == "IRF Estimator - Blind IRF Estimation"
    assert hasattr(widget, "load_button")
    assert hasattr(widget, "estimate_button")
    assert hasattr(widget, "main_plot")
