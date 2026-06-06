import pytest
from qtpy import QtWidgets


def test_process_output_widget_creation(qapp, qtbot):
    pytest.importorskip("pyqtgraph")
    from chisurf.plugins.fluorescence_decay.lltf.lltf_gui import ProcessOutputWidget
    widget = ProcessOutputWidget()
    qtbot.addWidget(widget)
    assert isinstance(widget, QtWidgets.QWidget)
    assert widget.process is None
    assert widget.running is False
    assert hasattr(widget, "output_text")
    assert hasattr(widget, "stop_button")
    assert hasattr(widget, "clear_button")


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


def test_import_lltf_gui_wizard(qapp):
    pytest.importorskip("pyqtgraph")
    from chisurf.plugins.fluorescence_decay.lltf.lltf_gui import LLTFGUIWizard
    assert LLTFGUIWizard is not None
