import pytest
from qtpy import QtWidgets, QtCore


def test_count_rate_analyzer_creation(qapp, qtbot):
    pytest.importorskip("pyqtgraph")
    pytest.importorskip("tttrlib")
    from chisurf.plugins.tttr.tttr_count_rate_analysis import CountRateAnalyzer
    widget = CountRateAnalyzer()
    qtbot.addWidget(widget)
    assert isinstance(widget, QtWidgets.QWidget)
    assert "Count Rate" in widget.windowTitle()
    assert hasattr(widget, "load_button")
    assert hasattr(widget, "calculate_button")
    assert hasattr(widget, "plot_widget")
