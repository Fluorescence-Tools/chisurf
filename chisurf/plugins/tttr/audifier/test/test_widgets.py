import pytest
from qtpy import QtWidgets


def test_waterfall_plot_widget_creation(qapp, qtbot):
    pytest.importorskip("pyqtgraph")
    from chisurf.plugins.tttr.audifier.waterfall_plot import WaterfallPlotWidget
    widget = WaterfallPlotWidget()
    qtbot.addWidget(widget)
    assert isinstance(widget, QtWidgets.QWidget)
    assert hasattr(widget, "plot_widget")
    assert hasattr(widget, "waterfall_img")
    assert hasattr(widget, "position_line")
