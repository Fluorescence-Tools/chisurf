import pytest
from qtpy import QtWidgets


def test_fcs_filter_calculator_widget(qapp, qtbot):
    from chisurf.plugins.fcs.fcs_filter_calculator import FcsFilterCalculatorWidget
    widget = FcsFilterCalculatorWidget()
    qtbot.addWidget(widget)
    assert isinstance(widget, QtWidgets.QWidget)
    assert "Filter Calculator" in widget.windowTitle()
