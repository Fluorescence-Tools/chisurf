import pytest
from qtpy import QtWidgets


def test_confocal_calc_widget(qapp, qtbot):
    from chisurf.plugins.fcs.fcs_calculator.wizard import ConfocalCalcWidget
    widget = ConfocalCalcWidget()
    qtbot.addWidget(widget)
    assert isinstance(widget, QtWidgets.QWidget)
    assert "FCS Confocal Calculator" in widget.windowTitle()
