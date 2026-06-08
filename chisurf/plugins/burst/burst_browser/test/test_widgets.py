import pytest
from qtpy import QtWidgets


def test_burst_browser_widget(qapp, qtbot):
    from chisurf.plugins.burst.burst_browser import BurstBrowserWidget
    widget = BurstBrowserWidget()
    qtbot.addWidget(widget)
    assert isinstance(widget, QtWidgets.QWidget)
    assert widget.windowTitle() == "Burst Browser"
