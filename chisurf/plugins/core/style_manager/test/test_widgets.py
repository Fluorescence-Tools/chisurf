import pytest
from qtpy import QtWidgets


def test_style_manager_widget_creation(qapp, qtbot):
    from chisurf.plugins.core.style_manager import StyleManagerWidget
    widget = StyleManagerWidget()
    qtbot.addWidget(widget)
    assert isinstance(widget, QtWidgets.QMainWindow)
    assert "Style" in widget.windowTitle()
    assert hasattr(widget, "file_combo")
    assert hasattr(widget, "editor")
    assert hasattr(widget, "status_bar")
