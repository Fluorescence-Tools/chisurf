import pytest
from qtpy import QtWidgets


def test_help_widget_creation(qapp, qtbot):
    from chisurf.plugins.core.help import HelpWidget
    widget = HelpWidget()
    qtbot.addWidget(widget)
    assert isinstance(widget, QtWidgets.QWidget)
    assert "Help" in widget.windowTitle() or "Help" in widget.__class__.__name__
    assert hasattr(widget, "tree")
    assert hasattr(widget, "viewer")
    assert hasattr(widget, "title_label")
