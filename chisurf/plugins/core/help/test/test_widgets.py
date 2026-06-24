"""Tests for the Help plugin widgets."""

from qtpy import QtWidgets


def test_help_widget_creation(qapp, qtbot):
    from chisurf.plugins.core.help.gui.tool import HelpWidget
    widget = HelpWidget()
    qtbot.addWidget(widget)
    assert isinstance(widget, QtWidgets.QWidget)
    assert "Help" in widget.windowTitle() or "Help" in widget.__class__.__name__
    assert hasattr(widget, "tree")
    assert hasattr(widget, "viewer")
    assert hasattr(widget, "title_label")
    assert hasattr(widget, "edit_btn")
    assert hasattr(widget, "save_btn")


def test_help_widget_toolbar(qapp, qtbot):
    from chisurf.plugins.core.help.gui.tool import HelpWidget
    widget = HelpWidget()
    qtbot.addWidget(widget)
    toolbars = widget.findChildren(QtWidgets.QToolBar)
    assert len(toolbars) >= 1
    toolbar = toolbars[0]
    actions = toolbar.actions()
    assert len(actions) >= 5
    labels = [a.text() for a in actions]
    assert any("Edit" in label or "👁" in label or "✏" in label for label in labels)
    assert any("Save" in label or "💾" in label for label in labels)
