import pytest
from qtpy import QtWidgets


def test_plugin_manager_widget_creation(qapp, qtbot):
    from chisurf.plugins.core.plugin_manager import PluginManagerWidget
    widget = PluginManagerWidget()
    qtbot.addWidget(widget)
    assert isinstance(widget, QtWidgets.QWidget)
    assert "Plugin" in widget.windowTitle()
    assert hasattr(widget, "plugin_list")
    assert hasattr(widget, "description_edit")
    assert hasattr(widget, "disabled_checkbox")
