import pytest
from qtpy import QtWidgets


def test_help_widget_creation(qapp, qtbot):
    from chisurf.plugins.chisurf.help import HelpWidget
    widget = HelpWidget()
    qtbot.addWidget(widget)
    assert isinstance(widget, QtWidgets.QWidget)
    assert "Help" in widget.windowTitle() or "Help" in widget.__class__.__name__
    assert hasattr(widget, "tree")
    assert hasattr(widget, "viewer")
    assert hasattr(widget, "title_label")


def test_about_dialog_creation(qapp, qtbot):
    from chisurf.plugins.chisurf.about import AboutDialog
    dialog = AboutDialog()
    qtbot.addWidget(dialog)
    assert isinstance(dialog, QtWidgets.QDialog)
    assert "About" in dialog.windowTitle()
    assert hasattr(dialog, "textEdit")
    assert hasattr(dialog, "toolButton")


def test_style_manager_widget_creation(qapp, qtbot):
    from chisurf.plugins.chisurf.style_manager import StyleManagerWidget
    widget = StyleManagerWidget()
    qtbot.addWidget(widget)
    assert isinstance(widget, QtWidgets.QMainWindow)
    assert "Style" in widget.windowTitle()
    assert hasattr(widget, "file_combo")
    assert hasattr(widget, "editor")
    assert hasattr(widget, "status_bar")


def test_plugin_manager_widget_creation(qapp, qtbot):
    from chisurf.plugins.chisurf.plugin_manager import PluginManagerWidget
    widget = PluginManagerWidget()
    qtbot.addWidget(widget)
    assert isinstance(widget, QtWidgets.QMainWindow)
    assert "Plugin" in widget.windowTitle()
    assert hasattr(widget, "plugin_list")
    assert hasattr(widget, "description_edit")
    assert hasattr(widget, "disabled_checkbox")


def test_package_manager_dialog_creation(qapp, qtbot):
    from chisurf.plugins.chisurf.updater.package_widget import PackageManagerDialog
    dialog = PackageManagerDialog()
    qtbot.addWidget(dialog)
    assert isinstance(dialog, QtWidgets.QDialog)
    assert "Package" in dialog.windowTitle()
    assert hasattr(dialog, "tabs")
