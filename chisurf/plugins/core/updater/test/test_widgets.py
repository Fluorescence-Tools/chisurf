from qtpy import QtWidgets


def test_package_manager_widget_creation(qapp, qtbot):
    from chisurf.plugins.core.updater.package_widget import PackageManagerWidget
    widget = PackageManagerWidget()
    qtbot.addWidget(widget)
    assert isinstance(widget, QtWidgets.QWidget)
    assert hasattr(widget, "tabs")
    assert widget.tabs.count() == 4


def test_package_manager_dialog_creation(qapp, qtbot):
    from chisurf.plugins.core.updater.package_widget import (
        PackageManagerDialog,
        PackageManagerWidget,
    )
    dialog = PackageManagerDialog()
    qtbot.addWidget(dialog)
    assert isinstance(dialog, QtWidgets.QDialog)
    assert "Package" in dialog.windowTitle()
    # The dialog now wraps the reusable PackageManagerWidget.
    assert isinstance(dialog.widget, PackageManagerWidget)
    assert hasattr(dialog.widget, "tabs")
