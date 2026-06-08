import pytest
from qtpy import QtWidgets


def test_package_manager_dialog_creation(qapp, qtbot):
    from chisurf.plugins.core.updater.package_widget import PackageManagerDialog
    dialog = PackageManagerDialog()
    qtbot.addWidget(dialog)
    assert isinstance(dialog, QtWidgets.QDialog)
    assert "Package" in dialog.windowTitle()
    assert hasattr(dialog, "tabs")
