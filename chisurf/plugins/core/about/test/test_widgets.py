import pytest
from qtpy import QtWidgets


def test_about_dialog_creation(qapp, qtbot):
    from chisurf.plugins.core.about import AboutDialog
    dialog = AboutDialog()
    qtbot.addWidget(dialog)
    assert isinstance(dialog, QtWidgets.QDialog)
    assert "About" in dialog.windowTitle()
    assert hasattr(dialog, "textEdit")
    assert hasattr(dialog, "toolButton")
