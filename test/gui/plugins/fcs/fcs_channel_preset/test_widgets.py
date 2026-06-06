import pytest
from qtpy import QtWidgets


def test_fcs_channel_dialog(qapp, qtbot):
    from chisurf.plugins.fcs.fcs_channel_preset import FCSChannelDialog
    widget = FCSChannelDialog()
    qtbot.addWidget(widget)
    assert isinstance(widget, QtWidgets.QDialog)
    assert "FCS Channel Definitions" in widget.windowTitle()
