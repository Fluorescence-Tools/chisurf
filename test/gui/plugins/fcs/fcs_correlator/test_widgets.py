import pytest
from qtpy import QtWidgets


def test_fcs_correlator_wizard(qapp, qtbot):
    from chisurf.plugins.fcs.fcs_correlator.wizard import ChisurfWizard
    widget = ChisurfWizard()
    qtbot.addWidget(widget)
    assert isinstance(widget, QtWidgets.QWizard)
