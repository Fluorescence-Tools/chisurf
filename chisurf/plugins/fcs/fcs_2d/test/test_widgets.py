import pytest
from qtpy import QtWidgets


def test_two_d_fcs_wizard(qapp, qtbot):
    from chisurf.plugins.fcs.fcs_2d import TwoDFCSWizard as _TwoDFCSWizard
    if _TwoDFCSWizard is None:
        pytest.skip("TwoDFCSWizard not available (missing dependencies)")
    widget = _TwoDFCSWizard()
    qtbot.addWidget(widget)
    assert isinstance(widget, QtWidgets.QWidget)
