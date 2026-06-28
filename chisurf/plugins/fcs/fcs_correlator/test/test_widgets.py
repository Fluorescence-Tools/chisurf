import pytest
from qtpy import QtWidgets


def test_fcs_correlator_tool(qapp, qtbot):
    from chisurf.plugins.fcs.fcs_correlator.tool import FcsCorrelatorTool
    from chisurf.gui.widgets.navigation import NavigationPanelTool
    widget = FcsCorrelatorTool()
    qtbot.addWidget(widget)
    assert isinstance(widget, NavigationPanelTool)
