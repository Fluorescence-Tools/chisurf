import pytest
from qtpy import QtWidgets


def test_fcs_filter_calculator_widget(qapp, qtbot):
    from chisurf.plugins.fcs.fcs_filter_calculator import FcsFilterCalculatorWidget
    widget = FcsFilterCalculatorWidget()
    qtbot.addWidget(widget)
    assert isinstance(widget, QtWidgets.QWidget)
    assert "Filter Calculator" in widget.windowTitle()


def test_no_embedded_detector_editor_uses_setup_selector(qapp, qtbot):
    """The embedded detector wizard is gone; the widget only *selects* saved
    setups (the authoritative editor is the toolbox's Detector Def tool)."""
    from chisurf.plugins.fcs.fcs_filter_calculator import FcsFilterCalculatorWidget

    widget = FcsFilterCalculatorWidget()
    qtbot.addWidget(widget)
    assert widget.detector_wizard_page is None
    assert hasattr(widget, "combo_detector_setup")
