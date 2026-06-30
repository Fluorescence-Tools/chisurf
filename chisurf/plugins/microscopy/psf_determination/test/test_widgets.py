import pytest
from qtpy import QtWidgets


def test_psf_determination_tool_creation(qapp, qtbot):
    pytest.importorskip("pyqtgraph")
    from chisurf.plugins.microscopy.psf_determination.gui.tool import PsfDeterminationTool

    widget = PsfDeterminationTool()
    qtbot.addWidget(widget)
    assert isinstance(widget, QtWidgets.QWidget)
    assert widget.windowTitle() == "PSF Determination"
    # The whole UI is a single AutoForm bound to the Qt-free view-model.
    assert hasattr(widget, "auto_form")
    assert widget.model is not None
