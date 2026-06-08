import pytest
from qtpy import QtWidgets


def test_psf_determination_widget_creation(qapp, qtbot):
    pytest.importorskip("pyqtgraph")
    from chisurf.plugins.microscopy.psf_determination import PSFDeterminationWidget
    widget = PSFDeterminationWidget()
    qtbot.addWidget(widget)
    assert isinstance(widget, QtWidgets.QWidget)
    assert widget.windowTitle() == "PSF Determination"
    assert hasattr(widget, "image_view")
    assert hasattr(widget, "fit_button")
    assert hasattr(widget, "results_text")
