import pytest
from qtpy import QtWidgets


def test_browser_widget_creation(qapp, qtbot):
    pytest.importorskip("pyqtgraph")
    from chisurf.plugins.microscopy.sm_image_mle import BrowserWidget
    widget = BrowserWidget()
    qtbot.addWidget(widget)
    assert isinstance(widget, QtWidgets.QWidget)
    assert hasattr(widget, "spin")
    assert hasattr(widget, "ptu_line")
    assert hasattr(widget, "mol_line")


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


def test_main_window_creation(qapp, qtbot):
    pytest.importorskip("pyqtgraph")
    from chisurf.plugins.microscopy.sm_image_mle import MainWindow
    widget = MainWindow()
    qtbot.addWidget(widget)
    assert isinstance(widget, QtWidgets.QMainWindow)
    assert widget.windowTitle() == "PTU Processor GUI"
