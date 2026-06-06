import pytest
from qtpy import QtWidgets


class TestJordiGFactorCalculator:
    def test_creation(self, qapp, qtbot):
        pytest.importorskip("pyqtgraph")
        from chisurf.plugins.jordi_g_factor import JordiGFactorCalculator
        widget = JordiGFactorCalculator()
        qtbot.addWidget(widget)
        assert widget is not None

    def test_window_title(self, qapp, qtbot):
        pytest.importorskip("pyqtgraph")
        from chisurf.plugins.jordi_g_factor import JordiGFactorCalculator
        widget = JordiGFactorCalculator()
        qtbot.addWidget(widget)
        assert widget.windowTitle() == "Jordi G-Factor Calculator"
