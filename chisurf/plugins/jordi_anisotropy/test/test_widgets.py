import pytest
from qtpy import QtWidgets


class TestJordiAnisotropyCalculator:
    def test_creation(self, qapp, qtbot):
        from chisurf.plugins.jordi_anisotropy import JordiAnisotropyCalculator
        widget = JordiAnisotropyCalculator()
        qtbot.addWidget(widget)
        assert widget is not None

    def test_window_title(self, qapp, qtbot):
        from chisurf.plugins.jordi_anisotropy import JordiAnisotropyCalculator
        widget = JordiAnisotropyCalculator()
        qtbot.addWidget(widget)
        assert "Jordi" in widget.windowTitle()
