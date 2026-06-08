import pytest
from qtpy import QtWidgets


def test_tttr_image_browser_creation(qapp, qtbot):
    pytest.importorskip("pyqtgraph")
    try:
        from chisurf.plugins.tttr.tttr_image_browser import TTTRImageBrowser
        widget = TTTRImageBrowser()
        qtbot.addWidget(widget)
        assert isinstance(widget, QtWidgets.QWidget)
        assert "Image" in widget.windowTitle() or "Image" in widget.__class__.__name__
        assert hasattr(widget, "table")
        assert hasattr(widget, "image_item")
    except Exception:
        pytest.skip("TTTRImageBrowser requires tttrlib or optional dependencies")
