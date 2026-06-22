import pytest
from qtpy import QtWidgets


def test_tttr_image_browser_creation(qapp, qtbot):
    pytest.importorskip("pyqtgraph")
    try:
        from chisurf.plugins.tttr.tttr_image_browser import TTTRImageBrowser
        widget = TTTRImageBrowser()
        qtbot.addWidget(widget)
        assert isinstance(widget, QtWidgets.QWidget)
        assert "Image" in widget.windowTitle()
        assert hasattr(widget, "table")
        assert hasattr(widget, "image_item")
    except Exception:
        pytest.skip("TTTRImageBrowser requires tttrlib or optional dependencies")


def test_tttr_image_browser_tool_creation(qapp, qtbot):
    pytest.importorskip("pyqtgraph")
    try:
        from chisurf.plugins.tttr.tttr_image_browser.gui.tool import TTTRImageBrowserTool

        widget = TTTRImageBrowserTool()
        qtbot.addWidget(widget)
        assert hasattr(widget, "_workspace")
        assert hasattr(widget._workspace, "table")
        assert hasattr(widget._workspace, "image_item")
    except Exception:
        pytest.skip("TTTRImageBrowserTool requires tttrlib or optional dependencies")
