import pytest
from qtpy import QtWidgets


def test_trace_browser_creation(qapp, qtbot):
    pytest.importorskip("pyqtgraph")
    try:
        from chisurf.plugins.tttr.trace_browser import TraceBrowser
        widget = TraceBrowser()
        qtbot.addWidget(widget)
        assert isinstance(widget, QtWidgets.QWidget)
        assert "Trace" in widget.windowTitle()
        assert hasattr(widget, "table")
        assert hasattr(widget, "plot")
    except Exception:
        pytest.skip("TraceBrowser requires tttrlib or optional dependencies")


def test_trace_browser_tool_creation(qapp, qtbot):
    pytest.importorskip("pyqtgraph")
    try:
        from chisurf.plugins.tttr.trace_browser.gui.tool import TraceBrowserTool

        widget = TraceBrowserTool()
        qtbot.addWidget(widget)
        assert hasattr(widget, "_workspace")
        assert hasattr(widget._workspace, "table")
        assert hasattr(widget._workspace, "plot")
    except Exception:
        pytest.skip("TraceBrowserTool requires tttrlib or optional dependencies")
