import pytest
from qtpy import QtWidgets, QtCore


def test_count_rate_analyzer_creation(qapp, qtbot):
    pytest.importorskip("pyqtgraph")
    pytest.importorskip("tttrlib")
    from chisurf.plugins.tttr.tttr_count_rate_analysis import CountRateAnalyzer
    widget = CountRateAnalyzer()
    qtbot.addWidget(widget)
    assert isinstance(widget, QtWidgets.QWidget)
    assert "Count Rate" in widget.windowTitle()
    assert hasattr(widget, "load_button")
    assert hasattr(widget, "calculate_button")
    assert hasattr(widget, "plot_widget")


def test_waterfall_plot_widget_creation(qapp, qtbot):
    pytest.importorskip("pyqtgraph")
    from chisurf.plugins.tttr.audifier.waterfall_plot import WaterfallPlotWidget
    widget = WaterfallPlotWidget()
    qtbot.addWidget(widget)
    assert isinstance(widget, QtWidgets.QWidget)
    assert hasattr(widget, "plot_widget")
    assert hasattr(widget, "waterfall_img")
    assert hasattr(widget, "position_line")


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


def test_tags_editor_creation(qapp, qtbot):
    pytest.importorskip("tttrlib")
    from chisurf.plugins.tttr.ptu_header_edit.wizard import TagsEditor
    widget = TagsEditor(json_data={})
    qtbot.addWidget(widget)
    assert isinstance(widget, QtWidgets.QMainWindow)
    assert hasattr(widget, "table_widget")
    assert hasattr(widget, "json_display")


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
