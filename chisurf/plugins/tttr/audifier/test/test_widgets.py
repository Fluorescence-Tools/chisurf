import pytest
from qtpy import QtWidgets


def test_waterfall_plot_widget_creation(qapp, qtbot):
    pytest.importorskip("pyqtgraph")
    # moved to a shared widget; the plugin path still re-exports it
    from chisurf.gui.widgets.waterfall_plot import WaterfallPlotWidget

    widget = WaterfallPlotWidget()
    qtbot.addWidget(widget)
    assert isinstance(widget, QtWidgets.QWidget)
    assert hasattr(widget, "plot_widget")
    assert hasattr(widget, "waterfall_img")
    assert hasattr(widget, "position_line")


def test_general_waterfall_section_registered():
    from chisurf.gui.autoform.sections.registry import get_section_factory

    assert get_section_factory("waterfall") is not None


def test_audifier_tool_builds_with_autoform(qapp, qtbot):
    pytest.importorskip("pyqtgraph")
    pytest.importorskip("tttrlib")
    from chisurf.gui.autoform import AutoForm
    from chisurf.gui.autoform.sections.registry import get_section_factory
    from chisurf.plugins.tttr.audifier.gui.tool import TTTRAudifierWidget

    w = TTTRAudifierWidget()
    qtbot.addWidget(w)
    assert isinstance(w.auto_form, AutoForm)
    for key in ("audifier_setup", "audifier_mix", "audifier_transport"):
        assert get_section_factory(key) is not None


def test_view_model_channel_state():
    from chisurf.plugins.tttr.audifier.gui.view_model import AudifierViewModel

    m = AudifierViewModel()
    m.set_detectors_from_settings({"detectors": {"green": {"chs": [0, 1]}, "red": {"chs": [2]}}})
    assert m.channels == [0, 1, 2]
    assert len(m.detectors) == 2
    # colours are Qt-free (r, g, b) float tuples
    assert all(len(d["color"]) == 3 for d in m.detectors)
    m.set_channel(0, enabled=False)
    assert 0 not in m.selected_channels()
    assert m.can_render() is not None  # no data loaded
