import numpy as np
from qtpy import QtWidgets


def test_flc_tool_builds(qapp, qtbot):
    from chisurf.plugins.fcs.flc_2d.gui.tool import FlcTwoDTool

    widget = FlcTwoDTool()
    qtbot.addWidget(widget)
    assert isinstance(widget, QtWidgets.QWidget)
    # the declarative plot sources must be callable and return lists
    assert widget._model.lifetime_series() == []
    assert widget._model.correlation_series() == []


def test_plugin_class_importable(qapp, qtbot):
    from chisurf.plugins.fcs.flc_2d import TwoDFCSPlugin, TwoDFLCPlugin

    assert TwoDFLCPlugin is TwoDFCSPlugin
    widget = TwoDFLCPlugin()
    qtbot.addWidget(widget)
    assert isinstance(widget, QtWidgets.QWidget)


def test_plot_sources_render_after_results(qapp, qtbot):
    from chisurf.plugins.fcs.flc_2d.gui.tool import FlcTwoDTool

    widget = FlcTwoDTool()
    qtbot.addWidget(widget)
    widget._model._lifetime = {"tau": np.geomspace(0.3, 8, 10), "amp": np.ones(10)}
    widget._model._correlation = [
        {"x": np.array([1e-3, 1e-2]), "y": np.array([2.0, 1.0]), "name": "auto"}
    ]
    widget._plots_form.refresh_plots()
    assert len(widget._model.lifetime_series()) == 1
    assert len(widget._model.correlation_series()) == 1
