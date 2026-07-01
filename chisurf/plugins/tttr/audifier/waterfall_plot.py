"""Backwards-compatible re-export.

``WaterfallPlotWidget`` moved to :mod:`chisurf.gui.widgets.waterfall_plot` so it
is a general, reusable widget (and backs the AutoForm ``waterfall`` section)
available to all plugins. This shim keeps existing imports working.
"""

from chisurf.gui.widgets.waterfall_plot import WaterfallPlotWidget

__all__ = ["WaterfallPlotWidget"]
