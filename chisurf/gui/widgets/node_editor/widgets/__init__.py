from __future__ import annotations

"""UI widgets used by the node editor (view layer).

This submodule groups lightweight Qt widgets that are embedded inside node
content areas (sliders, vector editors, themed controls). Keeping them in a
separate package from the core model/scene classes makes the MVC separation
clearer:

- :mod:`model` / :class:`NodeModel`  -> model
- :mod:`node_item`, :mod:`scene`     -> view/controller
- :mod:`widgets`                     -> small view widgets
"""

from ..inline_slider import InlineLabeledSlider
from ..vector_widget import Vector1DWidget
from ..theme_widgets import apply_node_ui_theme, StyledComboBox
from .text_box_widget import TextBoxWidget
from .numeric_value_widget import NumericValueWidget
from .widget_palette import WidgetPalette
from .pt_plot_widget import PtPlotWidget

__all__ = [
    "InlineLabeledSlider",
    "Vector1DWidget",
    "StyledComboBox",
    "TextBoxWidget",
    "NumericValueWidget",
    "WidgetPalette",
    "PtPlotWidget",
    "apply_node_ui_theme",
]
