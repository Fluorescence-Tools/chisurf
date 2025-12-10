from __future__ import annotations

"""Convenience re-exports for node editor UI widgets.

Historically all small widgets lived in this module; they have now been
split into dedicated files for clarity. Importing from ``ui`` remains
supported for compatibility.
"""

from .widgets import (
    InlineLabeledSlider,
    Vector1DWidget,
    StyledComboBox,
    TextBoxWidget,
    NumericValueWidget,
    WidgetPalette,
    PtPlotWidget,
    apply_node_ui_theme,
)

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
