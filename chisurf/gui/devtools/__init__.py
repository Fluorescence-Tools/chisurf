"""
Dev Tools for ChiSurf Development Mode.

This package provides developer utilities for ChiSurf, including:
- Source jumping (resolve widget/object source locations)
- Code badge buttons (floating `</>` buttons to open source)
"""

from .source_jump import (
    resolve_widget_source,
    resolve_object_source,
    resolve_focused_widget_source,
    resolve_fit_window_source,
    resolve_parameter_group_source,
    resolve_experiment_panel_source,
    open_in_editor,
    make_resolver,
    make_widget_resolver,
)

__all__ = [
    "resolve_widget_source",
    "resolve_object_source",
    "resolve_focused_widget_source",
    "resolve_fit_window_source",
    "resolve_parameter_group_source",
    "resolve_experiment_panel_source",
    "open_in_editor",
    "make_resolver",
    "make_widget_resolver",
]
