"""Section/plot registry for view-spec driven model editors.

Importing this package registers the built-in plot keys and custom sections so
:class:`~chisurf.gui.widgets.models.auto_model_widget.AutoModelWidget` can
resolve a model's :class:`~chisurf.core.models.view_spec.ModelView`.
"""
from __future__ import annotations

from .registry import (
    register_plot,
    register_section,
    get_plot_class,
    get_section_factory,
    resolve_plot_specs,
)
from . import builtin  # noqa: F401  (side effect: populate the registry)

__all__ = [
    "register_plot",
    "register_section",
    "get_plot_class",
    "get_section_factory",
    "resolve_plot_specs",
]
