"""Generic renderer that builds a Qt editor from a declarative ``DataSet``.

``AutoForm`` walks a UI-agnostic spec from :mod:`chisurf.core.dataspec` and
composes the editor from registered item/section widgets (PRD-40). It was lifted
out of ``chisurf.gui.widgets.models.auto_model_widget`` (PRD-38), where it was
scoped to fitting models, so settings, metadata and tool panels can be rendered
the same way. ``AutoModelWidget`` remains as a backwards-compatible alias.
"""

from __future__ import annotations

from chisurf.gui.autoform.auto_form import AutoForm, AutoModelWidget
from chisurf.gui.autoform.sections.registry import (
    get_plot_class,
    get_section_factory,
    register_plot,
    register_section,
    resolve_plot_specs,
)

__all__ = [
    "AutoForm",
    "AutoModelWidget",
    "get_section_factory",
    "get_plot_class",
    "register_section",
    "register_plot",
    "resolve_plot_specs",
]
