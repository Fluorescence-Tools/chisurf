"""Backwards-compatible shim re-exporting :mod:`chisurf.core.dataspec`.

PRD-40 lifted this module out from under ``models/`` so the same declarative
``dataset → editor`` framework can describe settings, metadata and tool panels,
not just models. Existing imports (``from chisurf.core.models import view_spec``)
keep working through this re-export; new code should import from
:mod:`chisurf.core.dataspec` directly.
"""
from __future__ import annotations

from chisurf.core.dataspec import (  # noqa: F401
    _SECTION_TYPES,
    ChoiceSection,
    CurveInputSection,
    CustomSection,
    DynamicGroupSection,
    ModelView,
    PanelSection,
    ParameterGroupSection,
    ParameterGroupTableSection,
    ParameterGroupView,
    PlotSpec,
    Section,
    ToggleSection,
    ToggleRowSection,
    ValueSection,
    _section_from_dict,
    load_view_spec,
)

__all__ = [
    "Section",
    "ParameterGroupSection",
    "ParameterGroupTableSection",
    "DynamicGroupSection",
    "CurveInputSection",
    "PanelSection",
    "ChoiceSection",
    "ToggleSection",
    "ToggleRowSection",
    "ValueSection",
    "CustomSection",
    "PlotSpec",
    "ModelView",
    "ParameterGroupView",
    "load_view_spec",
]
