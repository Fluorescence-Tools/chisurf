"""GUI for FRET modelling.

``FretDockingTool`` is the AutoForm-based docking widget (the maintained GUI,
driven by ``fret_dock.view.json`` and the IMP/IMP.bff engine). ``FretDockWizard``
is the legacy Qt wizard, kept importable while the legacy spring engine is phased
out.
"""

from __future__ import annotations

from .dock_tool import FretDockingTool

try:  # legacy wizard depends on the legacy spring engine
    from .wizard import FretDockWizard
except Exception:  # pragma: no cover
    FretDockWizard = None
