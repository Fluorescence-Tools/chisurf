"""Backwards-compatible shim — the renderer now lives in :mod:`chisurf.gui.autoform`.

PRD-40 lifted ``AutoModelWidget`` out from under ``models/`` and renamed it
``AutoForm`` so it can render any declarative editor, not just fitting models.
Existing imports keep working through this re-export.
"""
from __future__ import annotations

from chisurf.gui.autoform.auto_form import AutoForm, AutoModelWidget  # noqa: F401

__all__ = ["AutoForm", "AutoModelWidget"]
