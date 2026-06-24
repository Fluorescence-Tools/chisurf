"""Shared base classes for dockable transformer tools (PRD-23).

Provides a common dockable-tool base (`ChisurfDockTool`) and a reusable
path-drop list (`PathDropListWidget`) so every transformer GUI reuses one
implementation of drag-drop, dock central widget, window geometry persistence,
and MFDB-connectivity status instead of forking its own.
"""

from __future__ import annotations

from .chisurf_dock_tool import ChisurfDockTool, PathDropListWidget

__all__ = ["ChisurfDockTool", "PathDropListWidget"]
