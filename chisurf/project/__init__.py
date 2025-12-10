from __future__ import annotations

"""Core project abstraction for ChiSurf.

This package provides a minimal, GUI-independent `Project` model and
JSON-based save/load helpers. It is intentionally small for the first
incremental implementation and will be extended over time.
"""

from .project import Project, save_project, load_project

__all__ = ["Project", "save_project", "load_project"]
