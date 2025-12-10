from __future__ import annotations

"""Compatibility shim for the legacy editor module.

Historically the main widget lived in :mod:`editor`. This lightweight module
simply re-exports :class:`NodeEditorWidget` so callers can ``import
node_editor`` without depending on the old name.
"""

from .editor import NodeEditorWidget  # noqa: F401

__all__ = ["NodeEditorWidget"]
