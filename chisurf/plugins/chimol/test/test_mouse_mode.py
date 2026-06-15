"""Tests for Chimol mouse rotation modes."""

from __future__ import annotations

from chisurf.plugins.chimol.chimol.renderer.qtgl import QtGLRenderer
from chisurf.plugins.chimol.chimol.renderer.view import MolView
from chisurf.plugins.chimol.chimol import config


def test_display_config_default_mouse_mode():
    """The shipped display config should default to PyMOL-style rotation."""
    assert config._DISPLAY_CONFIG["camera"]["mouse_mode"] == "pymol"


def test_qtgl_normalize_mouse_mode():
    """QtGLRenderer should normalise mouse mode strings."""
    assert QtGLRenderer._normalize_mouse_mode("pymol") == "pymol"
    assert QtGLRenderer._normalize_mouse_mode("PyMOL") == "pymol"
    assert QtGLRenderer._normalize_mouse_mode("chimol") == "chimol"
    assert QtGLRenderer._normalize_mouse_mode("Chimol") == "chimol"
    assert QtGLRenderer._normalize_mouse_mode("unknown") == "pymol"
    assert QtGLRenderer._normalize_mouse_mode(None) == "pymol"


def test_qtgl_rotation_delta_multiplier():
    """PyMOL mode inverts left-drag rotation deltas relative to Chimol mode."""
    assert QtGLRenderer._rotation_delta_multiplier("pymol") == -1.0
    assert QtGLRenderer._rotation_delta_multiplier("chimol") == 1.0


def test_qtgl_pan_delta_multiplier():
    """PyMOL mode keeps pan deltas object-following; Chimol inverts them."""
    assert QtGLRenderer._pan_delta_multiplier("pymol") == 1.0
    assert QtGLRenderer._pan_delta_multiplier("chimol") == -1.0


def test_molview_normalize_mouse_mode():
    """MolView uses the same normalisation rules as the renderer."""
    assert MolView._normalize_mouse_mode("pymol") == "pymol"
    assert MolView._normalize_mouse_mode("chimol") == "chimol"
    assert MolView._normalize_mouse_mode("invalid") == "pymol"
