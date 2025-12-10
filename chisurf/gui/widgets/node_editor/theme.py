from __future__ import annotations

import json
import os
from typing import Any, Mapping, MutableMapping, Sequence

from qtpy import QtGui


_THEME: MutableMapping[str, Any] = {}


def _load_theme() -> None:
    """Load colors and layout metrics from theme.json.

    Color entries default to RGB triplets defined here. Additional keys in the
    JSON file can be either scalars (for layout metrics) or sequences. No error
    is raised for unexpected formats; callers provide sensible fallbacks.
    """

    global _THEME
    base: Mapping[str, Sequence[int]] = {
        "scene_background": (35, 35, 35),
        "scene_gradient_top": (30, 30, 30),
        "scene_gradient_bottom": (45, 45, 45),
        "scene_grid_color": (60, 60, 60),
        "node_body_top": (62, 68, 74),
        "node_body_bottom": (48, 52, 58),
        "node_title_top": (70, 130, 175),
        "node_title_bottom": (52, 105, 145),
        "node_title_top_selected": (88, 156, 204),
        "node_title_bottom_selected": (60, 120, 170),
        "node_border": (40, 40, 40),
        "node_border_selected": (90, 180, 255),
        "port_neutral": (135, 135, 135),
        "port_input_connected": (110, 170, 255),
        "port_output_connected": (235, 205, 70),
    }

    data: MutableMapping[str, Any] = dict(base)
    path = os.path.join(os.path.dirname(__file__), "theme", "theme.json")
    try:
        with open(path, "r", encoding="utf8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            for k, v in raw.items():
                data[k] = v
    except Exception:
        # Fall back to built-in defaults
        pass

    _THEME = data


_load_theme()


def color(name: str, fallback: Sequence[int]) -> QtGui.QColor:
    rgb = _THEME.get(name, fallback)
    try:
        r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])  # type: ignore[index]
    except Exception:
        r, g, b = fallback
    return QtGui.QColor(r, g, b)


def metric(name: str, fallback: float) -> float:
    """Return a float layout/spacing metric from the theme.

    Accepts either a scalar in the JSON or a sequence, using the first element.
    """

    v = _THEME.get(name, fallback)
    try:
        if isinstance(v, (list, tuple)):
            return float(v[0])
        return float(v)
    except Exception:
        return float(fallback)


def flag(name: str, fallback: bool) -> bool:
    v = _THEME.get(name, fallback)
    try:
        if isinstance(v, (list, tuple)):
            if not v:
                return bool(fallback)
            v = v[0]
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("1", "true", "yes", "on"):
                return True
            if s in ("0", "false", "no", "off"):
                return False
        return bool(v)
    except Exception:
        return bool(fallback)


def text(name: str, fallback: str) -> str:
    """Return a string value (e.g. font family) from the theme."""

    v = _THEME.get(name, fallback)
    try:
        if isinstance(v, (list, tuple)) and v:
            return str(v[0])
        return str(v)
    except Exception:
        return str(fallback)
