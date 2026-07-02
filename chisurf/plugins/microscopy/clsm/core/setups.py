"""CLSM acquisition-setup presets and marker auto-detection (Qt-free)."""

from __future__ import annotations

import pathlib
from typing import Any

import yaml

_SETTINGS_FILE = pathlib.Path(__file__).parent / "clsm_settings.yaml"


def _normalise_frame_marker(value: Any) -> list[int]:
    """Coerce a ``frame_marker`` field into a list of ints.

    The YAML presets store frame markers as e.g. ``4,`` or ``4, 6`` which PyYAML
    parses as a string or a list; normalise both to ``list[int]``.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(part) for part in str(value).split(",") if part.strip()]


def builtin_setups() -> dict[str, dict[str, Any]]:
    """Return the built-in CLSM setup presets keyed by display name.

    Each preset carries ``tttr_type``, ``routine``, ``frame_marker``
    (``list[int]``), ``line_start_marker``, ``line_stop_marker``,
    ``event_type_marker`` and an optional ``pixel_per_line``.
    """
    with open(_SETTINGS_FILE) as fp:
        raw = yaml.safe_load(fp) or {}
    setups: dict[str, dict[str, Any]] = {}
    for name, preset in raw.items():
        preset = dict(preset)
        preset["frame_marker"] = _normalise_frame_marker(preset.get("frame_marker"))
        preset.setdefault("pixel_per_line", 0)
        setups[name] = preset
    return setups


def read_clsm_markers(tttr: Any) -> dict[str, Any]:
    """Read CLSM marker settings embedded in a TTTR object.

    Returns a dict with the same keys used by :func:`builtin_setups`
    (``frame_marker``, ``line_start_marker``, ``line_stop_marker``,
    ``event_type_marker``, ``pixel_per_line``).  Empty dict if the header
    carries no CLSM metadata.
    """
    import tttrlib

    detected = tttrlib.CLSMImage.read_clsm_settings(tttr)
    if not detected:
        return {}
    return {
        "frame_marker": _normalise_frame_marker(detected.get("marker_frame_start")),
        "line_start_marker": detected.get("marker_line_start"),
        "line_stop_marker": detected.get("marker_line_stop"),
        "event_type_marker": detected.get("marker_event_type"),
        "pixel_per_line": detected.get("n_pixel_per_line", 0),
    }
