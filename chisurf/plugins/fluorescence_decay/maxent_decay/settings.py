"""User-editable settings for the MaxEnt TCSPC lifetime / FRET plugin.

The settings are stored as a JSON file in the user's ChiSurf settings
folder (typically ``~/.chisurf/maxent_decay/settings.json``).  This allows
advanced users to tweak base defaults that are not necessarily exposed in
the GUI, while keeping the scientific code in the core/API modules.
"""

from __future__ import annotations

from typing import Any, Dict

import json
import pathlib

try:
    # Preferred location: ChiSurf user settings directory
    from chisurf.settings.path_utils import get_path as _get_path  # type: ignore
except Exception:  # pragma: no cover - fallback when chisurf.settings is unavailable
    _get_path = None  # type: ignore[assignment]


# Default settings used when the JSON file does not yet exist or cannot be
# parsed. These are intentionally conservative and mirror the hard-coded
# defaults in the GUI.
_DEFAULT_SETTINGS: Dict[str, Any] = {
    "tau_grid": {
        "min": 0.001,
        "max": 10.0,
        "step": 0.02,
    },
    "lcurve_span_decades": {
        "left": 2.0,
        "right": 2.0,
    },
    "fret": {
        "tau0": 4.1,
        "R0": 50.0,
        "period_ns": 10.0,
        "use_periodic": False,
    },
    "sampling_defaults": {
        "steps_total": 500,
        "thin": 5,
        "walkers": 0,
        "substeps": 50,
        "nprocs": 0,
        # If null/None, the platform default applies. Users may force
        # vectorized or pooled mode explicitly via this key.
        "vectorized": None,
    },
}


def get_settings_path() -> pathlib.Path:
    """Return the directory where MaxEnt settings are stored.

    Prefer the ChiSurf user settings folder (``get_path('settings')``)
    with a dedicated ``maxent_decay`` subdirectory. As a fallback, use a
    ``.chisurf/maxent_decay`` folder in the user's home directory.
    """

    # Prefer the central ChiSurf settings location if available.
    if _get_path is not None:
        try:
            base = _get_path("settings")
            path = pathlib.Path(base) / "maxent_decay"
            path.mkdir(parents=True, exist_ok=True)
            return path
        except Exception:
            pass

    # Fallback: local folder in the user's home directory.
    path = pathlib.Path.home() / ".chisurf" / "maxent_decay"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_settings_file() -> pathlib.Path:
    """Return the full path to the JSON settings file."""

    return get_settings_path() / "settings.json"


def load_maxent_settings() -> Dict[str, Any]:
    """Load MaxEnt settings from JSON, creating a default file if needed.

    The returned dictionary always contains at least the keys from
    ``_DEFAULT_SETTINGS``; user values override these defaults when
    present and valid.
    """

    path = get_settings_file()

    # If the file does not exist yet, create it with the defaults.
    if not path.is_file():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as fh:
                json.dump(_DEFAULT_SETTINGS, fh, indent=2, sort_keys=True)
        except Exception:
            # Fall back to in-memory defaults only.
            return dict(_DEFAULT_SETTINGS)
        return dict(_DEFAULT_SETTINGS)

    # Load and merge with defaults.
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise TypeError("settings JSON must contain an object at the top level")
        merged: Dict[str, Any] = dict(_DEFAULT_SETTINGS)
        for key, value in data.items():
            merged[key] = value
        return merged
    except Exception:
        # On any error, do not break the plugin; just use defaults.
        return dict(_DEFAULT_SETTINGS)


def save_maxent_settings(settings: Dict[str, Any]) -> bool:
    """Persist MaxEnt settings to JSON.

    Parameters
    ----------
    settings:
        Dictionary to serialize. Callers are expected to pass a structure
        compatible with ``_DEFAULT_SETTINGS`` but additional keys are
        allowed and will be preserved.
    """

    path = get_settings_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(settings, fh, indent=2, sort_keys=True)
        return True
    except Exception:
        return False


__all__ = [
    "_DEFAULT_SETTINGS",
    "get_settings_path",
    "get_settings_file",
    "load_maxent_settings",
    "save_maxent_settings",
]
