"""GUI adapters for the Time Window Bins plugin."""

from __future__ import annotations

from typing import Any


def settings_from_controls(wizard: Any) -> dict[str, Any]:
    """Extract settings from the GUI controls as a JSON-compatible dict.

    Parameters
    ----------
    wizard : TTTRTimeWindowTool
        The main tool instance whose controls to read.

    Returns
    -------
    dict
        Settings dict with ``time_window_ms`` and optional ``output_dir``.
    """
    return {
        "time_window_ms": float(wizard.tws_spin.value()),
        "output_dir": wizard.output_edit.text().strip() or None,
    }
