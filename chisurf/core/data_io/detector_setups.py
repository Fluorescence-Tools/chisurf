from __future__ import annotations

import json
import logging
import pathlib
from typing import Any

from chisurf.core.settings.path_utils import get_path

logger = logging.getLogger(__name__)

DETECTOR_SETUPS_FILE = get_path('settings') / 'detector_setups.json'


def load_detector_setups(
    file_path: str | pathlib.Path | None = None,
) -> dict[str, Any]:
    """Load detector setups from the canonical JSON file.

    Returns ``{"setups": {...}, "last_used": str}``.

    This is the Qt-free variant used by the headless server.  It does not
    attempt MFDB migration and does not show a warning dialog when the file
    is missing — it simply returns an empty dict.
    """
    path = pathlib.Path(file_path) if file_path is not None else DETECTOR_SETUPS_FILE
    if not path.exists():
        logger.debug("Detector setups file not found: %s", path)
        return {"setups": {}}
    try:
        with open(path) as f:
            data: dict[str, Any] = json.load(f)
        setups = data.get("setups") or {}
        last_used = data.get("last_used") or ""
        return {"setups": setups, "last_used": last_used}
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load detector setups from %s: %s", path, exc)
        return {"setups": {}}


def save_detector_setups(
    setups_data: dict[str, Any],
    file_path: str | pathlib.Path | None = None,
    is_public: bool | None = None,
) -> None:
    """Save detector setups to the canonical JSON file.

    Parameters
    ----------
    setups_data : dict
        Must contain a ``"setups"`` key mapping name -> settings dict.
    file_path : str or Path, optional
        Override the default file path.
    is_public : bool, optional
        Ignored in the headless JSON-only variant (only relevant for MFDB).
    """
    path = pathlib.Path(file_path) if file_path is not None else DETECTOR_SETUPS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "setups": setups_data.get("setups") or {},
    }
    last_used = setups_data.get("last_used")
    if last_used:
        payload["last_used"] = last_used
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
