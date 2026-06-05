from __future__ import annotations

"""Common helpers for burst-wise FCS plugins.

This module centralizes small utilities that are shared between the
FCS correlation channel-definition plugin and the burst-wise diffusion
analysis plugin.

The helpers here are intentionally lightweight and have no GUI
dependencies so that they can be used from both GUI and CLI contexts.
"""

import json
import pathlib
from typing import Any, Dict

from chisurf.core.settings.path_utils import get_path


# Path to the user-level FCS channel-pair configuration
FCS_CHANNEL_SETUPS_FILE: pathlib.Path = get_path("settings") / "fcs_channel_setups.json"


def load_fcs_channel_setups(file_path: str | pathlib.Path | None = None) -> Dict[str, Any]:
    """Load FCS channel-pair setups from the given JSON file.

    The structure is kept simple and mirrors the detector_setups.json
    layout used elsewhere in ChiSurf:

    {
        "version": 1,
        "setups": {
            "DetectorSetupName": {
                "correlator": {
                    "n_bins": int,
                    "n_casc": int,
                    "make_fine": bool,
                },
                "pairs": [
                    {
                        "name": str,              # human-readable label, also column base name
                        "channel_a": str,        # logical channel key, e.g. "DD1" or "prompt_green"
                        "channel_b": str,        # logical channel key
                        "kind": str | None,      # "ACF", "CCF", etc. (optional, purely descriptive)
                    },
                    ...
                ],
            },
            ...
        },
        "last_used_setup": "DetectorSetupName" | null
    }
    """

    path = pathlib.Path(file_path) if file_path is not None else FCS_CHANNEL_SETUPS_FILE
    if not path.exists():
        # Minimal default structure
        return {"version": 1, "setups": {}, "last_used_setup": None}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        # On error fall back to an empty default so callers can continue
        return {"version": 1, "setups": {}, "last_used_setup": None}

    # Ensure required top-level keys exist
    if not isinstance(data, dict):
        data = {}
    data.setdefault("version", 1)
    data.setdefault("setups", {})
    data.setdefault("last_used_setup", None)
    if not isinstance(data["setups"], dict):
        data["setups"] = {}
    return data


def save_fcs_channel_setups(setups_data: Dict[str, Any], file_path: str | pathlib.Path | None = None) -> bool:
    """Save FCS channel-pair setups to the given JSON file.

    Returns True on success and False if any error occurs. Existing
    files are overwritten atomically by writing to a temporary file and
    then replacing the original.
    """

    path = pathlib.Path(file_path) if file_path is not None else FCS_CHANNEL_SETUPS_FILE
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(setups_data, fh, indent=4, sort_keys=False)
        tmp.replace(path)
        return True
    except Exception:
        return False


def build_channels_from_setup(windows: Dict[str, tuple[int, int]],
                              detectors: Dict[str, Dict[str, Any]]) -> Dict[str, list[Dict[str, Any]]]:
    """Recreate the channel mapping used by DetectorWizardPage.channels().

    Parameters
    ----------
    windows:
        Mapping of PIE-window name to a pair ``(start, end)`` in micro-time
        bins.
    detectors:
        Mapping of detector name to a dictionary that must at least
        contain the keys ``"chs"`` (routing-channel list) and
        ``"micro_time_ranges"`` (list of ``(start, end)`` tuples).

    Returns
    -------
    dict
        A dictionary mapping logical channel names (detector names from the
        setup) to a list of segment dictionaries with the keys
        ``"window_range"``, ``"detector_chs"`` and ``"micro_time_range"``.
    """

    channels: Dict[str, list[Dict[str, Any]]] = {}
    if not isinstance(detectors, dict):
        return channels

    # Logical channels are detector-based; each detector aggregates segments
    # over all windows and its own micro-time ranges.
    win_dict = windows if isinstance(windows, dict) else {}

    for dname, dinfo in detectors.items():
        try:
            chs = list(map(int, dinfo.get("chs", [])))
            mtr_list = dinfo.get("micro_time_ranges", []) or []
        except Exception:
            continue

        segments: list[Dict[str, Any]] = []

        if win_dict:
            # Combine each detector micro-time range with all defined windows
            for wrange in win_dict.values():
                try:
                    w_start, w_stop = int(wrange[0]), int(wrange[1])
                except Exception:
                    continue
                for mtr in mtr_list:
                    try:
                        mt0, mt1 = int(mtr[0]), int(mtr[1])
                    except Exception:
                        continue
                    segments.append({
                        "window_range": (w_start, w_stop),
                        "detector_chs": chs,
                        "micro_time_range": (mt0, mt1),
                    })
        else:
            # No windows defined: fall back to detector + micro-time ranges only
            for mtr in mtr_list:
                try:
                    mt0, mt1 = int(mtr[0]), int(mtr[1])
                except Exception:
                    continue
                segments.append({
                    "window_range": None,
                    "detector_chs": chs,
                    "micro_time_range": (mt0, mt1),
                })

        if segments:
            channels[dname] = segments

    return channels
