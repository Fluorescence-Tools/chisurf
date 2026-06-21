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
FCS_SETUP_TYPE = "fcs_channel_setup"
FCS_SETUP_PREFS_ID = "fcs_channel_setup_preferences"

_FCS_CONFIG_CACHE = None

def _fcs_config():
    global _FCS_CONFIG_CACHE
    if _FCS_CONFIG_CACHE is None:
        from chisurf.gui.widgets.wizard.tttr_channeldefinition.tttr_setup_utils import SetupTypeConfig
        _FCS_CONFIG_CACHE = SetupTypeConfig(
            setup_type=FCS_SETUP_TYPE,
            id_prefix="fcs_channel_setup",
            prefs_id=FCS_SETUP_PREFS_ID,
            canonical_file=FCS_CHANNEL_SETUPS_FILE,
            description="FCS channel correlation setup",
        )
    return _FCS_CONFIG_CACHE


# ---------------------------------------------------------------------------
# MFDB-backed persistence (mirrors the detector setup pattern)
# ---------------------------------------------------------------------------

def _use_mfdb(file_path: str | None = None) -> bool:
    from chisurf.gui.widgets.wizard.tttr_channeldefinition.tttr_setup_utils import use_mfdb as _use
    return _use(file_path, FCS_CHANNEL_SETUPS_FILE)


def _save_setup_row(
    db,
    setup_name: str,
    data: dict,
    user_id: str | None = None,
    is_public: bool | int | None = None,
) -> None:
    from chisurf.gui.widgets.wizard.tttr_channeldefinition.tttr_setup_utils import (
        save_setup_row as _save_row,
    )
    from chisurf.core.settings import cs_settings as _cs
    correlator = data.get("correlator") or {}
    pairs = data.get("pairs") or []
    pair_dict: dict[str, dict] = {}
    for p in pairs:
        if isinstance(p, dict):
            name = p.get("name", "")
            if name:
                pair_dict[name] = p
    # Setup-level defaults fall back to cs_settings when the UI no longer
    # carries a top-level "correlator" key (per-pair values are authoritative).
    _n_bins = correlator.get("n_bins") or _cs.get("correlator", {}).get("B")
    _n_casc = correlator.get("n_casc") or _cs.get("correlator", {}).get("number_of_cascades")
    _make_fine = correlator.get("make_fine") if correlator.get("make_fine") is not None else _cs.get("correlator", {}).get("fine")
    _save_row(
        db, _fcs_config(), setup_name, data,
        user_id=user_id, is_public=is_public,
        fcs_pairs=pair_dict,
        n_bins=_n_bins,
        n_casc=_n_casc,
        make_fine=_make_fine,
    )


def _fcs_row_to_data(row: dict) -> dict:
    """Extract FCS channel setup payload from an MFDB row, including
    child-table data (``fcs_pairs``) and typed correlator columns."""
    from chisurf.gui.widgets.wizard.tttr_channeldefinition.tttr_setup_utils import json_loads

    configuration = json_loads(row.get("configuration_json"))
    sd = configuration.get("setup_data")
    data: dict
    if isinstance(sd, dict):
        data = dict(sd)
    else:
        data = dict(configuration)
        data.pop("setup_type", None)
        data.pop("setup_data", None)

    # Fetch full setup with child tables
    from chisurf.core.mfdb.database_resolver import resolve_database_path
    from chisurf.core.mfdb.repository import MFDatabase
    sid = row.get("setup_id")
    if sid:
        with MFDatabase(resolve_database_path()) as _db:
            full = _db.get_setup(sid)
        if full:
            fcs_pairs = full.get("fcs_pairs") or []
            pairs = []
            for p in fcs_pairs:
                pair = {
                    "name": p.get("name", ""),
                    "channel_a": p.get("channel_a", ""),
                    "channel_b": p.get("channel_b", ""),
                    "kind": p.get("kind"),
                }
                # Per-pair correlator values from child-table columns
                if p.get("n_bins") is not None:
                    pair["n_bins"] = p["n_bins"]
                if p.get("n_casc") is not None:
                    pair["n_casc"] = p["n_casc"]
                if p.get("make_fine") is not None:
                    pair["make_fine"] = bool(p["make_fine"])
                pairs.append(pair)
            if pairs:
                data["pairs"] = pairs
    # Typed correlator columns from the row itself
    correlator = data.get("correlator") or {}
    if row.get("n_bins") is not None:
        correlator["n_bins"] = row["n_bins"]
    if row.get("n_casc") is not None:
        correlator["n_casc"] = row["n_casc"]
    if row.get("make_fine") is not None:
        correlator["make_fine"] = bool(row["make_fine"])
    if correlator:
        data["correlator"] = correlator

    owner = row.get("created_by_user_id")
    if owner is not None:
        data["_owner"] = owner
    if row.get("is_public") is not None:
        data["_is_public"] = bool(row["is_public"])
    return data


def load_fcs_channel_setups(file_path: str | pathlib.Path | None = None,
                            db_path: str | None = None,
                            skip_migration: bool = False,
                            user_id: str | None = None) -> Dict[str, Any]:
    """Load FCS channel-pair setups from MFDB (preferred) or JSON fallback.

    When using the canonical path and MFDB is available, data is loaded
    from the database with user-scoped filtering and automatic migration
    of legacy JSON data.  A custom ``file_path`` bypasses MFDB and reads
    the JSON file directly.

    Parameters
    ----------
    file_path : str or Path or None
        Explicit path.  When None, the canonical settings file is used.
    db_path : str or None
        Override the resolved MFDB path (injection seam for tests so they
        never touch the real database).
    skip_migration : bool
        If True, skip legacy JSON migration into the database.
    user_id : str or None
        If provided, load setups for this specific user. If None, resolves the
        currently active user.
    """
    path = pathlib.Path(file_path) if file_path is not None else FCS_CHANNEL_SETUPS_FILE

    if _use_mfdb(str(path) if file_path else None):
        from chisurf.gui.widgets.wizard.tttr_channeldefinition.tttr_setup_utils import (
            get_db, load_mfdb_setups, resolve_active_user_id,
        )
        db = get_db(db_path)
        if db is not None:
            if user_id is None:
                user_id = resolve_active_user_id()
            if not skip_migration:
                imported = _migrate_json_to_mfdb(db, path, user_id=user_id)
                # Only remove the legacy file when we actually imported data
                # (so users who already have MFDB setups don't lose a stale
                # JSON file that may contain additional data).
                if imported:
                    try:
                        if path.exists():
                            path.unlink()
                    except Exception:
                        pass
            result = load_mfdb_setups(db, _fcs_config(), user_id, row_to_data=_fcs_row_to_data)
            return {
                "version": 1,
                "setups": result.get("setups", {}),
                "last_used_setup": result.get("last_used") or None,
            }

    # JSON fallback
    if not path.exists():
        return {"version": 1, "setups": {}, "last_used_setup": None}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return {"version": 1, "setups": {}, "last_used_setup": None}
    if not isinstance(data, dict):
        data = {}
    data.setdefault("version", 1)
    data.setdefault("setups", {})
    data.setdefault("last_used_setup", None)
    if not isinstance(data["setups"], dict):
        data["setups"] = {}
    return data


def save_fcs_channel_setups(setups_data: Dict[str, Any], file_path: str | pathlib.Path | None = None,
                            is_public: bool | int | None = None) -> bool:
    """Save FCS channel-pair setups to MFDB (preferred) or JSON file.

    Returns True on success.
    """
    path = pathlib.Path(file_path) if file_path is not None else FCS_CHANNEL_SETUPS_FILE

    if _use_mfdb(str(path) if file_path else None):
        from chisurf.gui.widgets.wizard.tttr_channeldefinition.tttr_setup_utils import (
            save_setups as _save_setups, load_setups as _load_setups,
        )
        # Convert the setups_data to the format expected by save_setups
        # (which uses "setups" dict and "last_used" key)
        mfdb_payload = {
            "setups": setups_data.get("setups", {}),
            "last_used": setups_data.get("last_used_setup"),
        }
        return _save_setups(
            mfdb_payload, _fcs_config(),
            file_path=None if file_path is None else str(path),
            replace=False,
            is_public=is_public,
            save_row_fn=_save_setup_row,
            load_scoped_fn=lambda db, cfg, uid: _load_setups(None, cfg),
        )

    # JSON export path: explicit file_path only
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(setups_data, fh, indent=4, sort_keys=False)
        tmp.replace(path)
        return True
    except Exception:
        return False


def _migrate_json_to_mfdb(db, path: pathlib.Path, user_id: str | None = None) -> bool:
    """Per-user idempotent migration from fcs_channel_setups.json to MFDB.

    Returns ``True`` when at least one setup was imported.
    """
    from chisurf.gui.widgets.wizard.tttr_channeldefinition.tttr_setup_utils import (
        migrate_json_to_mfdb as _migrate,
    )
    return _migrate(db, _fcs_config(), path, user_id=user_id, save_row_fn=_save_setup_row)


# ---------------------------------------------------------------------------
# Legacy channel builder (no MFDB changes needed)
# ---------------------------------------------------------------------------

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
