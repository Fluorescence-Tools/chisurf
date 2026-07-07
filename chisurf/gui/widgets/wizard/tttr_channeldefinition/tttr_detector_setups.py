import json
import pathlib
from typing import Any

# NOTE: Qt is imported lazily inside ``load_detector_setups`` (only the
# missing-file warning dialog needs it).  Keeping this module import-time
# Qt-free lets the Qt-free server reuse ``load_detector_setups`` /
# ``save_detector_setups`` from ``chisurf.server.services.detector_setups``.

from chisurf.core.mfdb.repository import MFDatabase
from chisurf.core.settings.path_utils import get_path
from .tttr_setup_utils import (
    SetupTypeConfig,
    json_loads,
    load_setups,
    resolve_active_user_id,
    save_setup_row as _save_row,
    save_setups,
)

DETECTOR_SETUPS_FILE = get_path('settings') / 'detector_setups.json'
DETECTOR_SETUP_TYPE = "tttr_detector_setup"
DETECTOR_SETUP_PREFS_ID = "tttr_detector_setup_preferences"

DETECTOR_CONFIG = SetupTypeConfig(
    setup_type=DETECTOR_SETUP_TYPE,
    id_prefix="tttr_detector_setup",
    prefs_id=DETECTOR_SETUP_PREFS_ID,
    canonical_file=DETECTOR_SETUPS_FILE,
    description="TTTR detector and PIE-window setup",
)


def setup_id_for_name(name: str, user_id: str = "") -> str:
    from .tttr_setup_utils import setup_id_for_name as _sifn
    return _sifn(name, user_id, "tttr_detector_setup")


def _resolve_active_user_id() -> str:
    return resolve_active_user_id()


def _json_loads(value):
    return json_loads(value)


def _db(db_path=None):
    from .tttr_setup_utils import get_db
    return get_db(db_path)


def _use_mfdb(file_path=None) -> bool:
    from .tttr_setup_utils import use_mfdb
    return use_mfdb(file_path, DETECTOR_SETUPS_FILE)


def _setup_row_data(
    row: dict,
    detector_channels: list | None = None,
    pie_windows: list | None = None,
) -> dict:
    """Extract detector setup payload from an MFDB row with child tables."""
    configuration = json_loads(row.get("configuration_json"))
    setup_data = configuration.get("setup_data")
    if isinstance(setup_data, dict):
        return setup_data
    data = dict(configuration)
    data.pop("setup_type", None)
    data.pop("setup_data", None)

    if detector_channels:
        dets = {}
        for ch in detector_channels:
            name = ch.get("name", "")
            if not name:
                continue
            entry: dict[str, Any] = {}
            chs = ch.get("channels")
            if chs:
                entry["chs"] = json_loads(chs) if isinstance(chs, str) else chs
            mtr = ch.get("micro_time_ranges")
            if mtr:
                entry["micro_time_ranges"] = json_loads(mtr) if isinstance(mtr, str) else mtr
            if ch.get("g_factor") is not None:
                entry["g_factor"] = ch["g_factor"]
            if ch.get("l1") is not None:
                entry["l1"] = ch["l1"]
            if ch.get("l2") is not None:
                entry["l2"] = ch["l2"]
            gfc = ch.get("g_factor_channels")
            if gfc:
                entry["g_factor_channels"] = json_loads(gfc) if isinstance(gfc, str) else gfc
            if ch.get("g_factor_decay_uuid") is not None:
                entry["g_factor_decay_uuid"] = ch["g_factor_decay_uuid"]
            if ch.get("g_factor_calibration_id") is not None:
                entry["g_factor_calibration_id"] = ch["g_factor_calibration_id"]
            dets[name] = entry
        if dets:
            data["detectors"] = dets
    elif row.get("detectors_json"):
        data["detectors"] = json_loads(row.get("detectors_json"))

    if pie_windows:
        wins = {}
        for pw in pie_windows:
            name = pw.get("name", "")
            if not name:
                continue
            start = pw.get("start")
            end = pw.get("end")
            if start is not None and end is not None:
                wins[name] = (int(start), int(end))
        if wins:
            data["windows"] = wins

    timing = {}
    mtr = row.get("macro_time_resolution")
    mir = row.get("micro_time_resolution")
    mib = row.get("micro_time_binning")
    if mtr is not None:
        timing["macro_time_resolution"] = mtr
    if mir is not None:
        timing["micro_time_resolution"] = mir
    if mib is not None:
        timing["micro_time_binning"] = mib
    if not timing:
        timing = json_loads(row.get("timing_resolution_json"))
    if timing:
        data["tttr_reading"] = timing
    burst_defaults = json_loads(row.get("burst_defaults_json"))
    if burst_defaults:
        data["burst_selection"] = burst_defaults
    owner = row.get("created_by_user_id")
    if owner is not None:
        data["_owner"] = owner
    if row.get("is_public") is not None:
        data["_is_public"] = bool(row["is_public"])
    return data


def _detector_row_to_data(row: dict) -> dict:
    """Callback for ``load_mfdb_setups`` to extract detector data with child
    table priority."""
    full = None
    dcs, pws = None, None
    if row.get("setup_id"):
        from chisurf.core.mfdb.store.database_resolver import resolve_database_path
        with MFDatabase(resolve_database_path()) as _db_tmp:
            full = _db_tmp.get_setup(row["setup_id"])
        if full:
            dcs = full.get("detector_channels")
            pws = full.get("pie_windows")
    return _setup_row_data(row, detector_channels=dcs, pie_windows=pws)


def _save_setup_row(
    db: MFDatabase,
    setup_name: str,
    data: dict,
    user_id: str | None = None,
    is_public: bool | int | None = None,
) -> None:
    _save_row(
        db, DETECTOR_CONFIG, setup_name, data,
        user_id=user_id, is_public=is_public,
        detectors=data.get("detectors") or {},
        windows=data.get("windows") or {},
        timing_resolution=data.get("tttr_reading") or {},
        burst_defaults=data.get("burst_selection") or {},
    )


def _detector_save_row_fn(
    db: MFDatabase,
    setup_name: str,
    data: dict,
    user_id: str | None = None,
    is_public: bool | int | None = None,
) -> None:
    _save_setup_row(db, setup_name, data, user_id=user_id, is_public=is_public)


def _load_mfdb_detector_setups(db: MFDatabase, user_id: str | None = None) -> dict:
    from .tttr_setup_utils import load_mfdb_setups
    return load_mfdb_setups(db, DETECTOR_CONFIG, user_id, row_to_data=_detector_row_to_data)


def _set_last_used(db: MFDatabase, setup_name: str) -> None:
    from .tttr_setup_utils import set_last_used
    set_last_used(db, DETECTOR_CONFIG, setup_name)


def _migrate_json_setups_to_mfdb(
    db: MFDatabase,
    path: pathlib.Path,
    user_id: str | None = None,
) -> bool:
    from .tttr_setup_utils import migrate_json_to_mfdb
    return migrate_json_to_mfdb(db, DETECTOR_CONFIG, path, user_id=user_id, save_row_fn=_detector_save_row_fn)


def _load_json_detector_setups(path: pathlib.Path, file_path=None) -> dict:
    from .tttr_setup_utils import load_json_setups
    return load_json_setups(path, file_path=file_path)


def load_detector_setups(file_path=None, db_path=None, skip_migration=False, user_id=None):
    from .tttr_setup_utils import load_setups as _load_setups

    path = pathlib.Path(file_path or DETECTOR_SETUPS_FILE)
    if _use_mfdb(file_path):
        db = _db(db_path) if db_path is not None else _db()
        if db is not None:
            uid = user_id if user_id is not None else _resolve_active_user_id()
            if not skip_migration:
                imported = _migrate_json_setups_to_mfdb(db, path, user_id=uid)
                # Only remove legacy file when data was actually imported
                if imported:
                    try:
                        if path.exists():
                            path.unlink()
                    except Exception:
                        pass
            return _load_mfdb_detector_setups(db, user_id=uid)

    try:
        import chisurf.core.settings
        show_warning = bool(chisurf.core.settings.cs_settings.get('warn_missing_detector_setups', True))
    except Exception:
        show_warning = True

    if not path.exists():
        is_default = (file_path is None) or (path == DETECTOR_SETUPS_FILE)
        app_running = False
        try:
            from qtpy.QtWidgets import QApplication
            app_running = QApplication.instance() is not None
        except Exception:
            app_running = False

        try:
            import chisurf as _chisurf_mod
            if is_default and getattr(_chisurf_mod, "__startup_in_progress__", False):
                try:
                    _chisurf_mod.__pending_startup_onboarding__ = True
                except Exception:
                    pass
                return {"setups": {}}
        except Exception:
            pass

        if show_warning and is_default and app_running and not _module_warning_shown("detector_setups_file"):
            from qtpy.QtWidgets import QCheckBox, QMessageBox

            _mark_warning_shown("detector_setups_file")
            msg = QMessageBox()
            msg.setWindowTitle("Detector setups file not found")
            msg.setIcon(QMessageBox.Warning)
            msg.setText(f"Detector setups file was not found:\n{str(path)}")
            msg.setInformativeText(
                "You can create it by saving a setup from the Detector Wizard.\n"
                "Use the 'Save Settings' button to store your configuration.\n"
                "Alternatively, choose an existing JSON with the '...' button."
            )
            try:
                cb = QCheckBox("Don't show this warning again")
                msg.setCheckBox(cb)
            except Exception:
                cb = None

            open_btn = msg.addButton("Open Detector Wizard", QMessageBox.ActionRole)
            msg.addButton(QMessageBox.Ok)
            msg.exec_()

            try:
                if cb is not None and cb.isChecked():
                    from chisurf.core.settings.settings_utils import set_warn_missing_detector_setups as _set_w
                    _set_w(False)
                    try:
                        import chisurf.core.settings
                        chisurf.core.settings.cs_settings['warn_missing_detector_setups'] = False
                    except Exception:
                        pass
            except Exception:
                pass

            try:
                if msg.clickedButton() is open_btn:
                    from .tttr_channel_definition import DetectorWizard
                    wiz = DetectorWizard()
                    wiz.exec_()
                    if path.exists():
                        return _load_json_detector_setups(path, file_path=file_path)
            except Exception:
                pass

        return {"setups": {}}

    return _load_json_detector_setups(path, file_path=file_path)


def _module_warning_shown(key: str) -> bool:
    from chisurf.gui.widgets.warning_once import was_warning_shown
    return was_warning_shown(key)


def _mark_warning_shown(key: str) -> None:
    from chisurf.gui.widgets.warning_once import mark_warning_shown
    mark_warning_shown(key)


def save_detector_setups(setups_data, file_path=None, replace=False, is_public=None,
                         db_path=None, user_id=None):
    get_db_fn = (lambda: _db(db_path)) if db_path is not None else _db
    resolve_user_fn = (lambda: user_id) if user_id is not None else _resolve_active_user_id
    return save_setups(
        setups_data, DETECTOR_CONFIG,
        file_path=file_path, replace=replace, is_public=is_public,
        save_row_fn=_detector_save_row_fn,
        load_scoped_fn=lambda db, cfg, uid: load_mfdb_setups(db, cfg, uid, row_to_data=_detector_row_to_data),
        get_db_fn=get_db_fn,
        resolve_user_fn=resolve_user_fn,
    )
