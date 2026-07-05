import html
import pathlib
import shutil

from qtpy import QtCore, QtGui

import chisurf as cs
import chisurf.core.settings


def open_in_file_manager(path: pathlib.Path) -> None:
    """Open *path* in the OS file manager (best-effort, never raises)."""
    try:
        url = QtCore.QUrl.fromLocalFile(str(path))
        QtGui.QDesktopServices.openUrl(url)
    except Exception:
        pass


def import_check(module_name: str) -> tuple[bool, str]:
    """Return ``(importable, detail)`` for *module_name* without propagating errors."""
    try:
        __import__(module_name)
        return True, "ok"
    except Exception as e:
        return False, str(e)


def settings_paths() -> dict[str, pathlib.Path]:
    """Return the well-known ChiSurf user-settings paths keyed by short name."""
    user_dir = cs.core.settings.get_path("settings")
    return {
        "user_settings_dir": user_dir,
        "settings_chisurf_yaml": user_dir / "settings_chisurf.yaml",
        "settings_colors_yaml": user_dir / "settings_colors.yaml",
        "anisotropy_corrections_json": user_dir / "anisotropy_corrections.json",
        "detector_setups_json": user_dir / "detector_setups.json",
        "styles_dir": user_dir / "styles",
        "plugins_dir": user_dir / "plugins",
        "logs_dir": user_dir / "logs",
    }


def _html_code(text: str) -> str:
    return f"<code>{html.escape(str(text))}</code>"


def _close_db(db) -> None:
    """Best-effort close of an MFDB handle opened for a read-only probe."""
    try:
        if db is not None and hasattr(db, "close"):
            db.close()
    except Exception:
        pass


def mfdb_info() -> dict:
    """Return ``{'connected': bool, 'path': str}`` for the metadata database.

    The MFDB is the authoritative store for detector/FCS setups (and more); when
    it is reachable, the legacy JSON files are expected to be absent, so their
    absence must not be reported as an error.
    """
    path = ""
    try:
        from chisurf.core.mfdb.database_resolver import resolve_database_path

        path = str(resolve_database_path())
    except Exception:
        path = ""
    db = None
    try:
        from chisurf.gui.widgets.wizard.tttr_channeldefinition.tttr_setup_utils import get_db

        db = get_db()
    except Exception:
        db = None
    connected = db is not None
    _close_db(db)
    return {"connected": connected, "path": path}


def detector_setups_summary() -> dict:
    """Return ``{'count', 'store', 'detail'}`` for detector setups (read-only).

    Prefers MFDB (the authoritative store) and falls back to the legacy JSON
    file, counting quietly without triggering the file-missing warning dialog or
    any migration side effect.
    """
    try:
        from chisurf.gui.widgets.wizard.tttr_channeldefinition.tttr_detector_setups import (
            DETECTOR_SETUPS_FILE,
            _db,
            _load_mfdb_detector_setups,
        )
    except Exception:
        return {"count": None, "store": "unknown", "detail": ""}

    db = None
    try:
        db = _db()
    except Exception:
        db = None
    if db is not None:
        try:
            res = _load_mfdb_detector_setups(db)
            n = len((res or {}).get("setups", {}) or {})
            return {"count": n, "store": "mfdb", "detail": "in MFDB"}
        except Exception:
            pass
        finally:
            _close_db(db)

    p = DETECTOR_SETUPS_FILE
    if p.exists():
        try:
            import json

            with open(p, encoding="utf-8") as fh:
                n = len((json.load(fh) or {}).get("setups", {}) or {})
        except Exception:
            n = None
        return {"count": n, "store": "file", "detail": f"in {p}"}
    return {"count": 0, "store": "none", "detail": ""}


def fcs_setups_summary() -> dict:
    """Return ``{'count', 'store', 'detail'}`` for FCS channel setups (read-only)."""
    try:
        from chisurf.core.fluorescence.fcs.channel_setups import (
            FCS_CHANNEL_SETUPS_FILE,
            load_fcs_channel_setups,
        )
    except Exception:
        return {"count": None, "store": "unknown", "detail": ""}

    # The FCS loader is MFDB-first with a quiet JSON fallback (no dialogs).
    try:
        data = load_fcs_channel_setups(skip_migration=True)
        n = len((data or {}).get("setups", {}) or {})
    except Exception:
        n = None

    if mfdb_info()["connected"]:
        return {"count": n, "store": "mfdb", "detail": "in MFDB"}
    p = FCS_CHANNEL_SETUPS_FILE
    if p.exists():
        return {"count": n, "store": "file", "detail": f"in {p}"}
    return {"count": n or 0, "store": "none", "detail": ""}


def _status_row(label: str, status: str, color: str, detail: str = "") -> str:
    """Render one table row with a coloured status word and an optional detail cell."""
    native = QtCore.QDir.toNativeSeparators(str(detail or ""))
    detail_cell = _html_code(native) if native else ""
    return (
        "<tr>"
        f"<td style='padding:4px 10px 4px 0'>{html.escape(label)}</td>"
        f"<td style='padding:4px 10px 4px 0; color:{color}; font-weight:600'>{html.escape(status)}</td>"
        f"<td style='padding:4px 0 4px 0'>{detail_cell}</td>"
        "</tr>"
    )


#: Colours for the three-state status cells.
_OK_COLOR = "#2e7d32"
_MISSING_COLOR = "#c62828"
_NEUTRAL_COLOR = "#8a8a8a"


def _file_row(label: str, ok: bool, detail: str = "") -> str:
    """Render an OK/MISSING row for a genuinely file-backed setting."""
    return _status_row(
        label, "OK" if ok else "MISSING", _OK_COLOR if ok else _MISSING_COLOR, detail
    )


def _setups_row(label: str, summary: dict) -> str:
    """Render an MFDB-aware row for a setup type.

    A count > 0 is ``OK``; an empty store is a neutral "none yet" (not an error,
    since setups are created on demand); an unknown count is neutral too.
    """
    count = summary.get("count")
    store = summary.get("store", "unknown")
    where = summary.get("detail", "")
    if count is None:
        return _status_row(label, "—", _NEUTRAL_COLOR, "could not be determined")
    if count > 0:
        noun = "setup" if count == 1 else "setups"
        return _status_row(label, "OK", _OK_COLOR, f"{count} {noun} {where}".strip())
    # count == 0 → nothing defined yet; phrase by store so it never looks broken.
    if store == "mfdb":
        return _status_row(label, "none yet", _NEUTRAL_COLOR, "MFDB connected — none defined yet")
    return _status_row(label, "none yet", _NEUTRAL_COLOR, "define one when needed")


def build_status_html() -> str:
    """Return an HTML table describing the settings files and setup stores.

    Setup types that live in the MFDB (detectors, FCS channels) are reported by
    their actual availability — not by the presence of a legacy JSON file — so a
    connected MFDB with setups never shows a misleading "MISSING" file.
    """
    p = settings_paths()
    mfdb = mfdb_info()

    rows = []
    rows.append(
        _file_row(
            "User settings directory", p["user_settings_dir"].exists(), str(p["user_settings_dir"])
        )
    )
    rows.append(
        _file_row(
            "settings_chisurf.yaml",
            p["settings_chisurf_yaml"].is_file(),
            str(p["settings_chisurf_yaml"]),
        )
    )
    rows.append(
        _file_row(
            "settings_colors.yaml",
            p["settings_colors_yaml"].is_file(),
            str(p["settings_colors_yaml"]),
        )
    )
    rows.append(
        _file_row(
            "anisotropy_corrections.json",
            p["anisotropy_corrections_json"].is_file(),
            str(p["anisotropy_corrections_json"]),
        )
    )
    rows.append(_file_row("styles/", p["styles_dir"].is_dir(), str(p["styles_dir"])))
    rows.append(_file_row("plugins/", p["plugins_dir"].is_dir(), str(p["plugins_dir"])))
    rows.append(_file_row("logs/", p["logs_dir"].is_dir(), str(p["logs_dir"])))

    # Metadata store + the setup types it now backs (not plain files anymore).
    if mfdb["connected"]:
        rows.append(_status_row("Metadata store (MFDB)", "connected", _OK_COLOR, mfdb["path"]))
    else:
        rows.append(
            _status_row("Metadata store (MFDB)", "local files", _NEUTRAL_COLOR, "not connected")
        )
    rows.append(_setups_row("Detector setups", detector_setups_summary()))
    rows.append(_setups_row("FCS channel setups", fcs_setups_summary()))

    try:
        s = getattr(cs.core.settings, "cs_settings", None)
        ok = isinstance(s, dict) and bool(s)
    except Exception:
        ok = False

    rows.append(_file_row("Runtime settings", ok, "loaded" if ok else "not loaded"))

    return (
        "<h3>Settings status</h3>"
        "<table style='border-collapse:collapse'>" + "".join(rows) + "</table>"
    )


def build_deps_html() -> str:
    """Return an HTML table describing which optional dependencies are importable."""
    deps = [
        ("tttrlib", "TTTR reading and analysis"),
        ("pyqtgraph", "Plotting in the GUI"),
        ("markdown", "Rendering Markdown docs in Help"),
        ("pymol", "3D viewer (optional)"),
    ]

    rows = []
    for mod, purpose in deps:
        ok, detail = import_check(mod)
        status = "OK" if ok else "MISSING"
        color = "#2e7d32" if ok else "#c62828"
        extra = html.escape(purpose)
        if not ok and detail:
            extra = f"{extra}<br/><span style='color:#666'>{html.escape(str(detail))}</span>"
        rows.append(
            "<tr>"
            f"<td style='padding:4px 10px 4px 0'><code>{html.escape(mod)}</code></td>"
            f"<td style='padding:4px 10px 4px 0; color:{color}; font-weight:600'>{status}</td>"
            f"<td style='padding:4px 0 4px 0'>{extra}</td>"
            "</tr>"
        )

    return (
        "<h3>Optional dependencies</h3>"
        "<table style='border-collapse:collapse'>" + "".join(rows) + "</table>"
        "<p style='color:#666'>Missing optional dependencies do not necessarily prevent ChiSurf from running, but some features may be disabled.</p>"
    )


def update_experiment_config() -> tuple[bool, str]:
    """Refresh the user ``experiment_configs.yaml`` from the packaged default.

    New or renamed fitting models (e.g. the AutoForm-ported PDA models) only
    appear once the user's copy of ``experiment_configs.yaml`` is re-synced with
    the version shipped in the package. This copies just that one file (leaving
    all other user settings untouched) and reports whether a restart is needed.

    Returns
    -------
    tuple of (bool, str)
        ``(ok, message)``.
    """
    try:
        user_dir = cs.core.settings.get_path("settings")
        pkg = pathlib.Path(cs.core.settings.__file__).resolve().parent / "experiment_configs.yaml"
        if not pkg.exists():
            return False, "Packaged experiment_configs.yaml not found."
        target = user_dir / "experiment_configs.yaml"
        if target.exists() and target.read_bytes() == pkg.read_bytes():
            return True, "Experiment configuration already up to date."
        shutil.copyfile(pkg, target)
        return True, (
            "Experiment configuration updated. Restart ChiSurf to load the new "
            "models (e.g. the AutoForm PDA models)."
        )
    except Exception as e:
        return False, str(e)


def copy_defaults(overwrite: bool) -> tuple[bool, str]:
    """Copy packaged default settings into the user folder; return ``(ok, message)``.

    When *overwrite* is ``False`` only missing files are written; when ``True`` the
    packaged defaults replace the user's current settings files.
    """
    try:
        user_dir = cs.core.settings.get_path("settings")
        pkg_dir = pathlib.Path(cs.core.settings.__file__).resolve().parent

        if overwrite:
            allowed = {".yaml", ".yml", ".json", ".qss", ".css"}
            for file in pkg_dir.iterdir():
                if not file.is_file():
                    continue
                if file.suffix.lower() not in allowed:
                    continue
                if file.name == "help_mappings.yaml":
                    continue
                shutil.copyfile(file, user_dir / file.name)
            try:
                from chisurf.core.settings.settings_utils import copy_styles_to_user_folder

                copy_styles_to_user_folder()
            except Exception:
                pass
            return True, "Defaults copied (overwrite)."

        cs.core.settings.copy_settings_to_user_folder()
        return True, "Defaults copied (missing files only)."
    except Exception as e:
        return False, str(e)
