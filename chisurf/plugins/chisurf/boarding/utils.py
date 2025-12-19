import html
import pathlib
import shutil

from qtpy import QtCore, QtGui

import chisurf
import chisurf.settings


def open_in_file_manager(path: pathlib.Path) -> None:
    try:
        url = QtCore.QUrl.fromLocalFile(str(path))
        QtGui.QDesktopServices.openUrl(url)
    except Exception:
        pass


def import_check(module_name: str) -> tuple[bool, str]:
    try:
        __import__(module_name)
        return True, "ok"
    except Exception as e:
        return False, str(e)


def settings_paths() -> dict[str, pathlib.Path]:
    user_dir = chisurf.settings.get_path('settings')
    return {
        'user_settings_dir': user_dir,
        'settings_chisurf_yaml': user_dir / 'settings_chisurf.yaml',
        'settings_colors_yaml': user_dir / 'settings_colors.yaml',
        'anisotropy_corrections_json': user_dir / 'anisotropy_corrections.json',
        'detector_setups_json': user_dir / 'detector_setups.json',
        'styles_dir': user_dir / 'styles',
        'plugins_dir': user_dir / 'plugins',
        'logs_dir': user_dir / 'logs',
    }


def _html_code(text: str) -> str:
    return f"<code>{html.escape(str(text))}</code>"


def build_status_html() -> str:
    p = settings_paths()

    def _row(label: str, ok: bool, detail: str = "") -> str:
        status = "OK" if ok else "MISSING"
        color = "#2e7d32" if ok else "#c62828"
        native = QtCore.QDir.toNativeSeparators(str(detail or ""))
        detail_cell = _html_code(native) if native else ""
        return (
            "<tr>"
            f"<td style='padding:4px 10px 4px 0'>{html.escape(label)}</td>"
            f"<td style='padding:4px 10px 4px 0; color:{color}; font-weight:600'>{status}</td>"
            f"<td style='padding:4px 0 4px 0'>{detail_cell}</td>"
            "</tr>"
        )

    rows = []
    rows.append(_row("User settings directory", p['user_settings_dir'].exists(), str(p['user_settings_dir'])))
    rows.append(_row("settings_chisurf.yaml", p['settings_chisurf_yaml'].is_file(), str(p['settings_chisurf_yaml'])))
    rows.append(_row("settings_colors.yaml", p['settings_colors_yaml'].is_file(), str(p['settings_colors_yaml'])))
    rows.append(_row("anisotropy_corrections.json", p['anisotropy_corrections_json'].is_file(), str(p['anisotropy_corrections_json'])))
    rows.append(_row("detector_setups.json", p['detector_setups_json'].is_file(), str(p['detector_setups_json'])))
    rows.append(_row("styles/", p['styles_dir'].is_dir(), str(p['styles_dir'])))
    rows.append(_row("plugins/", p['plugins_dir'].is_dir(), str(p['plugins_dir'])))
    rows.append(_row("logs/", p['logs_dir'].is_dir(), str(p['logs_dir'])))

    try:
        cs = getattr(chisurf.settings, 'cs_settings', None)
        ok = isinstance(cs, dict) and bool(cs)
    except Exception:
        ok = False

    rows.append(_row("Runtime settings", ok, "loaded" if ok else "not loaded"))

    return (
        "<h3>Settings status</h3>"
        "<table style='border-collapse:collapse'>"
        + "".join(rows)
        + "</table>"
    )


def build_deps_html() -> str:
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
        "<table style='border-collapse:collapse'>"
        + "".join(rows)
        + "</table>"
        "<p style='color:#666'>Missing optional dependencies do not necessarily prevent ChiSurf from running, but some features may be disabled.</p>"
    )


def copy_defaults(overwrite: bool) -> tuple[bool, str]:
    try:
        user_dir = chisurf.settings.get_path('settings')
        pkg_dir = pathlib.Path(chisurf.settings.__file__).resolve().parent

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
                from chisurf.settings.settings_utils import copy_styles_to_user_folder
                copy_styles_to_user_folder()
            except Exception:
                pass
            return True, "Defaults copied (overwrite)."

        chisurf.settings.copy_settings_to_user_folder()
        return True, "Defaults copied (missing files only)."
    except Exception as e:
        return False, str(e)
