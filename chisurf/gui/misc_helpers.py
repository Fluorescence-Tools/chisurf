from __future__ import annotations

import importlib
import pathlib
from typing import Optional

import chisurf
from chisurf.gui import QtWidgets, QtCore


def add_dataset(main_window) -> None:
    filename = main_window.current_setup.controller.get_filename()
    if isinstance(filename, list):
        parts = [pathlib.Path(f).as_posix() for f in filename]
        s = '|'.join(parts)
    elif isinstance(filename, pathlib.Path):
        s = filename.as_posix()
    else:
        s = f"{filename}"
    s = s.replace("\\", "/")
    chisurf.run(f'chisurf.macros.add_dataset(filename=r"{s}")')


def open_help(main_window, topic: Optional[str] = None) -> None:
    """Open the help plugin with optional topic filter or markdown path."""
    try:
        try:
            help_plugin = importlib.import_module("chisurf.plugins.chisurf.help")
        except Exception:
            help_plugin = importlib.import_module("chisurf.plugins.help")
        window = getattr(main_window, "_help_window", None)
        if window is None or not isinstance(window, help_plugin.HelpWidget):
            window = help_plugin.HelpWidget()
            try:
                window.destroyed.connect(lambda _=None: setattr(main_window, "_help_window", None))
            except Exception:
                pass
            main_window._help_window = window

        try:
            if topic:
                txt = str(topic).strip()
                if txt:
                    handled = False
                    try:
                        path_part = txt
                        anchor = None
                        if "#" in txt:
                            path_part, frag = txt.split("#", 1)
                            path_part = path_part.strip()
                            anchor = frag.strip() or None

                        if path_part.lower().endswith(".md"):
                            raw_path = pathlib.Path(path_part)
                            if not raw_path.is_absolute():
                                try:
                                    base = pathlib.Path(chisurf.__file__).resolve().parent
                                    root = base.parent
                                    candidate = (root / raw_path).resolve()
                                except Exception:
                                    candidate = raw_path
                            else:
                                candidate = raw_path

                            if candidate.exists():
                                try:
                                    window.open_markdown_path(candidate, anchor)
                                    handled = True
                                except Exception:
                                    handled = False
                    except Exception:
                        handled = False

                    if not handled:
                        window.filter_line_edit.setText(txt)
        except Exception:
            pass

        window.show()
        try:
            window.raise_()
            window.activateWindow()
        except Exception:
            pass
    except Exception as e:
        chisurf.gui.widgets.general.MyMessageBox(
            label="Help Plugin Error",
            info=f"Error loading help plugin: {str(e)}",
            show_fortune=False,
        )


def open_updater(main_window) -> None:
    try:
        try:
            updater_plugin = importlib.import_module("chisurf.plugins.chisurf.updater")
        except ImportError:
            updater_plugin = importlib.import_module("chisurf.plugins.updater")
        window = updater_plugin.UpdaterWidget()
        window.show()
    except Exception as e:
        chisurf.gui.widgets.general.MyMessageBox(
            label="Updater Plugin Error",
            info=f"Error loading updater plugin: {str(e)}",
            show_fortune=False,
        )


def open_about(main_window) -> None:
    try:
        try:
            about_plugin = importlib.import_module("chisurf.plugins.chisurf.about")
        except ImportError:
            about_plugin = importlib.import_module("chisurf.plugins.about")
        window = about_plugin.AboutDialog(parent=main_window)
        window.show()
    except Exception as e:
        chisurf.gui.widgets.general.MyMessageBox(
            label="About Plugin Error",
            info=f"Error opening About dialog: {str(e)}",
            show_fortune=False,
        )


def clear_local_settings(main_window) -> None:
    chisurf.settings.clear_settings_folder()
    chisurf.gui.widgets.general.MyMessageBox(
        label="Settings Reset",
        info="Local settings have been reset successfully.",
        show_fortune=False,
    )


def clear_user_styles(main_window) -> None:
    user_styles_path = chisurf.settings.get_path('settings') / 'styles'
    if user_styles_path.exists() and user_styles_path.is_dir():
        for file in user_styles_path.glob('*.qss'):
            try:
                file.unlink()
            except Exception:
                pass

    chisurf.gui.widgets.general.MyMessageBox(
        label="User Styles Cleared",
        info="All user style files (QSS) have been removed.",
        show_fortune=False,
    )
