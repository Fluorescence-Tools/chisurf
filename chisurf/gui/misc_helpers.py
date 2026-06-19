from __future__ import annotations

import ast
import importlib
import os
import pathlib
from functools import partial

import numpy as np

import chisurf as cs
from chisurf import logging
from chisurf.gui import QtWidgets, QtGui, QtCore
from chisurf.gui.gui_tweaks import apply_platform_window_tweaks
from chisurf.gui.widgets.general import LogListWidget
from chisurf.gui.widgets import system_info_watermark as _system_info_watermark


class TruncatingStatusBar(QtWidgets.QStatusBar):
    """QStatusBar that truncates overly long messages by keeping the start and end
    and inserting ellipsis in the middle."""

    def __init__(self, *args, max_message_length: int = 160, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_message_length = max(7, int(max_message_length))  # minimum to allow x...y

    def setMaxMessageLength(self, n: int):
        try:
            self._max_message_length = max(7, int(n))
        except Exception:
            pass

    def _format_message(self, message: str) -> str:
        try:
            s = str(message)
        except Exception:
            return message
        max_len = self._max_message_length
        if not s or len(s) <= max_len:
            return s
        keep_total = max_len - 3
        start_keep = keep_total // 2
        end_keep = keep_total - start_keep
        return f"{s[:start_keep]}...{s[-end_keep:]}"

    def showMessage(self, message: str, timeout: int = 0):  # type: ignore[override]
        super().showMessage(self._format_message(message), timeout)


def warmup_imports():
    """Preload heavy modules to improve first-use responsiveness."""
    try:
        import pyqtgraph as _pg  # noqa: F401
        from matplotlib import colors as _mcolors  # noqa: F401
        import scipy.linalg as _sl  # noqa: F401
        import scipy.stats as _sstats  # noqa: F401
        import chisurf.gui.widgets.fitting as _fitwidgets  # noqa: F401
        _ = getattr(_fitwidgets, "FittingControllerWidget", None)
        _ = importlib.import_module("chisurf.core.models.global_model.globalfit")
    except Exception as e:
        try:
            logging.debug(f"warmup_imports encountered: {e}")
        except Exception:
            pass


def apply_window_tweaks(window) -> None:
    """Apply small, non-invasive tweaks to the main window frame."""
    apply_platform_window_tweaks(window)


def init_system_info_watermark(parent, label=None):
    try:
        return _system_info_watermark.ensure_watermark(parent, label)
    except Exception:
        return label


def update_system_info_watermark_geometry(label) -> None:
    try:
        _system_info_watermark.update_geometry(label)
    except Exception:
        pass


def setup_log_list_widget(window) -> None:
    """Replace the plainTextEditLog with a LogListWidget while preserving items."""
    existing_items = []
    if hasattr(window, "plainTextEditLog"):
        try:
            for i in range(window.plainTextEditLog.count()):
                existing_items.append(window.plainTextEditLog.item(i).text())
        except Exception:
            existing_items = []

    try:
        parent_widget = window.plainTextEditLog.parent()
        layout = parent_widget.layout()
    except Exception:
        return

    layout_index = None
    for i in range(layout.count()):
        if layout.itemAt(i).widget() == window.plainTextEditLog:
            layout_index = i
            break
    if layout_index is None:
        return

    window.plainTextEditLog.setParent(None)
    window.plainTextEditLog = LogListWidget(parent_widget)
    window.plainTextEditLog.setObjectName("plainTextEditLog")
    layout.insertWidget(layout_index, window.plainTextEditLog)

    for item_text in existing_items:
        window.plainTextEditLog.addItem(item_text)


def filter_log_content(window):
    """Highlight matching log rows and optionally hide non-matching rows."""
    try:
        filter_text = window.lineEdit_LogFilter.text().strip().lower()
    except Exception:
        filter_text = ""
    try:
        hide_non_matching = window.checkBox_filter_hide.isChecked()
    except Exception:
        hide_non_matching = False

    widget = getattr(window, "plainTextEditLog", None)
    if widget is None:
        return
    if hasattr(widget, "filter_log_content"):
        previous_external_filter = getattr(widget, "_external_filter", None)
        widget._external_filter = (filter_text, hide_non_matching)
        widget.filter_log_content()
        if previous_external_filter is None:
            del widget._external_filter
        else:
            widget._external_filter = previous_external_filter
        return
    if not hasattr(widget, "rowCount"):
        return

    for row in range(widget.rowCount()):
        message = widget.item(row, 3)
        row_text = widget.row_text(row) if hasattr(widget, "row_text") else message.text()
        match = not filter_text or filter_text in row_text.lower()
        widget.setRowHidden(row, hide_non_matching and not match)
        if hasattr(widget, "reset_row_styles"):
            widget.reset_row_styles(row)

        if filter_text and match:
            for column in range(widget.columnCount()):
                item = widget.item(row, column)
                if item is None:
                    continue
                item.setBackground(QtGui.QBrush(QtGui.QColor(255, 255, 0, 50)))
                font = item.font()
                font.setBold(True)
                item.setFont(font)

    if filter_text and hide_non_matching:
        visible_rows = [row for row in range(widget.rowCount()) if not widget.isRowHidden(row)]
        if not visible_rows:
            cs.logging.debug("No log entries match the current filter.")


def update_log_filter(window):
    """Update log filtering when new entries are added."""
    filter_log_content(window)


def run_macro(filename=None, executor: str = "console", globals=None, locals=None, main_window=None):
    """Run a macro file via console or exec."""
    if filename is None and main_window is not None:
        filename = cs.gui.widgets.get_filename("Python macros", file_type="Python file (*.py)")
    if filename is None:
        return
    cs.logging.info(f"Running script: {filename}")

    if executor == "console":
        cs.console.run_macro(filename=pathlib.Path(filename).as_posix())
        return

    # executor == exec
    if globals is None:
        globals = {
            "__name__": "__main__",
            "cs": cs,
            "np": np,
            "os": os,
            "QtCore": QtCore,
            "QtGui": QtGui,
            "cs": main_window,
        }
    globals.update({"__file__": filename})

    import sys
    import importlib

    macro_dir = str(pathlib.Path(filename).parent)
    original_sys_path = sys.path.copy()
    if macro_dir not in sys.path:
        sys.path.insert(0, macro_dir)

    def _get_plugin_separator(filename: str) -> str | None:
        for sep in ("/plugins/", "\\plugins\\"):
            if sep in filename:
                return sep
        return None

    try:
        # Detect plugin package context for reload behavior
        plugin_sep = _get_plugin_separator(str(filename))
        if plugin_sep is not None:
            path_sep = "\\" if "\\" in plugin_sep else "/"
            parts = str(filename).split(plugin_sep)
            if len(parts) > 1:
                plugin_path = parts[1].split(path_sep)
                if len(plugin_path) > 1:
                    module_parts = plugin_path[:-1]
                elif len(plugin_path) == 1:
                    module_parts = plugin_path
                else:
                    module_parts = []

                if module_parts:
                    package_name = ".".join(module_parts)
                    user_plugin_root = pathlib.Path.home() / ".cs" / "plugins"
                    is_user_plugin = str(filename).startswith(str(user_plugin_root))
                    if is_user_plugin:
                        globals.update({"__package__": None})
                        user_plugin_path = user_plugin_root / package_name / "__init__.py"
                        if user_plugin_path.exists():
                            try:
                                source = user_plugin_path.read_text(encoding="utf-8")
                                tree = ast.parse(source, filename=str(user_plugin_path))
                                for node in ast.walk(tree):
                                    if isinstance(node, ast.Assign):
                                        for target in node.targets:
                                            if isinstance(target, ast.Name) and target.id == "name":
                                                if isinstance(node.value, ast.Str):
                                                    plugin_name = node.value.s
                                                    cs.logging.info(f"User plugin name: {plugin_name}")
                                                elif isinstance(node.value, ast.Constant) and isinstance(
                                                    node.value.value, str
                                                ):
                                                    plugin_name = node.value.value
                                                    cs.logging.info(f"User plugin name: {plugin_name}")
                            except Exception as e:
                                cs.logging.warning(f"Error extracting name from {user_plugin_path}: {e}")
                    else:
                        globals.update({"__package__": f"chisurf.plugins.{package_name}"})
                        plugin_module_prefix = f"chisurf.plugins.{package_name}"
                        for module_name in list(sys.modules.keys()):
                            if module_name.startswith(plugin_module_prefix):
                                try:
                                    cs.logging.info(f"Reloading module: {module_name}")
                                    importlib.reload(sys.modules[module_name])
                                except Exception as e:
                                    cs.logging.warning(f"Failed to reload module {module_name}: {e}")

        # Resolve missing user plugin file
        if not pathlib.Path(filename).exists():
            user_plugin_root = pathlib.Path.home() / ".cs" / "plugins"
            plugin_sep = _get_plugin_separator(str(filename))
            if plugin_sep is not None:
                path_sep = "\\" if "\\" in plugin_sep else "/"
                parts = str(filename).split(plugin_sep)
                if len(parts) > 1:
                    plugin_path = parts[1].replace(path_sep, "/")
                    user_plugin_path = user_plugin_root / plugin_path
                    if user_plugin_path.exists():
                        filename = user_plugin_path
                        cs.logging.info(f"Found file in user plugins directory: {filename}")
                    else:
                        cs.logging.error(f"File not found: {filename}")
                        cs.logging.error(f"Also checked user plugin path: {user_plugin_path}")
                        raise FileNotFoundError(f"File not found: {filename}")
            else:
                cs.logging.error(f"File not found: {filename}")
                raise FileNotFoundError(f"File not found: {filename}")

        with open(filename, "rb") as file:
            exec(compile(file.read(), filename, "exec"), globals, locals)
    except Exception as e:
        cs.logging.error(f"Error executing macro: {e}")
        raise
    finally:
        sys.path = original_sys_path


def run_plugin_from_dir(main_window, plugin_dir_to_use):
    """Run a plugin given its directory using toolbar/menu logic."""
    try:
        plugin_dir_to_use = pathlib.Path(plugin_dir_to_use)
        wizard_path = plugin_dir_to_use / "wizard.py"
        init_path = plugin_dir_to_use / "__init__.py"

        # Keep a persistent globals dict per plugin directory so plugin-created widgets
        # (e.g. F-Test) are not garbage-collected immediately when launched from toolbar.
        if not hasattr(main_window, "_plugin_contexts"):
            main_window._plugin_contexts = {}
        plugin_key = str(plugin_dir_to_use)
        context = main_window._plugin_contexts.get(plugin_key)
        if context is None:
            context = {"__name__": "plugin"}
            main_window._plugin_contexts[plugin_key] = context

        # Check if wizard.py exists
        if wizard_path.exists():
            adr = "https://github.com/fluorescence-tools/cs"  # Default value
            context.setdefault("adr", adr)
            p = partial(
                main_window.onRunMacro, str(wizard_path),
                executor='exec',
                globals=context
            )
            p()
        elif init_path.exists():
            # If no wizard.py, run the plugin's __init__.py using onRunMacro
            main_window.onRunMacro(
                str(init_path),
                executor='exec',
                globals=context
            )
        else:
            cs.logging.warning(f"No wizard.py or __init__.py found for plugin directory: {plugin_dir_to_use}")
    except Exception as e:
        cs.logging.error(f"Error running plugin from {plugin_dir_to_use}: {e}")


def get_plugin_settings_path(plugin_name: str) -> pathlib.Path:
    """Get the path to the settings file for a given plugin inside the chisurf user settings folder.

    Parameters
    ----------
    plugin_name : str
        The name of the plugin.

    Returns
    -------
    pathlib.Path
        The path to the settings file.
    """
    try:
        from chisurf.core.settings import get_path
        settings_dir = get_path('settings')
    except Exception:
        settings_dir = pathlib.Path.home() / '.chisurf'
    settings_dir.mkdir(parents=True, exist_ok=True)
    return settings_dir / f"plugin_{plugin_name}_settings.ini"


def save_plugin_window_state(window, plugin_name: str) -> None:
    """Save the window geometry and state (dock layout) of a plugin.

    Parameters
    ----------
    window : QMainWindow
        The main window widget of the plugin.
    plugin_name : str
        The name of the plugin.
    """
    try:
        ini_path = get_plugin_settings_path(plugin_name)
        settings = QtCore.QSettings(str(ini_path), QtCore.QSettings.IniFormat)
        settings.setValue("geometry", window.saveGeometry())
        if hasattr(window, "saveState"):
            settings.setValue("state", window.saveState())
    except Exception as e:
        try:
            logging.warning(f"Failed to save window state for plugin {plugin_name}: {e}")
        except Exception:
            pass


def restore_plugin_window_state(window, plugin_name: str) -> None:
    """Restore the window geometry and state (dock layout) of a plugin.

    Parameters
    ----------
    window : QMainWindow
        The main window widget of the plugin.
    plugin_name : str
        The name of the plugin.
    """
    try:
        ini_path = get_plugin_settings_path(plugin_name)
        if ini_path.exists():
            settings = QtCore.QSettings(str(ini_path), QtCore.QSettings.IniFormat)
            geo = settings.value("geometry")
            if geo is not None:
                window.restoreGeometry(geo)
            if hasattr(window, "restoreState"):
                state = settings.value("state")
                if state is not None:
                    window.restoreState(state)
    except Exception as e:
        try:
            logging.warning(f"Failed to restore window state for plugin {plugin_name}: {e}")
        except Exception:
            pass


def persist_plugin_state(plugin_name: str):
    """Class decorator that adds automatic window state persistence to a plugin widget.

    When the widget is shown, it restores its geometry/state from the user settings folder.
    When the widget is closed, it saves its geometry/state.

    Parameters
    ----------
    plugin_name : str
        Unique identifier for this plugin (used as the settings filename key).

    Examples
    --------
    @persist_plugin_state("my_plugin")
    class MyPluginWidget(QMainWindow):
        ...
    """
    def decorator(cls):
        orig_init = cls.__init__
        orig_close = getattr(cls, 'closeEvent', None)
        orig_show = getattr(cls, 'showEvent', None)

        def _new_init(self, *args, **kwargs):
            self._persist_plugin_name = plugin_name
            self._persist_state_restored = False
            orig_init(self, *args, **kwargs)
            self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        def _new_showEvent(self, event):
            if not self._persist_state_restored:
                self._persist_state_restored = True
                try:
                    restore_plugin_window_state(self, self._persist_plugin_name)
                except Exception:
                    pass
            if orig_show is not None:
                orig_show(self, event)
            else:
                event.accept()

        def _new_closeEvent(self, event):
            try:
                save_plugin_window_state(self, self._persist_plugin_name)
            except Exception:
                pass
            if orig_close is not None:
                orig_close(self, event)
            else:
                event.accept()

        cls.__init__ = _new_init
        cls.showEvent = _new_showEvent
        cls.closeEvent = _new_closeEvent
        return cls

    return decorator

