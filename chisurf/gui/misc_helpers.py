from __future__ import annotations

import ast
import importlib
import os
import pathlib
from functools import partial

import numpy as np

import chisurf
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
        _ = importlib.import_module("chisurf.models.global_model.globalfit")
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
    """Filter log list widget based on filter text and hide checkbox state."""
    try:
        filter_text = window.lineEdit_LogFilter.text().strip().lower()
    except Exception:
        filter_text = ""
    try:
        hide_non_matching = window.checkBox_filter_hide.isChecked()
    except Exception:
        hide_non_matching = False

    if not hasattr(window, "_original_log_items"):
        window._original_log_items = []
        try:
            for i in range(window.plainTextEditLog.count()):
                window._original_log_items.append(window.plainTextEditLog.item(i).text())
        except Exception:
            window._original_log_items = []

    if not filter_text:
        try:
            window.plainTextEditLog.clear()
            for item_text in window._original_log_items:
                item = QtWidgets.QListWidgetItem(item_text)
                window.plainTextEditLog.addItem(item)
        except Exception:
            pass
        return

    try:
        window.plainTextEditLog.clear()
        if window._original_log_items:
            for item_text in window._original_log_items:
                if filter_text in item_text.lower():
                    item = QtWidgets.QListWidgetItem(item_text)
                    item.setForeground(QtGui.QBrush(QtGui.QColor(0, 0, 0)))
                    item.setBackground(QtGui.QBrush(QtGui.QColor(255, 255, 0, 50)))
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                    window.plainTextEditLog.addItem(item)
                elif not hide_non_matching:
                    item = QtWidgets.QListWidgetItem(item_text)
                    item.setForeground(QtGui.QBrush(QtGui.QColor(150, 150, 150)))
                    window.plainTextEditLog.addItem(item)

            if window.plainTextEditLog.count() == 0:
                window.plainTextEditLog.addItem("No matching log entries found.")
        else:
            window.plainTextEditLog.addItem("No log entries found.")
    except Exception:
        pass


def update_log_filter(window):
    """Update log filtering when new entries are added."""
    try:
        if window.plainTextEditLog.count() > 0:
            latest_item = window.plainTextEditLog.item(window.plainTextEditLog.count() - 1).text()
            if hasattr(window, "_original_log_items"):
                window._original_log_items.append(latest_item)
    except Exception:
        pass

    filter_log_content(window)


def run_macro(filename=None, executor: str = "console", globals=None, locals=None, main_window=None):
    """Run a macro file via console or exec."""
    if filename is None and main_window is not None:
        filename = chisurf.gui.widgets.get_filename("Python macros", file_type="Python file (*.py)")
    if filename is None:
        return
    chisurf.logging.info(f"Running script: {filename}")

    if executor == "console":
        chisurf.console.run_macro(filename=pathlib.Path(filename).as_posix())
        return

    # executor == exec
    if globals is None:
        globals = {
            "__name__": "__main__",
            "chisurf": chisurf,
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

    try:
        # Detect plugin package context for reload behavior
        if str(filename).find("\\plugins\\") > -1:
            parts = str(filename).split("\\plugins\\")
            if len(parts) > 1:
                plugin_path = parts[1].split("\\")
                if len(plugin_path) > 1:
                    module_parts = plugin_path[:-1]
                elif len(plugin_path) == 1:
                    module_parts = plugin_path
                else:
                    module_parts = []

                if module_parts:
                    package_name = ".".join(module_parts)
                    user_plugin_root = pathlib.Path.home() / ".chisurf" / "plugins"
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
                                                    chisurf.logging.info(f"User plugin name: {plugin_name}")
                                                elif isinstance(node.value, ast.Constant) and isinstance(
                                                    node.value.value, str
                                                ):
                                                    plugin_name = node.value.value
                                                    chisurf.logging.info(f"User plugin name: {plugin_name}")
                            except Exception as e:
                                chisurf.logging.warning(f"Error extracting name from {user_plugin_path}: {e}")
                    else:
                        globals.update({"__package__": f"chisurf.plugins.{package_name}"})
                        plugin_module_prefix = f"chisurf.plugins.{package_name}"
                        for module_name in list(sys.modules.keys()):
                            if module_name.startswith(plugin_module_prefix):
                                try:
                                    chisurf.logging.info(f"Reloading module: {module_name}")
                                    importlib.reload(sys.modules[module_name])
                                except Exception as e:
                                    chisurf.logging.warning(f"Failed to reload module {module_name}: {e}")

        # Resolve missing user plugin file
        if not pathlib.Path(filename).exists():
            user_plugin_root = pathlib.Path.home() / ".chisurf" / "plugins"
            if "\\plugins\\" in str(filename):
                parts = str(filename).split("\\plugins\\")
                if len(parts) > 1:
                    plugin_path = parts[1]
                    user_plugin_path = user_plugin_root / plugin_path
                    if user_plugin_path.exists():
                        filename = user_plugin_path
                        chisurf.logging.info(f"Found file in user plugins directory: {filename}")
                    else:
                        chisurf.logging.error(f"File not found: {filename}")
                        chisurf.logging.error(f"Also checked user plugin path: {user_plugin_path}")
                        raise FileNotFoundError(f"File not found: {filename}")
            else:
                chisurf.logging.error(f"File not found: {filename}")
                raise FileNotFoundError(f"File not found: {filename}")

        with open(filename, "rb") as file:
            exec(compile(file.read(), filename, "exec"), globals, locals)
    except Exception as e:
        chisurf.logging.error(f"Error executing macro: {e}")
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
            adr = "https://github.com/fluorescence-tools/chisurf"  # Default value
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
            chisurf.logging.warning(f"No wizard.py or __init__.py found for plugin directory: {plugin_dir_to_use}")
    except Exception as e:
        chisurf.logging.error(f"Error running plugin from {plugin_dir_to_use}: {e}")
