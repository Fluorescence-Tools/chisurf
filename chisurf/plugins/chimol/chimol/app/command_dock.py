from __future__ import annotations

from typing import Optional
from pathlib import Path
import os
import re

from qtpy import QtCore, QtWidgets, QtGui

try:
    from ..cmd import cmd as _cmd
except Exception:
    _cmd = None


def _set_hidden_on_windows(path: Path) -> None:
    if os.name != "nt":
        return
    try:
        import ctypes

        FILE_ATTRIBUTE_HIDDEN = 0x2
        GetFileAttributesW = ctypes.windll.kernel32.GetFileAttributesW
        SetFileAttributesW = ctypes.windll.kernel32.SetFileAttributesW
        GetFileAttributesW.argtypes = [ctypes.c_wchar_p]
        GetFileAttributesW.restype = ctypes.c_uint32
        SetFileAttributesW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32]
        SetFileAttributesW.restype = ctypes.c_int
        attrs = GetFileAttributesW(str(path))
        if attrs == 0xFFFFFFFF:
            return
        SetFileAttributesW(str(path), attrs | FILE_ATTRIBUTE_HIDDEN)
    except Exception:
        pass


def _resolve_history_path():
    try:
        try:
            import chisurf.core.settings as _cs_settings
        except Exception:
            _cs_settings = None
        if _cs_settings is not None:
            base = _cs_settings.get_path("settings")
            filename = "chimol_cmd_history.txt"
            force_hidden = False
            legacy = base / "molview_cmd_history.txt"
        else:
            base = Path.home()
            filename = ".chimol_cmd_history"
            force_hidden = True
            legacy = base / ".molview_cmd_history"
        path = Path(base) / filename
        if not path.exists() and legacy.exists():
            path = legacy
        return path, force_hidden
    except Exception:
        return None, False


# ---------------------------------------------------------------------------
# Completion data
# ---------------------------------------------------------------------------

_REP_NAMES = [
    "cartoon", "sticks", "atoms", "dots", "surface",
    "ca_trace", "lines", "spheres", "metaball", "plane", "all",
]
_COLOR_NAMES = [
    "red", "green", "blue", "yellow", "cyan", "magenta",
    "white", "black", "gray", "orange",
    "single", "by_residue", "by_ss", "by_sequence",
    "byelement", "bychain", "spectrum",
]
_SETTING_NAMES = [
    "bg_color", "color_mode", "line_width", "stick_radius",
    "sphere_scale", "cartoon_transparency", "surface_type",
    "metaball_alpha", "metaball_shininess", "metaball_threshold",
    "metaball_resolution", "metaball_radius", "metaball_padding",
]

_REP_COMMANDS = {"show", "hide", "as"}
_COLOR_COMMANDS = {"color", "bg_color", "bg_colour"}
_SET_COMMANDS = {"set", "get"}
_OBJECT_COMMANDS = {
    "select", "delete", "enable", "disable",
    "show", "hide", "color", "center", "zoom",
}


def _get_completion_pool(cmd_name: str) -> list[str]:
    """Return the argument-completion pool for a given chimol command."""
    pool = []
    if cmd_name in _REP_COMMANDS:
        pool.extend(_REP_NAMES)
    if cmd_name in _COLOR_COMMANDS:
        pool.extend(_COLOR_NAMES)
    if cmd_name in _SET_COMMANDS:
        pool.extend(_SETTING_NAMES)
    if cmd_name in _OBJECT_COMMANDS and _cmd is not None:
        try:
            _, viewer = _cmd._require_window_and_viewer()
            if viewer is not None:
                for obj in viewer.list_objects():
                    if "name" in obj:
                        pool.append(obj["name"])
        except Exception:
            pass
    # deduplicate preserving order
    seen = set()
    result = []
    for item in sorted(pool):
        key = item.lower()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _get_command_names() -> list[str]:
    """Return sorted list of registered chimol command names."""
    if _cmd is not None and hasattr(_cmd, "_commands"):
        return sorted(str(n) for n in _cmd._commands.keys())
    return []


# ---------------------------------------------------------------------------
# CommandDock
# ---------------------------------------------------------------------------

class CommandDock(QtCore.QObject):
    commandEntered = QtCore.Signal(str)

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        *,
        margins: tuple[int, int, int, int],
        spacing: int,
    ) -> None:
        super().__init__(parent)
        self._dock = QtWidgets.QDockWidget("Command", parent)
        self._dock.setObjectName("ChimolCommandDock")
        self._dock.setAllowedAreas(
            QtCore.Qt.BottomDockWidgetArea | QtCore.Qt.TopDockWidgetArea
        )
        self._dock.setMinimumHeight(10)

        container = QtWidgets.QWidget(parent)
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(*margins)
        layout.setSpacing(spacing)

        mono_font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        if mono_font.pixelSize() < 12:
            mono_font.setPointSize(12)

        self._output = QtWidgets.QPlainTextEdit(container)
        self._output.setReadOnly(True)
        self._output.setMinimumHeight(10)
        self._output.setFont(mono_font)
        self._output.setSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding
        )

        self._input = QtWidgets.QLineEdit(container)
        self._input.setPlaceholderText("Command line")
        self._input.setFont(mono_font)
        self._input.setMinimumHeight(18)
        self._input.setSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed
        )

        # ---- Completer setup ----
        self._completer_model = QtCore.QStringListModel()
        self._completer = QtWidgets.QCompleter(self._completer_model, container)
        self._completer.setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
        self._completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        self._completer.setFilterMode(QtCore.Qt.MatchContains)
        self._completer.setWidget(self._input)
        self._completer.activated[str].connect(self._insert_completion)
        
        # Set mono font and grid-like layout for the popup
        popup = self._completer.popup()
        if popup:
            popup.setFont(mono_font)
            # Make it a multi-column grid if possible
            # We use a QListView with IconMode for grid layout
            popup.setViewMode(QtWidgets.QListView.IconMode)
            popup.setResizeMode(QtWidgets.QListView.Adjust)
            popup.setMovement(QtWidgets.QListView.Static)
            popup.setFlow(QtWidgets.QListView.LeftToRight)
            popup.setWrapping(True)
            popup.setSpacing(2)
            # Ensure items have a reasonable width
            popup.setGridSize(QtCore.QSize(120, 20))
            
        self._input.textEdited.connect(self._on_text_edited)

        # Install event filter on *both* the input and the popup so we
        # can intercept Tab and Enter uniformly.
        self._input.installEventFilter(self)

        layout.addWidget(self._output, 1)
        layout.addWidget(self._input, 0)

        self._dock.setWidget(container)

        self._history: list[str] = []
        self._history_index: Optional[int] = None
        self._history_path, self._history_force_hidden = _resolve_history_path()
        self._load_history()

    # ------------------------------------------------------------------
    # Completer helpers
    # ------------------------------------------------------------------

    def _popup_visible(self) -> bool:
        popup = self._completer.popup()
        return popup is not None and popup.isVisible()

    def _insert_completion(self, completion_text: str) -> None:
        """Called when user clicks or presses Enter on a popup item.

        The ``completion_text`` is the raw model string (e.g. "cartoon").
        We need to splice it into the current line, replacing only the
        last token.
        """
        text = self._input.text()
        last_sep = max(text.rfind(" "), text.rfind(","))
        if last_sep >= 0:
            new_text = text[:last_sep + 1] + completion_text
        else:
            new_text = completion_text
        self._input.setText(new_text)
        self._input.setCursorPosition(len(new_text))

    def _on_text_edited(self, text: str) -> None:
        """Re-populate the model and show the popup as the user types."""
        parts = re.split(r"[\s,]+", text.lstrip())

        if not parts or not parts[0]:
            self._completer_model.setStringList([])
            return

        cmd_name = parts[0].lower()

        # Phase 1: typing the command name
        if len(parts) <= 1 and not text.endswith((" ", ",")):
            names = _get_command_names()
            self._completer_model.setStringList(names)
            prefix = parts[0]
        else:
            # Phase 2: typing arguments
            pool = _get_completion_pool(cmd_name)
            self._completer_model.setStringList(pool)
            # Prefix is the last (possibly empty) token
            if text.endswith((" ", ",")):
                prefix = ""
            else:
                prefix = parts[-1]

        self._completer.setCompletionPrefix(prefix)
        if self._completer.completionCount() > 0:
            self._completer.complete()
        else:
            self._completer.popup().hide()

    # ------------------------------------------------------------------
    # Event filter — single place for all keyboard interaction
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event):
        if obj is not self._input or event.type() != QtCore.QEvent.KeyPress:
            return super().eventFilter(obj, event)

        key = event.key()
        popup_up = self._popup_visible()

        # ---- Enter / Return ----
        if key in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            if popup_up:
                # Accept the highlighted popup item
                index = self._completer.popup().currentIndex()
                if index.isValid():
                    completion = self._completer_model.data(index, QtCore.Qt.DisplayRole)
                    self._insert_completion(completion)
                self._completer.popup().hide()
                return True  # consumed — do NOT execute the command
            else:
                # No popup → execute the command line
                self._execute_current_line()
                return True

        # ---- Tab ----
        if key == QtCore.Qt.Key_Tab and not event.modifiers():
            if popup_up:
                index = self._completer.popup().currentIndex()
                if index.isValid():
                    completion = self._completer_model.data(index, QtCore.Qt.DisplayRole)
                    self._insert_completion(completion)
                self._completer.popup().hide()
                return True
            else:
                # Force-trigger the completer (like pressing a character)
                self._on_text_edited(self._input.text())
                return True

        # ---- Up / Down when popup is NOT visible → history ----
        if not popup_up:
            if key == QtCore.Qt.Key_Up:
                self._navigate_history(-1)
                return True
            if key == QtCore.Qt.Key_Down:
                self._navigate_history(1)
                return True

        # ---- Escape ----
        if key == QtCore.Qt.Key_Escape and popup_up:
            self._completer.popup().hide()
            return True

        return super().eventFilter(obj, event)

    # ------------------------------------------------------------------
    # Command execution
    # ------------------------------------------------------------------

    def _execute_current_line(self) -> None:
        text = self._input.text().strip()
        if not text:
            return
        self._append_line("> " + text)
        self._add_to_history(text)
        self._input.clear()
        self.commandEntered.emit(text)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    @property
    def dock_widget(self) -> QtWidgets.QDockWidget:
        return self._dock

    def clear(self) -> None:
        self._output.clear()

    def append_message(self, text: str) -> None:
        if not text:
            return
        self._append_line(text)

    def append_error(self, text: str) -> None:
        if not text:
            return
        self._append_line(text)

    def _append_line(self, text: str) -> None:
        cursor = self._output.textCursor()
        cursor.movePosition(cursor.End)
        if self._output.toPlainText():
            cursor.insertBlock()
        cursor.insertText(text)
        self._output.setTextCursor(cursor)
        self._output.ensureCursorVisible()

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def _load_history(self) -> None:
        path = getattr(self, "_history_path", None)
        history = []
        if path is not None:
            try:
                if path.exists():
                    with path.open("rt", encoding="utf-8") as fh:
                        for line in fh:
                            line = line.rstrip("\n")
                            if line:
                                history.append(line)
            except Exception:
                history = []
        if len(history) > 500:
            history = history[-500:]
        self._history = history
        self._history_index = None

    def _save_history(self) -> None:
        path = getattr(self, "_history_path", None)
        data = getattr(self, "_history", None)
        if path is None or not data:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wt", encoding="utf-8") as fh:
                for line in data[-500:]:
                    fh.write(line + "\n")
            if getattr(self, "_history_force_hidden", False):
                _set_hidden_on_windows(path)
        except Exception:
            pass

    def _add_to_history(self, text: str) -> None:
        if not text:
            return
        if self._history and self._history[-1] == text:
            self._history_index = None
            return
        self._history.append(text)
        if len(self._history) > 500:
            self._history = self._history[-500:]
        self._history_index = None
        self._save_history()

    def _navigate_history(self, step: int) -> None:
        if not self._history:
            return
        if self._history_index is None:
            if step < 0:
                self._history_index = len(self._history) - 1
            else:
                return
        else:
            self._history_index += step
            if self._history_index < 0:
                self._history_index = 0
            elif self._history_index >= len(self._history):
                self._history_index = None
                self._input.clear()
                return
        if self._history_index is not None:
            text = self._history[self._history_index]
            self._input.setText(text)
            self._input.setCursorPosition(len(text))
