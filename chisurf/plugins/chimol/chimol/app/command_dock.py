from __future__ import annotations

from typing import Optional
from pathlib import Path
import os

from qtpy import QtCore, QtWidgets

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
            import chisurf.settings as _cs_settings
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

        self._output = QtWidgets.QPlainTextEdit(container)
        self._output.setReadOnly(True)
        self._output.setMinimumHeight(10)
        self._output.setSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding
        )

        self._input = QtWidgets.QLineEdit(container)
        self._input.setPlaceholderText("Command line")
        self._input.returnPressed.connect(self._on_return_pressed)
        self._input.installEventFilter(self)
        self._input.setMinimumHeight(18)
        self._input.setSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed
        )

        layout.addWidget(self._output, 1)
        layout.addWidget(self._input, 0)

        self._dock.setWidget(container)

        self._history: list[str] = []
        self._history_index: Optional[int] = None
        self._history_path, self._history_force_hidden = _resolve_history_path()
        self._load_history()

    @property
    def dock_widget(self) -> QtWidgets.QDockWidget:
        return self._dock

    def eventFilter(self, obj, event):
        if obj is self._input and event.type() == QtCore.QEvent.KeyPress:
            key = event.key()
            if key == QtCore.Qt.Key_Up:
                self._navigate_history(-1)
                return True
            if key == QtCore.Qt.Key_Down:
                self._navigate_history(1)
                return True
            if key == QtCore.Qt.Key_Tab and not event.modifiers():
                self._apply_completion()
                return True
        return super().eventFilter(obj, event)

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

    def _on_return_pressed(self) -> None:
        text = self._input.text().strip()
        if not text:
            return
        self._append_line("> " + text)
        self._add_to_history(text)
        self._input.clear()
        self.commandEntered.emit(text)

    def _append_line(self, text: str) -> None:
        cursor = self._output.textCursor()
        cursor.movePosition(cursor.End)
        if self._output.toPlainText():
            cursor.insertBlock()
        cursor.insertText(text)
        self._output.setTextCursor(cursor)
        self._output.ensureCursorVisible()

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

    def _apply_completion(self) -> None:
        raw = self._input.text()
        if not raw:
            return
        cursor_pos = self._input.cursorPosition()
        prefix = raw[:cursor_pos]
        if " " in prefix:
            return
        token = prefix.strip()
        if not token:
            return
        names = []
        cmd_obj = _cmd
        if cmd_obj is not None:
            try:
                commands = getattr(cmd_obj, "_commands", None)
                if isinstance(commands, dict):
                    names = sorted(str(name) for name in commands.keys())
            except Exception:
                names = []
        if not names:
            return
        matches = [name for name in names if name.startswith(token)]
        if not matches:
            return
        if len(matches) == 1:
            completed = matches[0] + " "
            rest = raw[cursor_pos:]
            self._input.setText(completed + rest)
            self._input.setCursorPosition(len(completed))
            return
        common = matches[0]
        for name in matches[1:]:
            i = 0
            limit = min(len(common), len(name))
            while i < limit and common[i] == name[i]:
                i += 1
            common = common[:i]
            if not common:
                break
        if common and common != token:
            new_text = common + raw[cursor_pos:]
            self._input.setText(new_text)
            self._input.setCursorPosition(len(common))
        else:
            self._append_line(" ".join(matches))
