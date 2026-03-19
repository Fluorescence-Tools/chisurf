from __future__ import annotations

from pathlib import Path
from shlex import split as shlex_split
from typing import Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..app.molview_main_window import MolViewPluginWindow


MessageCallback = Callable[[str], None]


class BaseCmd:
    """Shared infrastructure for the Moview/Chimol command layer."""

    def __init__(self, window: Optional["MolViewPluginWindow"] = None) -> None:
        self.window = window
        self._message_callback: Optional[MessageCallback] = None
        self._error_callback: Optional[MessageCallback] = None
        self._commands: Dict[str, Callable[[List[str]], object]] = {}
        self._named_selections: Dict[str, Dict[str, object]] = {}
        self._install_builtin_commands()

    # ------------------------------------------------------------------ #
    # Public API and core plumbing
    # ------------------------------------------------------------------ #
    def set_window(self, window: Optional["MolViewPluginWindow"]) -> None:
        self.window = window

    def set_message_callback(self, callback: Optional[MessageCallback]) -> None:
        self._message_callback = callback

    def set_error_callback(self, callback: Optional[MessageCallback]) -> None:
        self._error_callback = callback

    def register(self, name: str, func: Callable[[List[str]], object]) -> None:
        self._commands[name.lower()] = func

    def do(self, line: str) -> None:
        line = (line or "").strip()
        if not line:
            return

        # Script execution: '@filename' runs a text file with one command per line.
        if line.startswith("@"):
            script_path = line[1:].strip()
            if not script_path:
                self._emit_error("Usage: @<script_file>")
                return
            self._run_script_file(script_path)
            return

        try:
            parts = shlex_split(line)
        except Exception as exc:
            self._emit_error(f"Parse error: {exc}")
            return

        if not parts:
            return

        name = parts[0].lower()
        args = parts[1:]

        handler = self._commands.get(name)
        if handler is None:
            self._emit_error(
                f"Command '{name}' is not implemented in Moview/MolView cmd (PyMOL compatibility layer)."
            )
            return

        try:
            result = handler(args)
        except Exception as exc:
            self._emit_error(f"Error in command '{name}': {exc}")
            return

        if result is not None:
            self._emit_message(str(result))

    def _run_script_file(self, path: str) -> None:
        """Execute a simple cmd script file, one command per non-empty line.

        Lines starting with '#' are treated as comments and skipped. This is
        similar in spirit to PyMOL's '@script.pml' support, but limited to the
        Moview cmd language (no arbitrary Python execution).
        """

        try:
            p = Path(path).expanduser()
        except Exception as exc:
            self._emit_error(f"Invalid script path {path!r}: {exc}")
            return

        if not p.exists():
            self._emit_error(f"Script file not found: {p}")
            return

        try:
            with p.open("rt", encoding="utf-8") as fh:
                for raw in fh:
                    line = (raw or "").strip()
                    if not line or line.startswith("#"):
                        continue
                    self.do(line)
        except Exception as exc:
            self._emit_error(f"Failed to run script {p!s}: {exc}")

    # ------------------------------------------------------------------ #
    # Builtins registration
    # ------------------------------------------------------------------ #
    def _builtin_commands(self) -> Dict[str, Callable[[List[str]], object]]:
        """Mixins extend this to advertise the commands they handle."""
        return {}

    def _install_builtin_commands(self) -> None:
        for name, func in self._builtin_commands().items():
            self.register(name, func)

    # ------------------------------------------------------------------ #
    # Shared helpers
    # ------------------------------------------------------------------ #
    def _require_window_and_viewer(self):
        window = self.window
        if window is None:
            self._emit_error("No viewer window is attached")
            return None, None
        viewer = getattr(window, "viewer", None)
        if viewer is None:
            self._emit_error("Attached window has no 'viewer' attribute")
            return window, None
        return window, viewer

    def _find_object_by_name(self, viewer, name: str) -> Optional[dict]:
        target = (name or "").strip().lower()
        if not target:
            return None
        try:
            objects = viewer.list_objects()
        except Exception:
            return None
        for obj in objects:
            obj_id = str(obj.get("id", ""))
            obj_name = str(obj.get("name", ""))
            if not obj_id and not obj_name:
                continue
            if target == obj_id.lower() or target == obj_name.lower():
                return obj
        return None

    def _emit_message(self, text: str) -> None:
        callback = self._message_callback
        if callback is not None:
            callback(text)

    def _emit_error(self, text: str) -> None:
        callback = self._error_callback or self._message_callback
        if callback is not None:
            callback(text)

    # ------------------------------------------------------------------ #
    # Common command
    # ------------------------------------------------------------------ #
    def _cmd_help(self, args: List[str]) -> str:
        """Show available commands or detailed help for a specific command."""
        if not args:
            names = sorted(self._commands.keys())
            return "Available commands: " + ", ".join(names)

        target = args[0].lower()
        handler = self._commands.get(target)
        if handler is None:
            return f"No help available for unknown command: {target}"

        doc = getattr(handler, "__doc__", None)
        if not doc:
            return f"No detailed help available for '{target}'"

        return f"Help for '{target}':\n" + "-" * 20 + "\n" + doc.strip()

    def _cmd_quit(self, args: List[str]) -> None:
        window = self.window
        if window is None:
            return
        try:
            window.close()
        except Exception:
            pass
