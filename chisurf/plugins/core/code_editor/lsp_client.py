from __future__ import annotations

import json
import pathlib
import shutil
import sys
from collections.abc import Callable

from qtpy import QtCore


class PythonLspClient(QtCore.QObject):
    """Small JSON-RPC client for ``python-lsp-server`` over stdio."""

    status_changed = QtCore.Signal(str)
    diagnostics_received = QtCore.Signal(str, list)

    def __init__(
        self,
        root_path: str | pathlib.Path | None = None,
        command: list[str] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.root_path = pathlib.Path(root_path or pathlib.Path.cwd()).resolve()
        self.command = command or self._default_command()
        self.process: QtCore.QProcess | None = None
        self._buffer = bytearray()
        self._next_id = 1
        self._callbacks: dict[int, Callable[[object], None]] = {}
        self._opened_documents: set[str] = set()
        self._ready = False

    @property
    def is_ready(self) -> bool:
        """Return whether the server completed initialization."""
        return self._ready

    @staticmethod
    def _default_command() -> list[str] | None:
        """Return the best available command for starting pylsp."""
        pylsp = shutil.which("pylsp")
        if pylsp:
            return [pylsp]
        return [sys.executable, "-m", "pylsp"]

    def start(self) -> bool:
        """Start the language server process."""
        if self.process is not None:
            return True
        if not self.command:
            self.status_changed.emit("LSP unavailable")
            return False

        self.process = QtCore.QProcess(self)
        self.process.setProgram(self.command[0])
        self.process.setArguments(self.command[1:])
        self.process.setWorkingDirectory(str(self.root_path))
        self.process.readyReadStandardOutput.connect(self._read_stdout)
        self.process.readyReadStandardError.connect(self._read_stderr)
        self.process.errorOccurred.connect(self._on_process_error)
        self.process.finished.connect(self._on_process_finished)
        self.process.start()
        if not self.process.waitForStarted(1500):
            self.status_changed.emit("LSP unavailable")
            self.process.deleteLater()
            self.process = None
            return False

        self.status_changed.emit("LSP starting")
        self.request(
            "initialize",
            {
                "processId": None,
                "rootUri": self.root_path.as_uri(),
                "capabilities": {},
            },
            self._on_initialized,
        )
        return True

    def stop(self) -> None:
        """Shutdown the LSP process if it is running."""
        if self.process is None:
            return
        try:
            self.request("shutdown", {}, lambda _result: self.notify("exit", {}))
        except Exception:
            pass
        self.process.kill()
        self.process.deleteLater()
        self.process = None
        self._ready = False
        self.status_changed.emit("LSP stopped")

    def request(
        self,
        method: str,
        params: dict | list | None = None,
        callback: Callable[[object], None] | None = None,
    ) -> int:
        """Send a JSON-RPC request and return its id."""
        request_id = self._next_id
        self._next_id += 1
        if callback is not None:
            self._callbacks[request_id] = callback
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}})
        return request_id

    def notify(self, method: str, params: dict | list | None = None) -> None:
        """Send a JSON-RPC notification."""
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def open_document(self, path: str, text: str, language_id: str = "python") -> None:
        """Notify the server that a document is open."""
        uri = pathlib.Path(path).resolve().as_uri()
        if uri in self._opened_documents:
            self.change_document(path, text)
            return
        self._opened_documents.add(uri)
        self.notify(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": uri,
                    "languageId": language_id,
                    "version": 1,
                    "text": text,
                }
            },
        )

    def change_document(self, path: str, text: str, version: int = 1) -> None:
        """Notify the server that a document changed."""
        uri = pathlib.Path(path).resolve().as_uri()
        if uri not in self._opened_documents:
            self.open_document(path, text)
            return
        self.notify(
            "textDocument/didChange",
            {
                "textDocument": {"uri": uri, "version": version},
                "contentChanges": [{"text": text}],
            },
        )

    def close_document(self, path: str) -> None:
        """Notify the server that a document closed."""
        uri = pathlib.Path(path).resolve().as_uri()
        if uri not in self._opened_documents:
            return
        self._opened_documents.remove(uri)
        self.notify("textDocument/didClose", {"textDocument": {"uri": uri}})

    def definition(
        self,
        path: str,
        line: int,
        character: int,
        callback: Callable[[object], None],
    ) -> None:
        """Request the definition location for a document position."""
        self.request(
            "textDocument/definition",
            {
                "textDocument": {"uri": pathlib.Path(path).resolve().as_uri()},
                "position": {"line": max(0, line - 1), "character": max(0, character)},
            },
            callback,
        )

    def document_symbols(self, path: str, callback: Callable[[object], None]) -> None:
        """Request symbols for a document."""
        self.request(
            "textDocument/documentSymbol",
            {"textDocument": {"uri": pathlib.Path(path).resolve().as_uri()}},
            callback,
        )

    def completion(
        self,
        path: str,
        line: int,
        character: int,
        callback: Callable[[object], None],
    ) -> None:
        """Request completions for a document position."""
        self.request(
            "textDocument/completion",
            {
                "textDocument": {"uri": pathlib.Path(path).resolve().as_uri()},
                "position": {"line": max(0, line - 1), "character": max(0, character)},
            },
            callback,
        )

    def hover(
        self,
        path: str,
        line: int,
        character: int,
        callback: Callable[[object], None],
    ) -> None:
        """Request hover text for a document position."""
        self.request(
            "textDocument/hover",
            {
                "textDocument": {"uri": pathlib.Path(path).resolve().as_uri()},
                "position": {"line": max(0, line - 1), "character": max(0, character)},
            },
            callback,
        )

    def _send(self, payload: dict) -> None:
        if self.process is None:
            return
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        self.process.write(header + body)

    def _read_stdout(self) -> None:
        if self.process is None:
            return
        self._buffer.extend(bytes(self.process.readAllStandardOutput()))
        while True:
            header_end = self._buffer.find(b"\r\n\r\n")
            if header_end < 0:
                return
            header = bytes(self._buffer[:header_end]).decode("ascii", errors="replace")
            length = 0
            for line in header.split("\r\n"):
                key, _, value = line.partition(":")
                if key.lower() == "content-length":
                    length = int(value.strip())
                    break
            if length <= 0 or len(self._buffer) < header_end + 4 + length:
                return
            start = header_end + 4
            message = bytes(self._buffer[start:start + length])
            del self._buffer[:start + length]
            self._handle_message(json.loads(message.decode("utf-8")))

    def _read_stderr(self) -> None:
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace").strip()
        if text:
            self.status_changed.emit(f"LSP: {text.splitlines()[-1]}")

    def _handle_message(self, message: dict) -> None:
        if "id" in message:
            callback = self._callbacks.pop(int(message["id"]), None)
            if callback is not None:
                callback(message.get("result"))
            return
        if message.get("method") == "textDocument/publishDiagnostics":
            params = message.get("params", {})
            self.diagnostics_received.emit(params.get("uri", ""), params.get("diagnostics", []))

    def _on_initialized(self, _result: object) -> None:
        self._ready = True
        self.notify("initialized", {})
        self.status_changed.emit("LSP ready")

    def _on_process_error(self, _error: QtCore.QProcess.ProcessError) -> None:
        self._ready = False
        self.status_changed.emit("LSP unavailable")

    def _on_process_finished(self, _exit_code: int, _status: QtCore.QProcess.ExitStatus) -> None:
        self._ready = False
        self.process = None
        self.status_changed.emit("LSP stopped")


__all__ = ["PythonLspClient"]
