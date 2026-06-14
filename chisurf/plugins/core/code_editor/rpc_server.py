from __future__ import annotations

import queue
from typing import Any

from chisurf.plugins.core.code_editor.backend.services import dispatch
from chisurf.plugins.core.code_editor.document_store import DocumentStore
from chisurf.plugins.core.code_editor.ruff_runner import RuffRunner
from chisurf.server.transport.zmq import ZmqServer


class EditorRpcServer:
    """ZeroMQ JSON-RPC server for the GUI-owned editor document store."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        cmd_port: int = 8775,
        pub_port: int = 8776,
        store: DocumentStore | None = None,
        runner: RuffRunner | None = None,
    ) -> None:
        self.host = host
        self.cmd_port = cmd_port
        self.pub_port = pub_port
        self.store = store
        self.runner = runner
        self._events: queue.Queue[dict[str, Any]] = queue.Queue()
        self._server = ZmqServer(
            handler=self._dispatch,
            cmd_port=cmd_port,
            pub_port=pub_port,
            host=host,
        )
        self._thread = None

    def start(self) -> None:
        """Start the editor RPC server in a daemon thread."""
        if self._thread is not None:
            return
        import threading

        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the editor RPC server."""
        self._server.stop()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def drain_events(self) -> list[dict[str, Any]]:
        """Return queued document-change events."""
        events: list[dict[str, Any]] = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                return events

    def _dispatch(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        result = dispatch(method, params, store=self.store, runner=self.runner)
        event = result.pop("event", None)
        if isinstance(event, dict):
            self._events.put(event)
        return result
