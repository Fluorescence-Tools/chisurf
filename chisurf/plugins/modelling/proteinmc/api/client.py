"""ZMQ JSON-RPC client for the ProteinMC API."""

from __future__ import annotations

from typing import Any, Callable, Optional

from chisurf.server.transport.zmq import ZmqClient


class ProteinMCApiClient:
    """Client for :class:`ProteinMCApiServer`.

    Parameters
    ----------
    host : str
        Server hostname.
    cmd_port : int
        REQ/REP command port.
    pub_port : int
        PUB event port.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        cmd_port: int = 9765,
        pub_port: int = 9766,
    ):
        self._client = ZmqClient(host=host, cmd_port=cmd_port, pub_port=pub_port)
        self._connected = False

    def connect(self) -> None:
        """Connect to the ProteinMC API server."""
        self._client.connect()
        self._connected = True

    def run(
        self,
        structure_source: str,
        *,
        flexfit_set: Optional[str] = None,
        settings: Optional[dict[str, Any]] = None,
        settings_file: Optional[str] = None,
        output_file: Optional[str] = None,
    ) -> dict[str, Any]:
        """Start a ProteinMC run."""
        params: dict[str, Any] = {
            "structure_source": structure_source,
        }
        if flexfit_set is not None:
            params["flexfit_set"] = flexfit_set
        if settings is not None:
            params["settings"] = settings
        if settings_file is not None:
            params["settings_file"] = settings_file
        if output_file is not None:
            params["output_file"] = output_file
        return self._client.call("run", params)

    def stop(self) -> dict[str, Any]:
        """Request the running simulation to stop."""
        return self._client.call("stop", {})

    def status(self) -> dict[str, Any]:
        """Return current run status."""
        return self._client.call("status", {})

    def get_result(self) -> dict[str, Any]:
        """Return the completed result (if finished)."""
        return self._client.call("get_result", {})

    def subscribe_progress(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Subscribe to progress events."""
        self._client.subscribe("proteinmc.progress", callback)

    def subscribe_finished(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Subscribe to run-finished events."""
        self._client.subscribe("proteinmc.finished", callback)

    def subscribe_failed(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Subscribe to run-failed events."""
        self._client.subscribe("proteinmc.failed", callback)

    def drain(self) -> None:
        """Process buffered subscription events (call from main thread)."""
        self._client.drain()

    def close(self) -> None:
        """Disconnect the client."""
        self._client._req_socket.close()
        self._client._ctx.term()
        self._connected = False
