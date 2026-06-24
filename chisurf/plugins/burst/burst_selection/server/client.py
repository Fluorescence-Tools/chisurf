"""ZMQ client convenience wrapper for Burst Selection services."""

from __future__ import annotations

from typing import Any

from chisurf.server.transport.zmq import ZmqClient

from ..api.contract import (
    METHOD_ANALYZE_FILES,
    METHOD_DESCRIBE_CONTRACT,
    METHOD_FIT_GMM,
    METHOD_INSPECT_BUR,
)


class BurstSelectionClient:
    """Client wrapper for Burst Selection ZMQ services."""

    def __init__(
        self,
        cmd_port: int = 8765,
        pub_port: int = 8766,
        host: str = "127.0.0.1",
        timeout_ms: int = 5000,
    ) -> None:
        """Initialize the Burst Selection ZMQ client.

        Parameters
        ----------
        cmd_port : int, default=8765
            Command socket port.
        pub_port : int, default=8766
            Event socket port.
        host : str, default="127.0.0.1"
            Server host.
        timeout_ms : int, default=5000
            Request timeout in milliseconds.
        """
        self._client = ZmqClient(
            cmd_port=cmd_port,
            pub_port=pub_port,
            host=host,
            timeout_ms=timeout_ms,
        )

    @property
    def client(self) -> ZmqClient:
        """Return the underlying ZMQ client."""
        return self._client

    def _call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Call a ZMQ service and unwrap successful JSON-RPC results."""
        response = self._client.call(method, params)
        if "result" not in response:
            return response
        result = response["result"]
        if isinstance(result, dict) and "ok" in result and "result" in result:
            return result["result"]
        return result

    def analyze_files(self, files: list[str], **kwargs: Any) -> dict[str, Any]:
        """Run Burst Selection analysis over TTTR files."""
        return self._call(METHOD_ANALYZE_FILES, {"files": files, **kwargs})

    def inspect_bur(self, path: str) -> dict[str, Any]:
        """Inspect a ``.bur`` file."""
        return self._call(METHOD_INSPECT_BUR, {"path": path})

    def fit_gmm(
        self,
        path: str,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Fit a GMM to features extracted from a ``.bur`` file."""
        return self._call(
            METHOD_FIT_GMM,
            {"path": path, "settings": settings or {}},
        )

    def describe_contract(self) -> dict[str, Any]:
        """Return the Burst Selection workflow contract."""
        return self._call(METHOD_DESCRIBE_CONTRACT, {})

    def close(self) -> None:
        """Close the underlying ZMQ sockets."""
        self._client.close()
