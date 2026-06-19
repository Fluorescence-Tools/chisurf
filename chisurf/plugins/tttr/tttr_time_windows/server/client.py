"""ZMQ client convenience wrapper for Time Window Bins services."""

from __future__ import annotations

from typing import Any

from chisurf.server.transport.zmq import ZmqClient


class TimeWindowClient:
    """Client wrapper for Time Window Bins ZMQ services."""

    def __init__(
        self,
        cmd_port: int = 9765,
        pub_port: int = 9766,
        host: str = "127.0.0.1",
        timeout_ms: int = 5000,
    ) -> None:
        """Initialize the Time Window Bins ZMQ client.

        Parameters
        ----------
        cmd_port : int, default=9765
            Command socket port.
        pub_port : int, default=9766
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

    def analyze_files(
        self,
        files: list[str],
        time_window_ms: float = 10.0,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        """Split TTTR files into time-window BIDs."""
        return self._call(
            "tttr_time_windows.jobs.analyze_files",
            {
                "files": files,
                "time_window_ms": time_window_ms,
                "output_dir": output_dir,
            },
        )

    def _call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Call a ZMQ service and unwrap results."""
        response = self._client.call(method, params)
        if "result" not in response:
            return response
        result = response["result"]
        if isinstance(result, dict) and "ok" in result and "result" in result:
            return result["result"]
        return result

    def close(self) -> None:
        """Close the underlying ZMQ sockets."""
        self._client.close()
