"""ZMQ client convenience wrapper for MaxEnt MEM services."""

from __future__ import annotations

from typing import Any

from chisurf.server.transport.zmq import ZmqClient

from ..api.contract import METHOD_DESCRIBE, METHOD_RUN_FRET, METHOD_RUN_LCURVE, METHOD_RUN_LIFETIME


class MaxEntZmqClient:
    """Client wrapper for remote MaxEnt MEM ZMQ services."""

    def __init__(
        self,
        cmd_port: int = 8780,
        pub_port: int = 8781,
        host: str = "127.0.0.1",
        timeout_ms: int = 30000,
    ) -> None:
        """Initialize the MaxEnt MEM ZMQ client."""
        self._client = ZmqClient(
            cmd_port=cmd_port,
            pub_port=pub_port,
            host=host,
            timeout_ms=timeout_ms,
        )

    def _call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        response = self._client.call(method, params)
        result = response.get("result", {})
        if isinstance(result, dict) and "ok" in result and "result" in result:
            return result["result"]
        return result

    def run_lifetime_mem(self, decay: list[float], irf: list[float], dt: float, **kwargs: Any) -> dict[str, Any]:
        """Run lifetime MEM on a remote server."""
        return self._call(METHOD_RUN_LIFETIME, {"decay": decay, "irf": irf, "dt": dt, **kwargs})

    def run_fret_mem(self, decay: list[float], irf: list[float], dt: float, **kwargs: Any) -> dict[str, Any]:
        """Run FRET MEM on a remote server."""
        return self._call(METHOD_RUN_FRET, {"decay": decay, "irf": irf, "dt": dt, **kwargs})

    def run_lcurve(self, decay: list[float], irf: list[float], dt: float, **kwargs: Any) -> dict[str, Any]:
        """Run an L-curve sweep on a remote server."""
        return self._call(METHOD_RUN_LCURVE, {"decay": decay, "irf": irf, "dt": dt, **kwargs})

    def describe_contract(self) -> dict[str, Any]:
        """Return the MaxEnt MEM workflow contract."""
        return self._call(METHOD_DESCRIBE, {})

    def close(self) -> None:
        """Close the underlying ZMQ sockets."""
        self._client.close()
