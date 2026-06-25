"""ZMQ client convenience wrapper for light-path simulator services."""

from __future__ import annotations

from chisurf.server.transport.zmq import ZmqClient

from ..api.client import LightPathClient


class LightPathZmqClient(LightPathClient):
    """Light-path simulator client backed by ChiSurf ZMQ transport."""

    def __init__(
        self,
        cmd_port: int = 8765,
        pub_port: int = 8766,
        host: str = "127.0.0.1",
        timeout_ms: int = 5000,
    ) -> None:
        """Initialize the ZMQ client wrapper."""
        super().__init__(
            ZmqClient(
                cmd_port=cmd_port,
                pub_port=pub_port,
                host=host,
                timeout_ms=timeout_ms,
            )
        )

    def close(self) -> None:
        """Close the underlying ZMQ sockets."""
        self.client.close()
