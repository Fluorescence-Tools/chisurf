"""Transport-agnostic client for the CLSM backend services.

Wraps an :class:`~chisurf.core.plugin.client.InProcessClient` by default (local
dispatcher) or a ``ZmqClient`` for a remote server. The CLI and GUI both go
through this wrapper so they behave identically against local and server modes.
"""

from __future__ import annotations

from typing import Any

from .api import contract


class ClsmClient:
    """Typed convenience wrapper over the ``clsm.*`` RPC methods."""

    def __init__(
        self,
        client: Any = None,
        host: str = "127.0.0.1",
        cmd_port: int = 8765,
        pub_port: int = 8766,
    ) -> None:
        self._host = host
        self._cmd_port = cmd_port
        self._pub_port = pub_port
        self._client = client if client is not None else self._make_local_client()

    # ── transport plumbing ─────────────────────────────────────────────
    @staticmethod
    def _make_local_client() -> Any:
        """Create an in-process client with the CLSM services registered."""
        from chisurf.core.plugin.client import InProcessClient
        from chisurf.server.dispatcher import ServiceDispatcher
        from chisurf.server.session import SessionState

        from .backend.services import register_services

        dispatcher = ServiceDispatcher(SessionState())
        register_services(dispatcher)
        return InProcessClient(dispatcher)

    def make_remote_client(self) -> Any:
        """Switch this client to a ZMQ transport connected to ``host:port``."""
        from chisurf.server.transport.zmq import ZmqClient

        self._client = ZmqClient(
            host=self._host,
            cmd_port=self._cmd_port,
            pub_port=self._pub_port,
        )
        return self._client

    def _call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        result = self._client.call(method, params or {})
        if not result.get("ok", True):
            raise RuntimeError(result.get("error", f"{method} failed"))
        return result.get("result")

    # ── typed methods ──────────────────────────────────────────────────
    def setups(self) -> dict[str, Any]:
        """Return the built-in CLSM setup presets."""
        return self._call(contract.METHOD_SETUPS)

    def info(self, filename: str, **setup_kwargs: Any) -> dict[str, Any]:
        """Return image dimensions and the resolved setup for *filename*."""
        return self._call(contract.METHOD_INFO, {"filename": filename, **setup_kwargs})

    def representation(self, filename: str, **kwargs: Any) -> dict[str, Any]:
        """Compute an image representation (optionally written to disk)."""
        return self._call(contract.METHOD_REPRESENTATION, {"filename": filename, **kwargs})

    def decay(self, filename: str, **kwargs: Any) -> dict[str, Any]:
        """Extract a decay histogram from a pixel selection."""
        return self._call(contract.METHOD_DECAY, {"filename": filename, **kwargs})

    def frc(self, filename: str, **kwargs: Any) -> dict[str, Any]:
        """Compute the Fourier Ring Correlation for an image representation."""
        return self._call(contract.METHOD_FRC, {"filename": filename, **kwargs})

    def contract(self) -> dict[str, Any]:
        """Return the RPC contract descriptor."""
        return self._call(contract.METHOD_CONTRACT)
