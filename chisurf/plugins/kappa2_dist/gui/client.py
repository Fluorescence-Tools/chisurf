"""PluginClient wrapper for the k² Distribution Calculator.

GUI code should use this client instead of importing core directly.
"""

from __future__ import annotations

from typing import Any

from chisurf.core.plugin.client import InProcessClient


class Kappa2DistClient:
    """Client for the k² Distribution Calculator backend services."""

    def __init__(self, client: Any = None):
        if client is not None:
            self._client = client
        else:
            self._client = self._make_local_client()

    def compute(self, **params: Any) -> dict[str, Any]:
        """Compute k² distribution.

        Parameters
        ----------
        **params
            Forwarded to ``kappa2_dist.compute`` RPC method.

        Returns
        -------
        dict
            ``{"ok": True, "result": {...}}`` or ``{"ok": False, "error": ...}``.

        """
        return self._client.call("kappa2_dist.compute", params)

    @staticmethod
    def _make_local_client() -> InProcessClient:
        """Create a local in-process client with kappa2_dist services."""
        from chisurf.server.dispatcher import ServiceDispatcher
        from chisurf.server.session import SessionState

        state = SessionState()
        dispatcher = ServiceDispatcher(state)
        dispatcher._build_default_registry()

        from chisurf.plugins.kappa2_dist.backend.services import (
            register_services,
        )
        register_services(dispatcher)
        return InProcessClient(dispatcher)
