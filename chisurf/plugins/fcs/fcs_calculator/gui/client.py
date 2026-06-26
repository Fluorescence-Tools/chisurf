"""PluginClient wrapper for the FCS confocal calculator."""

from __future__ import annotations

from typing import Any, Dict

from chisurf.core.plugin.client import InProcessClient


class ConfocalCalcClient:
    """Client for the FCS confocal calculator backend services."""

    def __init__(self, client: Any = None):
        self._client = client if client is not None else self._make_local_client()

    def compute(self, **params) -> Dict[str, Any]:
        return self._client.call("fcs_calculator.compute", params)

    def water_viscosity(self, temp_C: float) -> Dict[str, Any]:
        return self._client.call("fcs_calculator.water_viscosity", {"temp_C": temp_C})

    @staticmethod
    def _make_local_client() -> InProcessClient:
        from chisurf.server.dispatcher import ServiceDispatcher
        from chisurf.server.session import SessionState

        from ..backend.services import register_services

        state = SessionState()
        dispatcher = ServiceDispatcher(state)
        register_services(dispatcher)
        return InProcessClient(dispatcher)
