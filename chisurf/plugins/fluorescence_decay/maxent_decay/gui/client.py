"""In-process client wrapper for MaxEnt MEM services."""

from __future__ import annotations

from typing import Any

from chisurf.core.plugin.client import InProcessClient

from ..api.contract import METHOD_DESCRIBE, METHOD_RUN_FRET, METHOD_RUN_LCURVE, METHOD_RUN_LIFETIME


class MaxEntClient:
    """Client for MaxEnt MEM backend services."""

    def __init__(self, client: Any | None = None) -> None:
        """Create a MaxEnt client."""
        self._client = client or self._make_local_client()

    def _call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        response = self._client.call(method, params)
        result = response.get("result", {})
        if isinstance(result, dict) and "ok" in result and "result" in result:
            return result["result"]
        return result

    def run_lifetime_mem(self, decay: list[float], irf: list[float], dt: float, **kwargs: Any) -> dict[str, Any]:
        """Run lifetime MEM."""
        return self._call(METHOD_RUN_LIFETIME, {"decay": decay, "irf": irf, "dt": dt, **kwargs})

    def run_fret_mem(self, decay: list[float], irf: list[float], dt: float, **kwargs: Any) -> dict[str, Any]:
        """Run FRET MEM."""
        return self._call(METHOD_RUN_FRET, {"decay": decay, "irf": irf, "dt": dt, **kwargs})

    def run_lcurve(self, decay: list[float], irf: list[float], dt: float, **kwargs: Any) -> dict[str, Any]:
        """Run an L-curve sweep."""
        return self._call(METHOD_RUN_LCURVE, {"decay": decay, "irf": irf, "dt": dt, **kwargs})

    def describe_contract(self) -> dict[str, Any]:
        """Return the workflow contract."""
        return self._call(METHOD_DESCRIBE, {})

    @staticmethod
    def _make_local_client() -> InProcessClient:
        """Create a local in-process client with MaxEnt services registered."""
        from chisurf.server.dispatcher import ServiceDispatcher
        from chisurf.server.session import SessionState

        from ..backend.services import register_services

        state = SessionState()
        dispatcher = ServiceDispatcher(state)
        dispatcher._build_default_registry()
        register_services(dispatcher)
        return InProcessClient(dispatcher)
