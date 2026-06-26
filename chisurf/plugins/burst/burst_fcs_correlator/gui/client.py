"""PluginClient wrapper for the burst-wise FCS correlator.

GUI code should use this client instead of importing ``..core`` directly, so the
same calls work in-process or against a remote server.
"""

from __future__ import annotations

from typing import Any, Dict, List

from chisurf.core.plugin.client import InProcessClient


class BurstFcsClient:
    """Client for the burst-FCS backend services."""

    def __init__(self, client: Any = None):
        self._client = client if client is not None else self._make_local_client()

    # -- RPC wrappers ---------------------------------------------------
    def parse_bst(self, path: str) -> Dict[str, Any]:
        return self._client.call("burst_fcs.parse_bst", {"path": str(path)})

    def parse_bur(self, path: str, analysis_root: str = None) -> Dict[str, Any]:
        return self._client.call(
            "burst_fcs.parse_bur", {"path": str(path), "analysis_root": analysis_root}
        )

    def fit_curve(self, tau, g, settings: Dict[str, Any] = None) -> Dict[str, Any]:
        return self._client.call(
            "burst_fcs.fit_curve",
            {"tau": list(tau), "g": list(g), "settings": settings or {}},
        )

    def fit_simple(self, tau, g) -> Dict[str, Any]:
        return self._client.call("burst_fcs.fit_simple", {"tau": list(tau), "g": list(g)})

    def fit_diffusion(self, tau, g) -> Dict[str, Any]:
        return self._client.call("burst_fcs.fit_diffusion", {"tau": list(tau), "g": list(g)})

    def correlate_file(
        self,
        tttr_path: str,
        ranges: List,
        pairs: List[Dict[str, Any]],
        settings: Dict[str, Any] = None,
        filetype=None,
    ) -> Dict[str, Any]:
        return self._client.call(
            "burst_fcs.correlate_file",
            {
                "tttr_path": str(tttr_path),
                "ranges": [list(r) for r in ranges],
                "pairs": pairs,
                "settings": settings or {},
                "filetype": filetype,
            },
        )

    # -- local backend --------------------------------------------------
    @staticmethod
    def _make_local_client() -> InProcessClient:
        from chisurf.server.dispatcher import ServiceDispatcher
        from chisurf.server.session import SessionState

        from ..backend.services import register_services

        state = SessionState()
        dispatcher = ServiceDispatcher(state)
        register_services(dispatcher)
        return InProcessClient(dispatcher)
