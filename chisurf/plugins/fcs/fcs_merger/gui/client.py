"""PluginClient wrapper for the FCS-Merger plugin."""

from __future__ import annotations

from typing import Any, Dict, List

from chisurf.core.plugin.client import InProcessClient


class FcsMergerClient:
    """Client for the FCS-Merger backend services."""

    def __init__(self, client: Any = None):
        self._client = client if client is not None else self._make_local_client()

    def merge_folder(self, folder: str, output: str = None) -> Dict[str, Any]:
        return self._client.call(
            "fcs_merger.merge_folder", {"folder": str(folder), "output": output}
        )

    def average(self, correlations: List[dict]) -> Dict[str, Any]:
        return self._client.call("fcs_merger.average", {"correlations": correlations})

    def parse_folder(self, folder: str) -> Dict[str, Any]:
        return self._client.call("fcs_merger.parse_folder", {"folder": str(folder)})

    @staticmethod
    def _make_local_client() -> InProcessClient:
        from chisurf.server.dispatcher import ServiceDispatcher
        from chisurf.server.session import SessionState

        from ..backend.services import register_services

        state = SessionState()
        dispatcher = ServiceDispatcher(state)
        register_services(dispatcher)
        return InProcessClient(dispatcher)
