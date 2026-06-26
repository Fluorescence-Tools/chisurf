"""PluginClient wrapper for the FCS filter calculator."""

from __future__ import annotations

from typing import Any, Dict, List

from chisurf.core.plugin.client import InProcessClient


class FilterCalcClient:
    """Client for the FCS filter calculator backend services."""

    def __init__(self, client: Any = None):
        self._client = client if client is not None else self._make_local_client()

    def compute(self, total_decay, species_decays, metadata=None) -> Dict[str, Any]:
        return self._client.call("fcs_filter.compute", {
            "total_decay": list(total_decay),
            "species_decays": [list(s) for s in species_decays],
            "metadata": metadata,
        })

    def compute_from_files(self, total_path: str, species_paths: List[str]) -> Dict[str, Any]:
        return self._client.call("fcs_filter.compute_from_files", {
            "total_path": str(total_path),
            "species_paths": [str(p) for p in species_paths],
        })

    def compute_mfd_from_files(self, total_par_path, total_perp_path,
                               species_par_paths, species_perp_paths) -> Dict[str, Any]:
        return self._client.call("fcs_filter.compute_mfd_from_files", {
            "total_par_path": str(total_par_path),
            "total_perp_path": str(total_perp_path),
            "species_par_paths": [str(p) for p in species_par_paths],
            "species_perp_paths": [str(p) for p in species_perp_paths],
        })

    @staticmethod
    def _make_local_client() -> InProcessClient:
        from chisurf.server.dispatcher import ServiceDispatcher
        from chisurf.server.session import SessionState

        from ..backend.services import register_services

        state = SessionState()
        dispatcher = ServiceDispatcher(state)
        register_services(dispatcher)
        return InProcessClient(dispatcher)
