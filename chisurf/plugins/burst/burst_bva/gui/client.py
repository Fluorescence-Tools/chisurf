"""PluginClient wrapper for BVA.

Provides typed convenience methods and shields the GUI from direct API imports.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from chisurf.core.plugin.client import InProcessClient


logger = logging.getLogger(__name__)


class BvaClient:
    """Client for BVA backend services."""

    def __init__(self, client: Any = None):
        if client is not None:
            self._client = client
        else:
            self._client = self._make_local_client()

    def compute_bva(
        self,
        analysis_folder: str | Path,
        settings: dict[str, Any],
        pattern: str = "bi4_bur",
        file_type: str = "SPC-130",
    ) -> dict[str, Any]:
        """Run BVA analysis.

        Parameters
        ----------
        analysis_folder : str or Path
            Folder containing burst data.
        settings : dict
            BVA settings as JSON-compatible dict.
        pattern : str
            Glob pattern for burst data directories.
        file_type : str
            tttrlib container name.

        Returns
        -------
        dict
            Result with ``n_bursts_total``, ``n_bursts_valid``,
            ``output_paths``, ``files``.

        """
        params = {
            "analysis_folder": str(analysis_folder),
            "pattern": pattern,
            "settings": settings,
            "files": [],
        }
        svc_result = self._client.call("burst_bva.jobs.compute", params)
        if not svc_result.get("ok", True):
            err_msg = svc_result.get("error", "unknown error")
            raise RuntimeError(f"burst_bva.jobs.compute failed: {err_msg}")
        return svc_result.get("result", {})

    def describe_contract(self) -> dict[str, Any]:
        """Return the BVA workflow contract."""
        result = self._client.call("burst_bva.contract.describe", {})
        return result.get("result", {})

    @staticmethod
    def _make_local_client() -> InProcessClient:
        from chisurf.server.dispatcher import ServiceDispatcher
        from chisurf.server.session import SessionState
        state = SessionState()
        dispatcher = ServiceDispatcher(state)
        dispatcher._build_default_registry()

        from chisurf.plugins.burst.burst_bva.backend.services import register_services
        register_services(dispatcher)
        return InProcessClient(dispatcher)
