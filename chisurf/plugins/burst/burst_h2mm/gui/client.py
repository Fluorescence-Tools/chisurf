"""PluginClient wrapper for the H2MM plugin.

Shields the GUI from direct backend imports and works identically in-process
or against a remote ZMQ server.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from chisurf.core.plugin.client import InProcessClient

logger = logging.getLogger(__name__)


class H2mmClient:
    """Client for the H2MM backend services."""

    def __init__(self, client: Any = None):
        self._client = client if client is not None else self._make_local_client()

    def compute(
        self,
        analysis_folder: str | Path | None = None,
        files: list[str] | None = None,
        settings: dict[str, Any] | None = None,
        pattern: str = "*.bur",
        write_output: bool = True,
    ) -> dict[str, Any]:
        """Run an H2MM analysis and return the result payload.

        Parameters
        ----------
        analysis_folder : str or Path, optional
            Folder containing ``.bur`` burst files.
        files : list of str, optional
            Explicit ``.bur`` file paths.
        settings : dict, optional
            H2MM settings as a JSON-compatible dict.
        pattern : str
            Glob for ``.bur`` files inside ``analysis_folder``.
        write_output : bool
            Whether to write ``h2mm/h2mm_result.json``.

        Returns
        -------
        dict
            The result payload (see :class:`H2mmResult`).
        """
        params = {
            "analysis_folder": str(analysis_folder) if analysis_folder else None,
            "files": files or [],
            "settings": settings or {},
            "pattern": pattern,
            "write_output": write_output,
        }
        res = self._client.call("burst_h2mm.jobs.compute", params)
        if not res.get("ok", True):
            raise RuntimeError(f"burst_h2mm.jobs.compute failed: {res.get('error', 'unknown')}")
        return res.get("result", {})

    def describe_contract(self) -> dict[str, Any]:
        """Return the H2MM workflow contract."""
        res = self._client.call("burst_h2mm.contract.describe", {})
        return res.get("result", {})

    @staticmethod
    def _make_local_client() -> InProcessClient:
        from chisurf.server.dispatcher import ServiceDispatcher
        from chisurf.server.session import SessionState

        state = SessionState()
        dispatcher = ServiceDispatcher(state)
        dispatcher._build_default_registry()

        from chisurf.plugins.burst.burst_h2mm.backend.services import register_services

        register_services(dispatcher)
        return InProcessClient(dispatcher)
