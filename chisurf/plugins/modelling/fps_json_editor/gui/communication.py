"""Communication layer for the FPS JSON Editor GUI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chisurf.core.plugin.client import InProcessClient


class FpsJsonEditorClient:
    """Client for FPS JSON Editor backend services.

    GUI code should use this client instead of importing backend services
    directly.
    """

    def __init__(self, client: Any | None = None) -> None:
        """Create an FPS JSON Editor client."""
        self._client = client or self._make_local_client()

    def fetch_pdb(
        self,
        pdb_id: str,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Download a PDB file by RCSB ID.

        Parameters
        ----------
        pdb_id : str
            Four-character RCSB PDB ID.
        output_dir : str or pathlib.Path, optional
            Directory where the PDB file should be written.

        Returns
        -------
        dict
            Result with ``pdb_id``, ``path`` and ``source`` keys.

        Raises
        ------
        RuntimeError
            If the RPC call failed.

        """
        params: dict[str, Any] = {"pdb_id": pdb_id}
        if output_dir is not None:
            params["output_dir"] = str(output_dir)
        result = self._client.call("fps_json_editor.pdb.fetch", params)
        if not result.get("ok", True):
            raise RuntimeError(f"fps_json_editor.pdb.fetch failed: {result.get('error')}")
        return result.get("result", {})

    def describe_contract(self) -> dict[str, Any]:
        """Return the FPS JSON Editor workflow contract."""
        result = self._client.call("fps_json_editor.contract.describe", {})
        return result.get("result", {})

    @staticmethod
    def _make_local_client() -> InProcessClient:
        """Create a local in-process client with FPS JSON Editor services."""
        from chisurf.server.dispatcher import ServiceDispatcher
        from chisurf.server.session import SessionState

        state = SessionState()
        dispatcher = ServiceDispatcher(state)
        dispatcher._build_default_registry()

        from chisurf.plugins.modelling.fps_json_editor.backend.services import (
            register_services,
        )

        register_services(dispatcher)
        return InProcessClient(dispatcher)
