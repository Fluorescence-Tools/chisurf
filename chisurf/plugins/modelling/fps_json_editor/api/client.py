"""Client API for FPS JSON Editor services."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chisurf.core.plugin.client import InProcessClient

from .contract import (
    METHOD_DESCRIBE_CONTRACT,
    METHOD_FETCH_PDB,
    METHOD_NORMALIZE_PAYLOAD,
    METHOD_SAVE_AV_MRC,
    METHOD_SUMMARIZE_PAYLOAD,
    METHOD_VALIDATE_PAYLOAD,
)


class FpsJsonEditorClient:
    """Client for FPS JSON Editor backend services."""

    def __init__(self, client: Any | None = None) -> None:
        """Create an FPS JSON Editor client."""
        self._client = client or self._make_local_client()

    def fetch_pdb(
        self,
        pdb_id: str,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Download a PDB file by RCSB ID."""
        params: dict[str, Any] = {"pdb_id": pdb_id}
        if output_dir is not None:
            params["output_dir"] = str(output_dir)
        result = self._client.call(METHOD_FETCH_PDB, params)
        if not result.get("ok", True):
            raise RuntimeError(f"{METHOD_FETCH_PDB} failed: {result.get('error')}")
        return result.get("result", {})

    def describe_contract(self) -> dict[str, Any]:
        """Return the FPS JSON Editor workflow contract."""
        result = self._client.call(METHOD_DESCRIBE_CONTRACT, {})
        return result.get("result", {})

    def validate_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Validate an fps.json payload."""
        return self._call_result(METHOD_VALIDATE_PAYLOAD, {"payload": payload})

    def summarize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Summarize an fps.json payload."""
        return self._call_result(METHOD_SUMMARIZE_PAYLOAD, {"payload": payload})

    def normalize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Normalize an fps.json payload through the core model."""
        return self._call_result(METHOD_NORMALIZE_PAYLOAD, {"payload": payload})

    def save_av_mrc(
        self,
        path: str | Path,
        points: list[list[float]],
        grid_step: float,
    ) -> dict[str, Any]:
        """Save AV points as an IMP-backed MRC map."""
        return self._call_result(
            METHOD_SAVE_AV_MRC,
            {"path": str(path), "points": points, "grid_step": grid_step},
        )

    def _call_result(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Call a service method and unwrap the standard result envelope."""
        result = self._client.call(method, params)
        if not result.get("ok", True):
            raise RuntimeError(f"{method} failed: {result.get('error')}")
        return result.get("result", {})

    @staticmethod
    def _make_local_client() -> InProcessClient:
        """Create a local in-process client with FPS JSON Editor services."""
        from chisurf.server.dispatcher import ServiceDispatcher
        from chisurf.server.session import SessionState

        state = SessionState()
        dispatcher = ServiceDispatcher(state)
        dispatcher._build_default_registry()

        from ..rpc.services import register_services

        register_services(dispatcher)
        return InProcessClient(dispatcher)
