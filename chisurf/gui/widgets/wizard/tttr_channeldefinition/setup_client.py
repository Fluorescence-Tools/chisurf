"""Shared GUI-side client for the central detector-setup RPC service.

Wraps a :class:`~chisurf.core.plugin.client.InProcessClient` bound to a
``ServiceDispatcher`` carrying the default registry (which includes the
``detector_setups.*`` methods).  Any coordinating tool that owns a detector
wizard (Imaging Tools, Burst Analysis, …) talks to the shared setup store only
through this client — never by importing the service module directly.

The client is Qt-free; it is co-located with the detector-setup wizard purely
because that is the natural shared dependency for its callers.
"""

from __future__ import annotations

from typing import Any

from chisurf.core.plugin.client import InProcessClient


class DetectorSetupClient:
    """Typed convenience wrapper over the ``detector_setups.*`` RPC methods."""

    def __init__(self, client: Any = None):
        """Wrap *client*, or build a local in-process client when omitted."""
        self._client = client or self._make_local_client()

    @staticmethod
    def _make_local_client() -> InProcessClient:
        """Create a local in-process client with the default method registry."""
        from chisurf.server.dispatcher import ServiceDispatcher
        from chisurf.server.session import SessionState

        state = SessionState()
        dispatcher = ServiceDispatcher(state)
        dispatcher._build_default_registry()
        return InProcessClient(dispatcher)

    # ── live session-active definition ─────────────────────────────────
    def set_current(self, settings: dict[str, Any]) -> None:
        """Publish the live (possibly unsaved) detector definition."""
        self._client.call("detector_setups.set_current", {"settings": settings})

    def get_current(self) -> dict[str, Any]:
        """Return the live detector definition (or persisted ``last_used``)."""
        resp = self._client.call("detector_setups.current", {})
        if resp.get("ok"):
            return resp.get("result") or {}
        return {}

    # ── persistent named setups ────────────────────────────────────────
    def list_setups(self) -> dict[str, Any]:
        """Return ``{"setups": [names], "last_used": name}``."""
        resp = self._client.call("detector_setups.list", {})
        return (resp.get("result") or {}) if resp.get("ok") else {}

    def get_setup(self, name: str) -> dict[str, Any]:
        """Return the full settings dict for a saved setup by name."""
        resp = self._client.call("detector_setups.get", {"name": name})
        return (resp.get("result") or {}) if resp.get("ok") else {}

    def save_setup(self, name: str, settings: dict[str, Any]) -> None:
        """Persist ``settings`` under ``name`` in the canonical store."""
        self._client.call(
            "detector_setups.save", {"name": name, "settings": settings}
        )


__all__ = ["DetectorSetupClient"]
