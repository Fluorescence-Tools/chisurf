"""RPC client for the Trace Browser GUI."""

from __future__ import annotations

from typing import Any

from chisurf.core.plugin.client import InProcessClient
from chisurf.plugins.tttr.trace_browser.api.contract import (
    METHOD_CONTRACT,
    METHOD_EXPORT_CSV,
    METHOD_GET_METADATA,
    METHOD_LIST_FILES,
    METHOD_LOAD_TRACE,
    METHOD_SET_METADATA,
)


class TraceBrowserClient:
    """Typed RPC client for Trace Browser."""

    def __init__(self, client: Any | None = None):
        """Wrap *client* or create a local in-process client."""
        self._client = client or self._make_local_client()

    def list_files(self, folder: str, recursive: bool = False, setup_settings: dict[str, Any] | None = None, selected_channels: list[int] | None = None) -> list[dict[str, Any]]:
        """List trace files through RPC."""
        result = self._call(
            METHOD_LIST_FILES,
            {
                "folder": folder,
                "recursive": recursive,
                "setup_settings": setup_settings,
                "selected_channels": selected_channels,
            },
        )
        return result.get("files", []) if result else []

    def get_metadata(self, folder: str) -> dict[str, dict[str, Any]]:
        """Load metadata through RPC."""
        result = self._call(METHOD_GET_METADATA, {"folder": folder})
        return result.get("metadata", {}) if result else {}

    def set_metadata(self, folder: str, metadata: dict[str, dict[str, Any]]) -> bool:
        """Save metadata through RPC."""
        result = self._call(METHOD_SET_METADATA, {"folder": folder, "metadata": metadata})
        return bool(result and result.get("ok"))

    def load_trace(self, path: str, time_window_ms: float, setup_settings: dict[str, Any] | None = None, selected_channels: list[int] | None = None, cache_folder: str | None = None) -> dict[str, Any] | None:
        """Load binned trace data through RPC."""
        result = self._call(
            METHOD_LOAD_TRACE,
            {
                "path": path,
                "time_window_ms": time_window_ms,
                "setup_settings": setup_settings,
                "selected_channels": selected_channels,
                "cache_folder": cache_folder,
            },
        )
        return result.get("trace") if result else None

    def export_csv(self, paths: list[str], output_dir: str, time_window_ms: float, setup_settings: dict[str, Any] | None = None, selected_channels: list[int] | None = None) -> list[str]:
        """Export traces to CSV through RPC."""
        result = self._call(
            METHOD_EXPORT_CSV,
            {
                "paths": paths,
                "output_dir": output_dir,
                "time_window_ms": time_window_ms,
                "setup_settings": setup_settings,
                "selected_channels": selected_channels,
            },
        )
        return result.get("paths", []) if result else []

    def describe_contract(self) -> dict[str, Any]:
        """Return the Trace Browser RPC contract."""
        result = self._call(METHOD_CONTRACT)
        return result or {}

    def _call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        try:
            result = self._client.call(method, params)
            if result and result.get("ok"):
                return result.get("result")
        except Exception:
            return None
        return None

    @staticmethod
    def _make_local_client() -> InProcessClient:
        """Create a local client with Trace Browser services registered."""
        from chisurf.plugins.tttr.trace_browser.backend.services import register_services
        from chisurf.server.dispatcher import ServiceDispatcher
        from chisurf.server.session import SessionState

        state = SessionState()
        dispatcher = ServiceDispatcher(state)
        dispatcher._build_default_registry()
        register_services(dispatcher)
        return InProcessClient(dispatcher)
