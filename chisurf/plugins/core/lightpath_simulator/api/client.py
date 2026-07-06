"""Client wrapper for light-path simulator plugin RPC services."""

from __future__ import annotations

from typing import Any

from .contract import (
    METHOD_DESCRIBE_CONTRACT,
    METHOD_GET,
    METHOD_GET_PROBES_INFO,
    METHOD_LIST,
    METHOD_SAVE,
    METHOD_SIMULATE,
)


class LightPathClient:
    """Typed client wrapper for light-path simulator RPC methods."""

    def __init__(self, client: Any):
        """Wrap a ChiSurf plugin/RPC client."""
        self._client = client

    @classmethod
    def from_settings(cls, timeout_ms: int = 5000) -> LightPathClient:
        """Create a ZMQ MFDB client using the current ChiSurf settings."""
        import chisurf.core.settings as cs_settings
        from chisurf.core.mfdb.credentials import (
            load_runtime_session_token,
            load_session_token,
            store_runtime_session_token,
        )
        from mfdb.admin.gui.client import MFDBClient

        mfdb_settings = cs_settings.cs_settings.get("mfdb", {})
        server_host = mfdb_settings.get("last_server", "127.0.0.1")
        server_port = int(mfdb_settings.get("last_port", 8765))
        user_id = mfdb_settings.get("default_user_id", "user_default")
        client = MFDBClient(
            host=server_host,
            cmd_port=server_port,
            pub_port=server_port + 1,
            timeout_ms=timeout_ms,
        )
        token = load_runtime_session_token(server_host, server_port, user_id)
        if token is None:
            token = load_session_token(server_host, server_port, user_id)
        if token:
            client.token = token
            store_runtime_session_token(server_host, server_port, user_id, token)
        return cls(client)

    @property
    def client(self) -> Any:
        """Return the wrapped client."""
        return self._client

    def _call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Call a plugin RPC method and unwrap standard service envelopes."""
        if hasattr(self._client, "call"):
            response = self._client.call(method, params or {})
        else:
            response = self._client._call(method, params or {})
        if isinstance(response, dict) and response.get("ok") is False:
            raise RuntimeError(response.get("error", method))
        if isinstance(response, dict) and "result" in response:
            result = response["result"]
            if isinstance(result, dict):
                return result
            return {"value": result}
        return response

    def close(self) -> None:
        """Close the wrapped transport client when it exposes a close hook."""
        close = getattr(self._client, "close", None)
        if callable(close):
            close()
            return
        nested = getattr(self._client, "_client", None)
        nested_close = getattr(nested, "close", None)
        if callable(nested_close):
            nested_close()

    def simulate(self, graph: dict[str, Any], db_path: str | None = None) -> dict[str, Any]:
        """Run a light-path simulation."""
        return self._call(METHOD_SIMULATE, {"graph": graph, "db_path": db_path})

    def save(
        self,
        graph: dict[str, Any],
        name: str | None = None,
        db_path: str | None = None,
    ) -> dict[str, Any]:
        """Persist a light-path simulation in MFDB."""
        return self._call(METHOD_SAVE, {"graph": graph, "name": name, "db_path": db_path})

    def list_saved(self, db_path: str | None = None) -> list[dict[str, Any]]:
        """List saved light-path simulations."""
        return self._call(METHOD_LIST, {"db_path": db_path}).get("simulations", [])

    def get(self, operation_id: str, db_path: str | None = None) -> dict[str, Any]:
        """Load one saved light-path simulation."""
        return self._call(METHOD_GET, {"operation_id": operation_id, "db_path": db_path})

    def get_probes_info(self, db_path: str | None = None) -> list[dict[str, Any]]:
        """Return probe metadata for GUI palettes."""
        return self._call(METHOD_GET_PROBES_INFO, {"db_path": db_path}).get("probes", [])

    def describe_contract(self) -> dict[str, Any]:
        """Return the plugin workflow contract."""
        return self._call(METHOD_DESCRIBE_CONTRACT, {})
