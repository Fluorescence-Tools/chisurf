from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GlobalViewState:
    """Serializable state for the GlobalView plugin window."""
    namespace: str = "globalview"
    selected_fit_indices: List[int] = field(default_factory=list)
    graph_layout: str = "kamada_kawai"
    include_fixed: bool = True
    connect_fits: bool = False
    graph_scale: float = 1.0
    node_size: float = 0.02


class GlobalViewClient:
    """Client for GlobalView backend services.

    Wraps a ZmqClient connected to the running ChiSurf RPC server.
    """

    def __init__(self, client: Optional[Any] = None):
        self._client = client if client is not None else self._make_local_client()

    @staticmethod
    def _make_local_client() -> Any:
        """Create a ZMQ client connected to the running ChiSurfServer.

        Returns
        -------
        ZmqClient
            Connected ZMQ client using the MFDB RPC port settings.
        """
        from chisurf.server.transport.zmq import ZmqClient
        try:
            import chisurf.core.settings as _cs_settings
            mfdb_cfg = _cs_settings.cs_settings.get("mfdb", {}) or {}
        except Exception:
            mfdb_cfg = {}
        client = ZmqClient(
            cmd_port=int(mfdb_cfg.get("cmd_port", 8765)),
            pub_port=int(mfdb_cfg.get("pub_port", 8766)),
            host=str(mfdb_cfg.get("rpc_host", "127.0.0.1")),
        )
        client.connect()
        return client

    def call(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Call an RPC method, logging any failure.

        Parameters
        ----------
        method : str
            The method name.
        params : dict or None
            Parameters to pass.

        Returns
        -------
        dict
            Response with at least ``"ok"``.
        """
        try:
            return self._client.call(method, params or {})
        except Exception:
            import chisurf.logging
            chisurf.logging.exception("GlobalViewClient: RPC call '%s' failed", method)
            return {"ok": False}

    def build_graph(
        self,
        fit_indices: Optional[List[int]] = None,
        fit_uids: Optional[List[str]] = None,
        include_fixed: bool = True,
        connect_fits: bool = False,
    ) -> Dict[str, Any]:
        """Build a parameter relationship graph from fit objects."""
        params: Dict[str, Any] = {
            "include_fixed": include_fixed,
            "connect_fits": connect_fits,
        }
        if fit_indices is not None:
            params["fit_indices"] = fit_indices
        if fit_uids is not None:
            params["fit_uids"] = fit_uids
        return self.call("globalview.graph.build", params)

    def list_parameters(
        self,
        fit_indices: Optional[List[int]] = None,
        fit_uids: Optional[List[str]] = None,
        include_fixed: bool = True,
    ) -> Dict[str, Any]:
        """List all parameters across fits."""
        params: Dict[str, Any] = {"include_fixed": include_fixed}
        if fit_indices is not None:
            params["fit_indices"] = fit_indices
        if fit_uids is not None:
            params["fit_uids"] = fit_uids
        return self.call("globalview.parameters.list", params)

    def link_parameters(
        self,
        source_parameter_name: str,
        target_parameter_name: str,
        source_fit_index: int,
        target_fit_index: int,
    ) -> Dict[str, Any]:
        """Link two parameters by name across fits."""
        return self.call("globalview.parameters.link", {
            "source_parameter_name": source_parameter_name,
            "target_parameter_name": target_parameter_name,
            "source_fit_index": source_fit_index,
            "target_fit_index": target_fit_index,
        })

    def unlink_parameter(
        self,
        parameter_name: str,
        fit_index: int,
    ) -> Dict[str, Any]:
        """Unlink a parameter."""
        return self.call("globalview.parameters.unlink", {
            "parameter_name": parameter_name,
            "fit_index": fit_index,
        })

    @property
    def is_connected(self) -> bool:
        return getattr(self._client, "is_connected", True)
