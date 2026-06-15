"""GUI-facing ZMQ client adapter for fitting operations.

This module provides a single ``FittingClient`` class that wraps
:class:`~chisurf.core.api._client.ChisurfClient` (itself a wrapper around
:class:`~chisurf.server.transport.zmq.ZmqClient`) with explicit, typed
methods for every fitting operation the GUI needs.

All GUI/widget communication **must** go through this adapter.
The adapter ensures that:

* Every state mutation is a JSON-RPC call to the server.
* Return values are plain JSON-safe DTOs (dicts/lists), never live objects.
* Long-running operations (sampling, parameter scans) return job
  identifiers that can be polled or cancelled.
"""

from __future__ import annotations

import atexit
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import warnings

from chisurf.core.api._client import ChisurfClient, RemoteError


# ── Global adapter instance ─────────────────────────────────────────

_FITTING_CLIENT: Optional["FittingClient"] = None
"""Module-level singleton fitting client used by all migrated widgets.

Set via :func:`install_fitting_client` during application startup
(when a ZMQ server is available).  Widgets access it through
:func:`get_fitting_client`.
"""


def ensure_fitting_client() -> "FittingClient":
    """Return the current :class:`FittingClient`, creating one with no ZMQ
    client if none is installed yet.

    Returns
    -------
    FittingClient
        Always returns a valid instance (never ``None``).
    """
    global _FITTING_CLIENT
    if _FITTING_CLIENT is None:
        _FITTING_CLIENT = FittingClient()
    return _FITTING_CLIENT


def install_fitting_client(client: Optional[ChisurfClient] = None) -> "FittingClient":
    """Install the global :class:`FittingClient` adapter.

    Call this once during app startup, after the ZMQ client is
    connected.

    Parameters
    ----------
    client : ChisurfClient or None
        Connected ZMQ client instance.  When ``None``, the adapter
        has no transport backend and all RPC calls will fail
        gracefully.

    Returns
    -------
    FittingClient
        The installed adapter.
    """
    global _FITTING_CLIENT
    _FITTING_CLIENT = FittingClient(client)
    return _FITTING_CLIENT


def get_fitting_client() -> Optional["FittingClient"]:
    """Return the global :class:`FittingClient` or ``None``.

    Returns ``None`` when no :class:`FittingClient` has been installed
    (e.g. application startup hasn't completed yet).  Widgets should
    always check for ``None`` before calling methods.

    See Also
    --------
    install_fitting_client : Install the global adapter.
    """
    return _FITTING_CLIENT


def has_fitting_client() -> bool:
    """Return ``True`` if a :class:`FittingClient` is installed."""
    return _FITTING_CLIENT is not None


class FittingClient:
    """Typed GUI adapter over a ``ChisurfClient`` for fitting operations.

    All state mutations go through JSON-RPC calls to the server.
    When the ZMQ server is unreachable or an RPC call fails, the
    adapter returns ``{"ok": False}`` — it never performs direct
    in-process mutations.

    Parameters
    ----------
    client : ChisurfClient or None
        Connected ZMQ client instance, or ``None`` (RPC will fail
        gracefully).
    """

    def __init__(
        self,
        client: Optional[ChisurfClient] = None,
    ) -> None:
        self._client = client
        self._last_bootstrap_warning: float = 0.0

    @property
    def _rpc_available(self) -> bool:
        try:
            return self._client is not None and hasattr(self._client, "call")
        except Exception:
            return False

    def _try_rpc(self, method: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Call an RPC method, logging any failure.

        Parameters
        ----------
        method : str
            RPC method name.
        params : dict
            Parameters to pass.

        Returns
        -------
        dict or None
            Response dict, or ``None`` on failure.
        """
        if self._in_server_dispatch():
            return None
        if not self._rpc_available:
            self._try_bootstrap_transport()
        if not self._rpc_available:
            import chisurf.logging
            chisurf.logging.warning(
                "FittingClient: RPC not available (no client) for method '%s'", method
            )
            return None
        try:
            return self._client.call(method, params)
        except RemoteError as e:
            if self._is_transport_failure(str(e)):
                self._drop_transport(method, e)
                return None
            import chisurf.logging
            chisurf.logging.exception(
                "FittingClient: RPC call '%s' failed", method
            )
            return None
        except Exception:
            import chisurf.logging
            chisurf.logging.exception(
                "FittingClient: RPC call '%s' failed", method
            )
            return None

    def _in_server_dispatch(self) -> bool:
        """Return whether this call is already running inside the RPC server."""
        try:
            from chisurf.server.transport.zmq import in_server_dispatch

            return in_server_dispatch()
        except Exception:
            return False

    def _try_bootstrap_transport(self) -> bool:
        """Attach to or start the embedded ChiSurf RPC transport."""
        if self._in_server_dispatch():
            return False
        if self._rpc_available:
            return True

        try:
            import chisurf
            import chisurf.core.settings as cs_settings
            import chisurf.logging
            from chisurf.server.startup import (
                rpc_is_available,
                session_state_from_live_chisurf,
            )

            mfdb_cfg = cs_settings.cs_settings.get("mfdb", {}) or {}
            host = str(mfdb_cfg.get("rpc_host", mfdb_cfg.get("last_server", "127.0.0.1")))
            cmd_port = int(mfdb_cfg.get("cmd_port", mfdb_cfg.get("last_port", 8765)))
            pub_port = int(mfdb_cfg.get("pub_port", cmd_port + 1))
            timeout_ms = int(mfdb_cfg.get("fitting_timeout_ms", 300))

            server = (
                getattr(chisurf, "__chisurf_rpc_server__", None)
                or getattr(chisurf, "__mfdb_rpc_server__", None)
            )
            if not rpc_is_available(host, cmd_port, pub_port, timeout_ms=100):
                if server is not None:
                    deadline = time.time() + 1.0
                    while time.time() < deadline:
                        if rpc_is_available(host, cmd_port, pub_port, timeout_ms=100):
                            break
                        time.sleep(0.05)
                    else:
                        try:
                            server.stop()
                        except Exception:
                            pass
                        server = None
                        chisurf.__chisurf_rpc_server__ = None
                        chisurf.__chisurf_rpc_server_thread__ = None
                        chisurf.__mfdb_rpc_server__ = None
                        chisurf.__mfdb_rpc_server_thread__ = None

                if server is None:
                    from chisurf.server.app import ChiSurfServer

                    server = ChiSurfServer(
                        host=host,
                        cmd_port=cmd_port,
                        pub_port=pub_port,
                        state=session_state_from_live_chisurf(),
                    )
                    thread = threading.Thread(
                        target=server.serve_forever,
                        daemon=True,
                        name="chisurf-rpc-server",
                    )
                    thread.start()
                    chisurf.__chisurf_rpc_server__ = server
                    chisurf.__chisurf_rpc_server_thread__ = thread
                    chisurf.__mfdb_rpc_server__ = server
                    chisurf.__mfdb_rpc_server_thread__ = thread
                    atexit.register(server.stop)

                deadline = time.time() + 3.0
                while time.time() < deadline:
                    if rpc_is_available(host, cmd_port, pub_port, timeout_ms=100):
                        break
                    time.sleep(0.05)

            client = ChisurfClient(
                host=host,
                cmd_port=cmd_port,
                pub_port=pub_port,
                timeout_ms=timeout_ms,
            )
            client.connect()
            client.call("meta.ping", {})
            self._client = client
            chisurf.logging.info(
                "FittingClient: attached RPC transport at %s:%s",
                host,
                cmd_port,
            )
            return True
        except Exception as e:
            now = time.time()
            if now - self._last_bootstrap_warning > 5.0:
                self._last_bootstrap_warning = now
                try:
                    import chisurf.logging

                    chisurf.logging.warning(
                        "FittingClient: could not bootstrap RPC transport: %s",
                        e,
                    )
                except Exception:
                    pass
            return False

    def _is_transport_failure(self, message: str) -> bool:
        """Return ``True`` for errors that mean the RPC transport is unusable."""
        lowered = message.lower()
        return any(
            marker in lowered
            for marker in (
                "timeout",
                "socket operation on non-socket",
                "send failed",
                "poll failed",
                "recv failed",
                "not a socket",
            )
        )

    def _drop_transport(self, method: str, error: Exception) -> None:
        """Close and forget an unreachable transport after a failed RPC call."""
        import chisurf.logging

        chisurf.logging.warning(
            "FittingClient: disabling RPC after transport failure in '%s': %s",
            method,
            error,
        )
        client = self._client
        self._client = None
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    # ── Fit CRUD ─────────────────────────────────────────────────────

    def list_fits(self) -> List[Dict[str, Any]]:
        result = self._try_rpc("fit.list", {})
        if result is not None:
            return (result or {}).get("fits", [])
        return []

    def get_fit(self, fit_uid: Optional[str] = None, fit_index: Optional[int] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if fit_uid is not None:
            params["fit_uid"] = fit_uid
        if fit_index is not None:
            params["fit_index"] = fit_index
        result = self._try_rpc("fit.get", params)
        if result is not None:
            return (result or {}).get("fit", {})
        return {}

    def create_fit(
        self,
        dataset_indices: Optional[List[int]] = None,
        model_name: Optional[str] = None,
        fit_name: Optional[str] = None,
        model_kw: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if dataset_indices is not None:
            params["dataset_indices"] = dataset_indices
        if model_name is not None:
            params["model_name"] = model_name
        if fit_name is not None:
            params["fit_name"] = fit_name
        if model_kw is not None:
            params["model_kw"] = model_kw
        result = self._try_rpc("fit.create", params)
        if result is not None:
            return result
        return {"ok": False}

    def remove_fits(
        self,
        fit_indices: Optional[List[int]] = None,
        fit_uids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if fit_indices is not None:
            params["fit_indices"] = fit_indices
        if fit_uids is not None:
            params["fit_uids"] = fit_uids
        result = self._try_rpc("fit.remove", params)
        if result is not None:
            return result
        return {"ok": False, "removed_count": 0, "remaining_count": 0}

    def clear_fits(self) -> Dict[str, Any]:
        result = self._try_rpc("fit.clear", {})
        if result is not None:
            return result
        return {"ok": False}

    def reorder_fits(self, fit_order: List[str]) -> Dict[str, Any]:
        result = self._try_rpc("fit.reorder", {"fit_order": fit_order})
        if result is not None:
            return result
        return {"ok": False}

    # ── Fit actions ──────────────────────────────────────────────────

    def run_fit(
        self,
        fit_uid: Optional[str] = None,
        fit_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if fit_uid is not None:
            params["fit_uid"] = fit_uid
        if fit_index is not None:
            params["fit_index"] = fit_index
        result = self._try_rpc("fit.run", params)
        if result is not None and result.get("ok", False):
            return result
        return {"ok": False}

    def update_fit(
        self,
        fit_uid: Optional[str] = None,
        fit_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if fit_uid is not None:
            params["fit_uid"] = fit_uid
        if fit_index is not None:
            params["fit_index"] = fit_index
        result = self._try_rpc("fit.update", params)
        if result is not None and result.get("ok", False):
            return result
        return {"ok": False}

    def save_fit(
        self,
        filename: str = "fit_export",
        fit_uid: Optional[str] = None,
        fit_index: Optional[int] = None,
        file_type: str = "csv",
        save_curves: bool = False,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "filename": filename,
            "file_type": file_type,
            "save_curves": save_curves,
        }
        if fit_uid is not None:
            params["fit_uid"] = fit_uid
        if fit_index is not None:
            params["fit_index"] = fit_index
        result = self._try_rpc("fit.save", params)
        if result is not None:
            return result
        return {"ok": False}

    def set_fit_dataset(
        self,
        fit_index: int,
        dataset_index: Optional[int] = None,
        dataset_uid: Optional[str] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"fit_index": fit_index}
        if dataset_index is not None:
            params["dataset_index"] = dataset_index
        if dataset_uid is not None:
            params["dataset_uid"] = dataset_uid
        result = self._try_rpc("fit.set_dataset", params)
        if result is not None:
            return result
        return {"ok": False}

    def set_fit_result_idx(
        self,
        fit_index: int,
        result_idx: int,
        fit_uid: Optional[str] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"fit_index": fit_index, "result_idx": result_idx}
        if fit_uid is not None:
            params["fit_uid"] = fit_uid
        result = self._try_rpc("fit.set_result_idx", params)
        if result is not None:
            return result
        return {"ok": False}

    def set_fit_range(
        self,
        fit_uid: Optional[str] = None,
        fit_index: Optional[int] = None,
        xmin: int = 0,
        xmax: int = 0,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"xmin": xmin, "xmax": xmax}
        if fit_uid is not None:
            params["fit_uid"] = fit_uid
        if fit_index is not None:
            params["fit_index"] = fit_index
        result = self._try_rpc("fit.set_fit_range", params)
        if result is not None:
            return result
        return {"ok": False}

    def auto_fit_range(
        self,
        fit_uid: Optional[str] = None,
        fit_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if fit_uid is not None:
            params["fit_uid"] = fit_uid
        if fit_index is not None:
            params["fit_index"] = fit_index
        result = self._try_rpc("fit.range.auto", params)
        if result is not None:
            return result
        return {"ok": False}

    def set_fit_mask(
        self,
        mask: List[float],
        fit_uid: Optional[str] = None,
        fit_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"mask": mask}
        if fit_uid is not None:
            params["fit_uid"] = fit_uid
        if fit_index is not None:
            params["fit_index"] = fit_index
        result = self._try_rpc("fit.mask.set", params)
        if result is not None:
            return result
        return {"ok": False}

    # ── Fit selection ────────────────────────────────────────────────

    def select_fit(self, fit_uid: Optional[str] = None, fit_index: Optional[int] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if fit_uid is not None:
            params["fit_uid"] = fit_uid
        if fit_index is not None:
            params["fit_index"] = fit_index
        result = self._try_rpc("fit.select", params)
        if result is not None:
            return result
        return {"ok": False}

    def get_active_fit(self) -> Dict[str, Any]:
        result = self._try_rpc("fit.select", {"_action": "get"})
        if result is not None:
            return result.get("fit", {})
        return {}

    # ── Fit group operations ─────────────────────────────────────────

    def group_select_member(self, fit_uid: str, member_index: int) -> Dict[str, Any]:
        result = self._try_rpc("fit.group.select_member", {
            "fit_uid": fit_uid, "member_index": member_index,
        })
        if result is not None:
            return result
        return {"ok": False}

    def group_add_member(self, group_fit_uid: str, member_fit_uid: str) -> Dict[str, Any]:
        result = self._try_rpc("fit.group.add_member", {
            "group_fit_uid": group_fit_uid, "member_fit_uid": member_fit_uid,
        })
        if result is not None:
            return result
        return {"ok": False}

    def group_remove_member(self, group_fit_uid: str, member_index: int) -> Dict[str, Any]:
        result = self._try_rpc("fit.group.remove_member", {
            "group_fit_uid": group_fit_uid, "member_index": member_index,
        })
        if result is not None:
            return result
        return {"ok": False}

    def group_link_parameters_by_name(self, group_fit_uid: str, parameter_name: str) -> Dict[str, Any]:
        result = self._try_rpc("fit.group.link_parameters_by_name", {
            "group_fit_uid": group_fit_uid, "parameter_name": parameter_name,
        })
        if result is not None:
            return result
        return {"ok": False}

    # ── Parameter operations ─────────────────────────────────────────

    def get_parameter(
        self,
        parameter_name: str,
        fit_uid: Optional[str] = None,
        fit_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"parameter_name": parameter_name}
        if fit_uid is not None:
            params["fit_uid"] = fit_uid
        if fit_index is not None:
            params["fit_index"] = fit_index
        result = self._try_rpc("parameter.get", params)
        if result is not None:
            return (result or {}).get("parameter", {})
        return {}

    def set_parameter_value(
        self,
        parameter_name: str,
        value: float,
        fit_uid: Optional[str] = None,
        fit_index: Optional[int] = None,
        local_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"parameter_name": parameter_name, "value": value}
        if fit_uid is not None:
            params["fit_uid"] = fit_uid
        if fit_index is not None:
            params["fit_index"] = fit_index
        if local_idx is not None:
            params["local_idx"] = local_idx
        result = self._try_rpc("parameter.set_value", params)
        if result is not None and result.get("ok", False):
            return result
        return {"ok": False}

    def set_parameter_fixed(
        self,
        parameter_name: str,
        fixed: bool,
        fit_uid: Optional[str] = None,
        fit_index: Optional[int] = None,
        local_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"parameter_name": parameter_name, "fixed": fixed}
        if fit_uid is not None:
            params["fit_uid"] = fit_uid
        if fit_index is not None:
            params["fit_index"] = fit_index
        if local_idx is not None:
            params["local_idx"] = local_idx
        result = self._try_rpc("parameter.set_fixed", params)
        if result is not None and result.get("ok", False):
            return result
        return {"ok": False}

    def set_parameter_bounds(
        self,
        parameter_name: str,
        bounds: Tuple[float, float],
        fit_uid: Optional[str] = None,
        fit_index: Optional[int] = None,
        local_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "parameter_name": parameter_name,
            "bounds": list(bounds),
        }
        if fit_uid is not None:
            params["fit_uid"] = fit_uid
        if fit_index is not None:
            params["fit_index"] = fit_index
        if local_idx is not None:
            params["local_idx"] = local_idx
        result = self._try_rpc("parameter.set_bounds", params)
        if result is not None and result.get("ok", False):
            return result
        return {"ok": False}

    def set_parameter_bounds_on(
        self,
        parameter_name: str,
        bounds_on: bool,
        fit_uid: Optional[str] = None,
        fit_index: Optional[int] = None,
        local_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"parameter_name": parameter_name, "bounds_on": bounds_on}
        if fit_uid is not None:
            params["fit_uid"] = fit_uid
        if fit_index is not None:
            params["fit_index"] = fit_index
        if local_idx is not None:
            params["local_idx"] = local_idx
        result = self._try_rpc("parameter.set_bounds_on", params)
        if result is not None and result.get("ok", False):
            return result
        return {"ok": False}

    def link_parameters(
        self,
        parameter_name: str,
        target_parameter_name: str,
        fit_index: Optional[int] = None,
        target_fit_index: Optional[int] = None,
        fit_uid: Optional[str] = None,
        target_fit_uid: Optional[str] = None,
        local_idx: Optional[int] = None,
        target_local_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "parameter_name": parameter_name,
            "target_parameter_name": target_parameter_name,
        }
        if fit_index is not None:
            params["fit_index"] = fit_index
        if target_fit_index is not None:
            params["target_fit_index"] = target_fit_index
        if fit_uid is not None:
            params["fit_uid"] = fit_uid
        if target_fit_uid is not None:
            params["target_fit_uid"] = target_fit_uid
        if local_idx is not None:
            params["local_idx"] = local_idx
        if target_local_idx is not None:
            params["target_local_idx"] = target_local_idx
        result = self._try_rpc("parameter.link", params)
        if result is not None and result.get("ok", False):
            return result
        return {"ok": False}

    def unlink_parameter(
        self,
        parameter_name: str,
        fit_uid: Optional[str] = None,
        fit_index: Optional[int] = None,
        local_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"parameter_name": parameter_name}
        if fit_uid is not None:
            params["fit_uid"] = fit_uid
        if fit_index is not None:
            params["fit_index"] = fit_index
        if local_idx is not None:
            params["local_idx"] = local_idx
        result = self._try_rpc("parameter.unlink", params)
        if result is not None and result.get("ok", False):
            return result
        return {"ok": False}

    # ── Model operations ─────────────────────────────────────────────

    def model_finalize(
        self,
        fit_uid: Optional[str] = None,
        fit_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if fit_uid is not None:
            params["fit_uid"] = fit_uid
        if fit_index is not None:
            params["fit_index"] = fit_index
        result = self._try_rpc("model.finalize", params)
        if result is not None and result.get("ok", False):
            return result
        return {"ok": False}

    def model_set_parse_function(
        self,
        parse_function: str,
        fit_uid: Optional[str] = None,
        fit_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"parse_function": parse_function}
        if fit_uid is not None:
            params["fit_uid"] = fit_uid
        if fit_index is not None:
            params["fit_index"] = fit_index
        result = self._try_rpc("model.set_parse_function", params)
        if result is not None:
            return result
        return {"ok": False}

    def model_add_component(
        self,
        component_type: str,
        fit_uid: Optional[str] = None,
        fit_index: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"component_type": component_type}
        if fit_uid is not None:
            params["fit_uid"] = fit_uid
        if fit_index is not None:
            params["fit_index"] = fit_index
        params.update(kwargs)
        result = self._try_rpc("model.component.add", params)
        if result is not None:
            return result
        return {"ok": False}

    def model_remove_component(
        self,
        component_index: int,
        fit_uid: Optional[str] = None,
        fit_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"component_index": component_index}
        if fit_uid is not None:
            params["fit_uid"] = fit_uid
        if fit_index is not None:
            params["fit_index"] = fit_index
        result = self._try_rpc("model.component.remove", params)
        if result is not None:
            return result
        return {"ok": False}

    def model_get_state(
        self,
        fit_uid: Optional[str] = None,
        fit_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if fit_uid is not None:
            params["fit_uid"] = fit_uid
        if fit_index is not None:
            params["fit_index"] = fit_index
        result = self._try_rpc("model.state.get", params)
        if result is not None:
            return result.get("state", {})
        return {}

    def model_set_state(
        self,
        state_data: Dict[str, Any],
        fit_uid: Optional[str] = None,
        fit_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"state_data": state_data}
        if fit_uid is not None:
            params["fit_uid"] = fit_uid
        if fit_index is not None:
            params["fit_index"] = fit_index
        result = self._try_rpc("model.state.set", params)
        if result is not None:
            return result
        return {"ok": False}

    # ── Plot data ────────────────────────────────────────────────────

    def get_plot_data(
        self,
        plot_type: str = "fit_data",
        fit_uid: Optional[str] = None,
        fit_index: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"plot_type": plot_type}
        if fit_uid is not None:
            params["fit_uid"] = fit_uid
        if fit_index is not None:
            params["fit_index"] = fit_index
        params.update(kwargs)
        result = self._try_rpc("plot.fit_data", params)
        if result is not None:
            return result.get("plot", {})
        return {}

    # ── Sampling ─────────────────────────────────────────────────────

    def start_sampling(
        self,
        fit_uid: Optional[str] = None,
        fit_index: Optional[int] = None,
        n_steps: int = 1000,
        n_runs: int = 1,
        target_directory: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "n_steps": n_steps,
            "n_runs": n_runs,
        }
        if fit_uid is not None:
            params["fit_uid"] = fit_uid
        if fit_index is not None:
            params["fit_index"] = fit_index
        if target_directory is not None:
            params["target_directory"] = target_directory
        params.update(kwargs)
        result = self._try_rpc("fit.sample.start", params)
        if result is not None:
            return result
        return {"ok": False}

    def cancel_sampling(self, job_id: str) -> Dict[str, Any]:
        result = self._try_rpc("fit.sample.cancel", {"job_id": job_id})
        if result is not None:
            return result
        return {"ok": False}

    def sampling_status(self, job_id: str) -> Dict[str, Any]:
        result = self._try_rpc("fit.sample.status", {"job_id": job_id})
        if result is not None:
            return result
        return {"ok": False}

    # ── Parameter scan ───────────────────────────────────────────────

    def start_parameter_scan(
        self,
        parameter_name: str,
        fit_uid: Optional[str] = None,
        fit_index: Optional[int] = None,
        n_steps: int = 50,
        range_factor: float = 2.0,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "parameter_name": parameter_name,
            "n_steps": n_steps,
            "range_factor": range_factor,
        }
        if fit_uid is not None:
            params["fit_uid"] = fit_uid
        if fit_index is not None:
            params["fit_index"] = fit_index
        result = self._try_rpc("fit.parameter_scan.start", params)
        if result is not None:
            return result
        return {"ok": False}

    def cancel_parameter_scan(self, job_id: str) -> Dict[str, Any]:
        result = self._try_rpc("fit.parameter_scan.cancel", {"job_id": job_id})
        if result is not None:
            return result
        return {"ok": False}

    def parameter_scan_result(self, job_id: str) -> Dict[str, Any]:
        result = self._try_rpc("fit.parameter_scan.result", {"job_id": job_id})
        if result is not None:
            return result
        return {"ok": False}

    # ── GlobalView helpers ─────────────────────────────────────────────

    def get_fit_objects(self) -> List[Any]:
        """Return a snapshot of fit objects from the global fit list.

        .. deprecated::
            Use :meth:`list_fits` or :meth:`get_fit` instead.
        """
        warnings.warn(
            "get_fit_objects() is deprecated, use list_fits() or get_fit() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            import chisurf as cs
            return list(getattr(cs, "fits", []))
        except Exception:
            return []

    # ── Events ───────────────────────────────────────────────────────

    def subscribe(self, topic: str, callback: Callable) -> Any:
        if self._client is not None:
            return self._client.subscribe(topic, callback)
        return None

    def drain(self) -> None:
        if self._client is not None:
            self._client.drain()

    def unsubscribe(self, topic: str, callback: Callable) -> None:
        if self._client is not None:
            self._client.unsubscribe(topic, callback)

    # ── Convenience helpers ──────────────────────────────────────────

    def fit_count(self) -> int:
        return len(self.list_fits())

    def parameter_dict(
        self,
        fit_uid: Optional[str] = None,
        fit_index: Optional[int] = None,
    ) -> Dict[str, Dict[str, Any]]:
        fit = self.get_fit(fit_uid=fit_uid, fit_index=fit_index)
        return fit.get("parameters", {})

    def subscribe_to_fit_events(self, callback: Callable) -> Any:
        return self.subscribe("fit.", callback)

    def subscribe_to_parameter_events(self, callback: Callable) -> Any:
        return self.subscribe("parameter.", callback)
