from __future__ import annotations

import json
import threading
from importlib import import_module, resources
from typing import Any, Callable, Dict, List, Optional

import chisurf.server.protocol
from chisurf.server.services import ServiceResult, service_error
from chisurf.server.session import SessionState


class ServiceDispatcher:
    """Maps JSON-RPC method names to service handler functions.

    Each handler receives a ``params`` dict and must return a
    ``ServiceResult`` dict (``{"ok": bool, ...}``).

    Holds a ``SessionState`` reference and injects it into every
    registered service call.  If an ``event_bus`` is provided it is
    forwarded to services that accept an ``event_bus`` keyword argument.
    """

    def __init__(
        self,
        state: SessionState,
        event_bus: Optional[Any] = None,
    ):
        """Initialise the dispatcher.

        Parameters
        ----------
        state : SessionState
            Shared server-side runtime state.
        event_bus : object, optional
            Event bus instance for broadcasting.

        """
        self._state = state
        self._lock = threading.RLock()
        self._handlers: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}
        self._event_bus = event_bus

    def register(self, name: str, handler: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
        """Register a handler for an RPC method.

        Parameters
        ----------
        name : str
            Method name.
        handler : callable
            Handler accepting ``(params: dict) -> dict``.

        """
        with self._lock:
            self._handlers[name] = handler

    def has_method(self, name: str) -> bool:
        """Return ``True`` if *name* is a registered method.

        Parameters
        ----------
        name : str
            Method name to check.

        """
        with self._lock:
            return name in self._handlers

    def list_methods(self) -> List[str]:
        """Return sorted list of all registered method names."""
        with self._lock:
            return sorted(self._handlers.keys())

    def dispatch(self, method: str, params: Optional[Dict[str, Any]] = None) -> ServiceResult:
        """Look up and invoke a registered RPC handler.

        Parameters
        ----------
        method : str
            Method name.
        params : dict, optional
            Parameters forwarded to the handler.

        Returns
        -------
        ServiceResult
            The handler result or a structured error dict.

        """
        params = params or {}
        with self._lock:
            handler = self._handlers.get(method)
        if handler is None:
            return service_error(
                f"method '{method}' not found",
                error_code="METHOD_NOT_FOUND",
                jsonrpc_code=chisurf.server.protocol.METHOD_NOT_FOUND,
            )
        try:
            result = handler(params)
            if not isinstance(result, dict):
                result = {"ok": True, "result": result}
            return result
        except TypeError as e:
            return service_error(
                str(e),
                error_code="INVALID_PARAMS",
                jsonrpc_code=chisurf.server.protocol.INVALID_PARAMS,
                exception=e,
            )
        except Exception as e:
            return service_error(
                str(e),
                error_code="INTERNAL_ERROR",
                jsonrpc_code=chisurf.server.protocol.INTERNAL_ERROR,
                exception=e,
            )

    def _build_default_registry(self) -> None:
        """Register core service handlers from the declarative method table."""
        with resources.files("chisurf.server").joinpath("server_methods.json").open() as fp:
            methods = json.load(fp)["methods"]
        for spec in methods:
            self.register(spec["rpc"], self._handler_from_spec(spec))

    def _handler_from_spec(self, spec: Dict[str, Any]) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """Build a handler callable from a declarative method spec.

        Parameters
        ----------
        spec : dict
            Method specification from ``server_methods.json``.

        """
        meta = spec.get("meta")
        if meta == "methods":
            return lambda params: {"ok": True, "result": self.list_methods()}
        if meta == "protocol":
            return lambda params: {
                "ok": True,
                "protocol_version": chisurf.server.protocol.PROTOCOL_VERSION,
                "catalogue": chisurf.server.protocol.METHOD_CATALOGUE,
                "schemas": chisurf.server.protocol.METHOD_SCHEMAS,
            }

        module_name, function_name = spec["service"].split(".", 1)
        module = import_module(f"chisurf.server.services.{module_name}")
        function = getattr(module, function_name)
        if spec.get("event_bus") and self._event_bus is not None:
            return lambda params: function(self._state, event_bus=self._event_bus, **params)
        return lambda params: function(self._state, **params)
