from __future__ import annotations

import copy
import json
from importlib import resources
from typing import Any, Callable, Dict, List, Optional

from chisurf.server.transport.zmq import ZmqClient


class RemoteError(Exception):
    """Raised when a server call returns an error or a transport failure."""

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        jsonrpc_code: int | None = None,
        exception_type: str | None = None,
        details: dict | None = None,
    ):
        super().__init__(message)
        self.error_code = error_code
        self.jsonrpc_code = jsonrpc_code
        self.exception_type = exception_type
        self.details = details


def _method_specs() -> list[dict[str, Any]]:
    with resources.files("chisurf.server").joinpath("client_methods.json").open() as fp:
        return json.load(fp)["methods"]


def _default(spec: dict[str, Any]) -> Any:
    return copy.deepcopy(spec.get("default"))


def _params_from_spec(
    spec: dict[str, Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    param_specs = spec.get("params", [])
    if len(args) > len(param_specs):
        raise TypeError(f"{spec['name']}() takes {len(param_specs)} positional arguments but {len(args)} were given")

    raw: dict[str, Any] = {}
    for param_spec, value in zip(param_specs, args):
        raw[param_spec["name"]] = value

    for param_spec in param_specs[len(args):]:
        name = param_spec["name"]
        if name in kwargs:
            raw[name] = kwargs.pop(name)
        elif param_spec.get("required"):
            raise TypeError(f"{spec['name']}() missing required argument: {name!r}")
        elif "default" in param_spec:
            raw[name] = _default(param_spec)

    if kwargs and not spec.get("allow_extra"):
        unexpected = next(iter(kwargs))
        raise TypeError(f"{spec['name']}() got an unexpected keyword argument {unexpected!r}")
    raw.update(kwargs)

    if spec["name"] == "parameter__set_bounds":
        raw["bounds"] = [raw.pop("lower"), raw.pop("upper")]

    return {key: value for key, value in raw.items() if value is not None}


def _make_method(spec: dict[str, Any]) -> Callable[..., Any]:
    def rpc_method(self: "ChisurfClient", *args: Any, **kwargs: Any) -> Any:
        params = _params_from_spec(spec, args, kwargs)
        result = self.call(spec["rpc"], params or None)
        if "result_key" in spec:
            return result.get(spec["result_key"], _default(spec))
        return result

    rpc_method.__name__ = spec["name"]
    rpc_method.__qualname__ = f"ChisurfClient.{spec['name']}"
    rpc_method.__doc__ = f"Call RPC method {spec['rpc']}."
    return rpc_method


def _install_rpc_methods(cls: type["ChisurfClient"]) -> None:
    for spec in _method_specs():
        setattr(cls, spec["name"], _make_method(spec))


def _make_alias(target: str, param_map: dict[str, str] | None = None) -> Callable[..., Any]:
    param_map = param_map or {}

    def alias(self: "ChisurfClient", *args: Any, **kwargs: Any) -> Any:
        for old_name, new_name in param_map.items():
            if old_name in kwargs:
                kwargs[new_name] = kwargs.pop(old_name)
        return getattr(self, target)(*args, **kwargs)

    alias.__name__ = target
    alias.__doc__ = f"Compatibility alias for {target}."
    return alias


def _install_legacy_aliases(cls: type["ChisurfClient"]) -> None:
    aliases = {
        "list_datasets": ("dataset__list", {}),
        "get_dataset_info": ("dataset__get", {}),
        "remove_datasets": ("dataset__remove", {}),
        "clear_datasets": ("dataset__clear", {}),
        "list_fits": ("fit__list", {}),
        "get_fit_info": ("fit__get", {}),
        "run_fit": ("fit__run", {}),
        "remove_fits": ("fit__remove", {}),
        "clear_fits": ("fit__clear", {}),
        "save_project": ("project__save", {"filename": "target_path"}),
        "load_project": ("project__load", {"filename": "project_path"}),
        "get_project_info": ("project__info", {}),
        "list_methods": ("meta__methods", {}),
        "ping": ("meta__ping", {}),
    }
    for name, (target, param_map) in aliases.items():
        setattr(cls, name, _make_alias(target, param_map))


class ChisurfClient:
    """Small client wrapper around the ChiSurf JSON-RPC transport."""

    def __init__(
        self,
        cmd_port: int = 8765,
        pub_port: int = 8766,
        host: str = "127.0.0.1",
        timeout_ms: int = 5000,
    ):
        self._client = ZmqClient(
            cmd_port=cmd_port,
            pub_port=pub_port,
            host=host,
            timeout_ms=timeout_ms,
        )

    def connect(self) -> None:
        self._client.connect()

    def close(self) -> None:
        self._client.close()

    def call(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        raw = self._client.call(method, params)
        if "error" not in raw:
            return raw.get("result", {})

        err = raw["error"]
        if not isinstance(err, dict):
            raise RemoteError(str(err))

        data = err.get("data")
        details = data if isinstance(data, dict) else {}
        raise RemoteError(
            message=err.get("message", str(err)),
            error_code=details.get("error_code"),
            jsonrpc_code=err.get("code"),
            exception_type=details.get("exception_type"),
            details=err,
        )

    def _call(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.call(method, params)

    def _call_result(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        return self._call(method, params).get("result")

    def dataset__load(
        self,
        reader: Any = None,
        name: Optional[str] = None,
        reader_name: Optional[str] = None,
        filename: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = dict(kwargs)
        if reader is not None:
            params["reader"] = reader
        if reader_name is not None:
            params["reader_name"] = reader_name
        if filename is not None:
            params["filename"] = filename
        if name is not None:
            params["name"] = name
        if "reader" not in params and "reader_name" not in params:
            return {"ok": False, "error": "dataset.load requires reader or reader_name"}
        return self.call("dataset.load", params)

    def add_dataset(
        self,
        reader_name: str,
        filename: str,
        curves: Optional[List[int]] = None,
        dataset_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"reader_name": reader_name, "filename": filename}
        if curves is not None:
            params["curves"] = curves
        if dataset_name is not None:
            params["dataset_name"] = dataset_name
        params.update(kwargs)
        return self._call("add_dataset", params)

    def get_parameter(self, fit_index: int, param_id: str) -> Dict[str, Any]:
        return self._call(
            "get_parameter",
            {"fit_index": fit_index, "parameter_name": param_id},
        ).get("parameter", {})

    def set_parameter_value(self, fit_index: int, param_id: str, value: float) -> Dict[str, Any]:
        return self._call(
            "set_parameter_value",
            {"fit_index": fit_index, "parameter_name": param_id, "value": value},
        )

    def set_parameter_fixed(self, fit_index: int, param_id: str, fixed: bool) -> Dict[str, Any]:
        return self._call(
            "set_parameter_fixed",
            {"fit_index": fit_index, "parameter_name": param_id, "fixed": fixed},
        )

    def set_parameter_bounds(self, fit_index: int, param_id: str, lower: float, upper: float) -> Dict[str, Any]:
        return self._call(
            "set_parameter_bounds",
            {"fit_index": fit_index, "parameter_name": param_id, "bounds": [lower, upper]},
        )

    def subscribe(self, topic: str = "", callback: Optional[Callable] = None) -> Any:
        return self._client.subscribe(topic, callback)

    def unsubscribe(self, topic: str = "", callback: Optional[Callable] = None) -> None:
        import zmq

        if callback is not None:
            self._client.unsubscribe(topic, callback)
            return
        if self._client._sub_socket is not None:
            self._client._sub_socket.setsockopt_string(zmq.UNSUBSCRIBE, topic)

    def drain(self) -> None:
        """Process all buffered subscriber events from the main thread."""
        self._client.drain()


_install_rpc_methods(ChisurfClient)
_install_legacy_aliases(ChisurfClient)
