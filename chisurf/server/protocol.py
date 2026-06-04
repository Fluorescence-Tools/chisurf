from __future__ import annotations

import threading
import uuid
from typing import Any, Dict, Optional, Tuple


def encode_request(
    method: str,
    params: Optional[Dict[str, Any]] = None,
    request_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a JSON-RPC 2.0 request dict."""
    msg: Dict[str, Any] = {
        "jsonrpc": "2.0",
        "method": method,
        "id": request_id if request_id is not None else _next_id(),
    }
    if params is not None:
        msg["params"] = params
    return msg


def decode_request(msg: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any], Optional[int]]]:
    """Validate and split a JSON-RPC request into (method, params, id).

    Returns ``None`` if the message is not a valid request.
    """
    if not isinstance(msg, dict):
        return None
    method = msg.get("method")
    if not isinstance(method, str) or not method:
        return None
    params = msg.get("params", {})
    if not isinstance(params, dict):
        params = {}
    req_id = msg.get("id")
    return method, params, req_id


def encode_response(
    result: Any,
    request_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a JSON-RPC 2.0 success response dict."""
    return {
        "jsonrpc": "2.0",
        "result": result,
        "id": request_id,
    }


def encode_error(
    code: int,
    message: str,
    data: Any = None,
    request_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a JSON-RPC 2.0 error response dict."""
    err: Dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {
        "jsonrpc": "2.0",
        "error": err,
        "id": request_id,
    }


def decode_response(msg: Dict[str, Any]) -> Tuple[Optional[Any], Optional[Dict[str, Any]], Optional[int]]:
    """Split a JSON-RPC response into (result, error, id).

    Exactly one of *result* or *error* will be non-``None``.
    """
    if not isinstance(msg, dict):
        return None, None, None
    result = msg.get("result")
    error = msg.get("error")
    req_id = msg.get("id")
    return result, error, req_id


def is_valid_request(msg: Any) -> bool:
    """Return ``True`` if *msg* is a structurally valid JSON-RPC request."""
    if not isinstance(msg, dict):
        return False
    return (
        msg.get("jsonrpc") == "2.0"
        and isinstance(msg.get("method"), str)
        and bool(msg.get("method"))
    )


def is_valid_response(msg: Any) -> bool:
    """Return ``True`` if *msg* is a structurally valid JSON-RPC response."""
    if not isinstance(msg, dict):
        return False
    if msg.get("jsonrpc") != "2.0":
        return False
    return "result" in msg or "error" in msg


# ── internal helpers ────────────────────────────────────────────────

_ID_COUNTER: int = 0
_ID_LOCK = threading.Lock()


def _next_id() -> int:
    """Return the next monotonically increasing request ID."""
    global _ID_COUNTER
    with _ID_LOCK:
        _ID_COUNTER += 1
        return _ID_COUNTER


# Standard JSON-RPC error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# ChiSurf server protocol version
PROTOCOL_VERSION = "1.0"

# Namespaced RPC method catalogue (for meta.protocol)
METHOD_CATALOGUE = {
    "meta": {
        "description": "Liveness, metadata, method discovery",
        "methods": [
            "meta.ping",
            "meta.methods",
            "meta.protocol",
        ],
    },
    "dataset": {
        "description": "Dataset CRUD and data access",
        "methods": [
            "dataset.list",
            "dataset.get",
            "dataset.curve_data",
            "dataset.remove",
            "dataset.clear",
            "dataset.rename",
            "dataset.group",
            "dataset.ungroup",
            "dataset.load",
        ],
    },
    "fit": {
        "description": "Fit CRUD, execution, and results",
        "methods": [
            "fit.list",
            "fit.get",
            "fit.run",
            "fit.remove",
            "fit.clear",
            "fit.set_dataset",
            "fit.set_result_idx",
            "fit.set_fit_range",
            "fit.create",
            "fit.update",
            "fit.save",
            "fit.curve_data",
        ],
    },
    "parameter": {
        "description": "Parameter inspection and mutation",
        "methods": [
            "parameter.get",
            "parameter.set_value",
            "parameter.set_fixed",
            "parameter.set_bounds",
            "parameter.set_bounds_on",
            "parameter.link",
            "parameter.unlink",
        ],
    },
    "project": {
        "description": "Project serialisation (save/load)",
        "methods": [
            "project.info",
            "project.save",
            "project.load",
        ],
    },
    "session": {
        "description": "Session lifecycle and snapshots",
        "methods": [
            "session.describe",
            "session.clear",
            "session.snapshot",
            "session.restore",
        ],
    },
    "model": {
        "description": "Model configuration",
        "methods": [
            "model.finalize",
            "model.set_parse_function",
        ],
    },
    "graph": {
        "description": "Fit graph construction for visualisation",
        "methods": [
            "graph.build",
            "graph.build_fits",
        ],
    },
}


METHOD_SCHEMAS = {
    "meta.ping": {
        "required_params": [],
        "optional_params": [],
        "result": "PingResult",
        "events": [],
    },
    "meta.methods": {
        "required_params": [],
        "optional_params": [],
        "result": "MethodListResult",
        "events": [],
    },
    "meta.protocol": {
        "required_params": [],
        "optional_params": [],
        "result": "ProtocolResult",
        "events": [],
    },
    "dataset.list": {
        "required_params": [],
        "optional_params": [],
        "result": "DatasetListResult",
        "events": [],
    },
    "dataset.get": {
        "required_params": [],
        "optional_params": ["dataset_index", "dataset_uid"],
        "result": "DatasetDetailResult",
        "events": [],
    },
    "dataset.curve_data": {
        "required_params": [],
        "optional_params": ["dataset_index", "dataset_uid"],
        "result": "DatasetCurveDataResult",
        "events": [],
    },
    "dataset.remove": {
        "required_params": [],
        "optional_params": ["dataset_indices", "dataset_uids"],
        "result": "ActionResult",
        "events": ["dataset.removed"],
    },
    "dataset.clear": {
        "required_params": [],
        "optional_params": [],
        "result": "ActionResult",
        "events": ["dataset.cleared"],
    },
    "dataset.rename": {
        "required_params": ["dataset_index", "name"],
        "optional_params": ["dataset_uid"],
        "result": "ActionResult",
        "events": [],
    },
    "dataset.group": {
        "required_params": ["dataset_indices"],
        "optional_params": ["name"],
        "result": "ActionResult",
        "events": [],
    },
    "dataset.ungroup": {
        "required_params": ["dataset_index"],
        "optional_params": [],
        "result": "ActionResult",
        "events": [],
    },
    "dataset.load": {
        "required_params": [],
        "optional_params": ["reader_name", "filename", "name", "curve_data"],
        "result": "DatasetCreateResult",
        "events": ["dataset.added"],
    },
    "fit.list": {
        "required_params": [],
        "optional_params": [],
        "result": "FitListResult",
        "events": [],
    },
    "fit.get": {
        "required_params": [],
        "optional_params": ["fit_index", "fit_uid"],
        "result": "FitDetailResult",
        "events": [],
    },
    "fit.run": {
        "required_params": [],
        "optional_params": ["fit_index", "fit_uid"],
        "result": "FitRunResult",
        "events": ["fit.ran"],
    },
    "fit.remove": {
        "required_params": [],
        "optional_params": ["fit_indices", "fit_uids"],
        "result": "ActionResult",
        "events": ["fit.removed"],
    },
    "fit.clear": {
        "required_params": [],
        "optional_params": [],
        "result": "ActionResult",
        "events": ["fit.cleared"],
    },
    "fit.set_dataset": {
        "required_params": ["fit_index", "dataset_index"],
        "optional_params": ["fit_uid", "dataset_uid"],
        "result": "ActionResult",
        "events": ["fit.dataset_changed"],
    },
    "fit.set_result_idx": {
        "required_params": ["fit_index", "result_idx"],
        "optional_params": ["fit_uid"],
        "result": "ActionResult",
        "events": ["fit.result_idx_changed"],
    },
    "fit.set_fit_range": {
        "required_params": ["fit_index"],
        "optional_params": ["fit_uid", "xmin", "xmax", "data_range"],
        "result": "ActionResult",
        "events": [],
    },
    "fit.create": {
        "required_params": [],
        "optional_params": ["dataset_index", "dataset_indices", "model_name", "fit_name", "model_kw"],
        "result": "FitCreateResult",
        "events": ["fit.added"],
    },
    "fit.update": {
        "required_params": [],
        "optional_params": ["fit_index", "fit_uid"],
        "result": "ActionResult",
        "events": ["fit.updated"],
    },
    "fit.save": {
        "required_params": ["target_path"],
        "optional_params": ["fit_index", "fit_uid"],
        "result": "ActionResult",
        "events": [],
    },
    "fit.curve_data": {
        "required_params": [],
        "optional_params": ["fit_index", "fit_uid"],
        "result": "FitCurveDataResult",
        "events": [],
    },
}
