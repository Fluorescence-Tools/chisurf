from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import numpy as np

from ._version import __version__

SCHEMA_NAME = "chinet.session.v1"
SCHEMA_VERSION = 1


def _json_value(value: Any) -> Any:
    """Convert numpy and non-JSON values into a JSON-serializable form.

    Parameters
    ----------
    value : Any
        Value to normalize.

    Returns
    -------
    Any
        JSON-serializable value.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _port_value(port: Any) -> Any:
    """Return a JSON-serializable port value.

    Parameters
    ----------
    port : Port
        Chinet port.

    Returns
    -------
    Any
        Scalar or list value.
    """
    return _json_value(port.value)


def _bounds(port: Any) -> list[float | None]:
    """Return port bounds as a two-item JSON list.

    Parameters
    ----------
    port : Port
        Chinet port.

    Returns
    -------
    list of float or None
        Lower and upper bounds.
    """
    lb, ub = port.bounds
    return [None if lb is None else float(lb), None if ub is None else float(ub)]


def _port_document(port: Any, node_id: str, port_key: str) -> dict[str, Any]:
    """Serialize one chinet port for the canonical session schema.

    Parameters
    ----------
    port : Port
        Chinet port.
    node_id : str
        Owning node identifier.
    port_key : str
        Port key within the owning node.

    Returns
    -------
    dict
        Canonical port document.
    """
    return {
        "port_id": port.oid,
        "node_id": node_id,
        "name": port_key or port.name,
        "direction": "output" if port.is_output else "input",
        "value": _port_value(port),
        "value_type": int(port.get_value_type()),
        "fixed": bool(port.fixed),
        "is_bounded": bool(port.is_bounded),
        "bounds": _bounds(port),
        "is_reactive": bool(port.is_reactive),
        "link": port.link.oid if port.link is not None else None,
    }


def _node_document(node: Any, node_key: str) -> dict[str, Any]:
    """Serialize one chinet node without executable callback state.

    Parameters
    ----------
    node : Node
        Chinet node.
    node_key : str
        Node key within the session.

    Returns
    -------
    dict
        Canonical node document.
    """
    callback = node.callback
    callback_type = node.callback_type_string or ""
    if node.callback_class is not None and not callback:
        callback = getattr(node.callback_class, "__qualname__", type(node.callback_class).__name__)
        callback_type = "python"
    return {
        "node_id": node.oid,
        "name": node_key or node.name,
        "callback": callback,
        "callback_type": callback_type,
        "valid": bool(node.node_valid_),
        "ports": [port.oid for port in node.ports.values()],
    }


def session_to_schema(session: Any) -> dict[str, Any]:
    """Convert a chinet session to the canonical ``chinet.session.v1`` schema.

    Parameters
    ----------
    session : Session
        Chinet session to serialize.

    Returns
    -------
    dict
        JSON-serializable session schema.

    Examples
    --------
    >>> payload = session_to_schema(chinet.Session())
    >>> payload["schema_name"]
    'chinet.session.v1'
    """
    nodes: list[dict[str, Any]] = []
    ports: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []
    port_by_id: dict[str, tuple[Any, str]] = {}

    for node_key, node in session.nodes.items():
        node_doc = _node_document(node, str(node_key))
        nodes.append(node_doc)
        for port_key, port in node.ports.items():
            port_doc = _port_document(port, node.oid, str(port_key))
            ports.append(port_doc)
            port_by_id[port.oid] = (port, str(port_key))

    for port, _port_key in port_by_id.values():
        if port.link is None:
            continue
        links.append(
            {
                "source_port_id": port.link.oid,
                "target_port_id": port.oid,
                "relationship_type": "parameter_depends_on",
            }
        )

    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "session_id": session.oid,
        "created_at": session._document.get("created_at") or datetime.now(timezone.utc).isoformat(),
        "software": {
            "package": "chinet",
            "version": __version__,
            "producer": "chisurf",
        },
        "nodes": nodes,
        "ports": ports,
        "links": links,
        "fit_refs": session._document.get("fit_refs", []),
        "metadata": session._document,
    }


def _validate_schema(payload: dict[str, Any]) -> None:
    """Validate a canonical chinet session schema payload.

    Parameters
    ----------
    payload : dict
        Schema payload to validate.

    Raises
    ------
    ValueError
        If the payload is malformed or has an unsupported schema version.
    """
    if not isinstance(payload, dict):
        raise ValueError("chinet session schema must be a dictionary")
    if payload.get("schema_name") != SCHEMA_NAME:
        raise ValueError(
            f"Unsupported chinet session schema {payload.get('schema_name')!r}; "
            f"expected {SCHEMA_NAME!r}"
        )
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported chinet session schema version {payload.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    if "session_id" not in payload:
        raise ValueError("chinet session schema is missing session_id")
    for key in ("nodes", "ports", "links"):
        if not isinstance(payload.get(key), list):
            raise ValueError(f"chinet session schema field {key!r} must be a list")


def session_from_schema(payload: dict[str, Any]) -> Any:
    """Reconstruct a chinet session from a canonical schema payload.

    Parameters
    ----------
    payload : dict
        Canonical ``chinet.session.v1`` payload.

    Returns
    -------
    Session
        Reconstructed chinet session.

    Raises
    ------
    ValueError
        If the payload is malformed or has an unsupported schema version.
    """
    from .db import DB
    from .node import Node
    from .port import Port
    from .session import Session

    _validate_schema(payload)
    DB.clear()

    session = Session()
    session.oid = str(payload["session_id"])
    session._document.setdefault("created_at", payload.get("created_at"))
    session._document["fit_refs"] = payload.get("fit_refs", [])

    id_map: dict[str, Any] = {session.oid: session}
    node_by_key: dict[str, Any] = {}

    for node_payload in payload.get("nodes", []):
        if not isinstance(node_payload, dict) or "node_id" not in node_payload:
            raise ValueError("each chinet node schema entry requires node_id")
        node_id = str(node_payload["node_id"])
        node = Node(name=str(node_payload.get("name") or node_id))
        node.oid = node_id
        node.callback = str(node_payload.get("callback") or "")
        node.callback_type_string = str(node_payload.get("callback_type") or "")
        node.callback_class = None
        node.node_valid_ = bool(node_payload.get("valid", False))
        node_key = node_payload.get("name", node_id)
        session.add_node(str(node_key), node)
        id_map[node_id] = node
        node_by_key[str(node_key)] = node

    port_by_id: dict[str, Any] = {}
    for port_payload in payload.get("ports", []):
        if not isinstance(port_payload, dict) or "port_id" not in port_payload:
            raise ValueError("each chinet port schema entry requires port_id")
        port_id = str(port_payload["port_id"])
        node = id_map.get(str(port_payload.get("node_id")))
        if node is None:
            raise ValueError(f"chinet port {port_id!r} references unknown node")
        bounds = port_payload.get("bounds") or [None, None]
        lb = bounds[0] if len(bounds) > 0 else None
        ub = bounds[1] if len(bounds) > 1 else None
        port = Port(
            value=port_payload.get("value", 0),
            oid=port_id,
            name=str(port_payload.get("name") or port_id),
            fixed=bool(port_payload.get("fixed", False)),
            is_output=str(port_payload.get("direction", "input")) == "output",
            is_reactive=bool(port_payload.get("is_reactive", False)),
            is_bounded=bool(port_payload.get("is_bounded", False)),
            lb=0.0 if lb is None else float(lb),
            ub=0.0 if ub is None else float(ub),
            value_type=int(port_payload.get("value_type", 0)),
        )
        port_key = port_payload.get("name", port_id)
        node.ports[str(port_key)] = port
        port.set_node(node)
        port_by_id[port_id] = port

    for node in node_by_key.values():
        node.fill_input_output_port_lookups()

    for link_payload in payload.get("links", []):
        if not isinstance(link_payload, dict):
            raise ValueError("each chinet link schema entry must be a dictionary")
        source = port_by_id.get(str(link_payload.get("source_port_id")))
        target = port_by_id.get(str(link_payload.get("target_port_id")))
        if source is None or target is None:
            raise ValueError("chinet link schema references an unknown port")
        target.set_link(source)

    return session


def schema_to_json(payload: dict[str, Any], indent: int | None = None) -> str:
    """Serialize a chinet session schema payload as JSON.

    Parameters
    ----------
    payload : dict
        Canonical schema payload.
    indent : int or None, optional
        JSON indentation.

    Returns
    -------
    str
        JSON string.
    """
    return json.dumps(payload, indent=indent, sort_keys=True)


def schema_from_json(text: str) -> dict[str, Any]:
    """Parse a chinet session schema JSON string.

    Parameters
    ----------
    text : str
        JSON string.

    Returns
    -------
    dict
        Schema payload.
    """
    return json.loads(text)
