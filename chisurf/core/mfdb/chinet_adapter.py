from __future__ import annotations

import hashlib
import json
from typing import Any

try:
    import chinet
except ModuleNotFoundError as exc:
    if exc.name != "chinet":
        raise
    chinet = None

from chisurf.core.mfdb.models import (
    OPERATION_TYPES,
    PARAMETER_TYPES,
    RELATIONSHIP_TYPES,
    STATUS_VALUES,
    STORAGE_MODES,
    VALIDATION_STATUS_VALUES,
    validate_vocabulary,
)
from chisurf.core.mfdb.repository import MFDatabase

CHINET_SESSION_ARTIFACT = "chinet_session"
CHINET_NODE_ARTIFACT = "chinet_node"
CHINET_PARAMETER_SCHEMA = "chinet.parameter_ref.v1"
FIT_STATE_SCHEMA = "chisurf.fit_state.v1"
#: chinet >=0.3 serializes sessions via to_dict()/from_dict() and no longer ships a
#: schema wrapper (the old Session.to_schema()/session_from_schema()); the adapter now
#: owns the schema_name/version + software metadata it records for the stored session.
CHINET_SESSION_SCHEMA = "chinet.session.v1"
CHINET_SESSION_SCHEMA_VERSION = "1"
#: chinet >=0.3 replaced the pluggable DB backend (DB.set_backend/get_backend) with an
#: in-process object registry, so the transparent "MFDB-as-chinet-backend" integration
#: (connect_to_db("mfdb", …) → write_to_db/read_from_db) is unsupported until it is
#: redesigned against the new chinet DB API. store_chinet_session/load_chinet_session
#: (the explicit, supported path) are unaffected.
_MFDB_BACKEND_UNSUPPORTED = (
    "The transparent MFDB-as-chinet-backend integration requires chinet's removed "
    "DB.set_backend API (chinet <0.3); use store_chinet_session/load_chinet_session "
    "instead, or redesign MFDBChinetBackend against chinet's current DB registry."
)


def _session_to_schema(session: Any) -> dict[str, Any]:
    """Build the stored chinet-session schema document from a chinet ``Session``.

    Wraps chinet's ``session.to_dict()`` serialization with the schema_name/version +
    software metadata the MFDB records expect (chinet no longer provides ``to_schema``).
    """
    import chinet as _cn

    return {
        "schema_name": CHINET_SESSION_SCHEMA,
        "schema_version": CHINET_SESSION_SCHEMA_VERSION,
        "software": {"package": "chinet", "version": getattr(_cn, "__version__", "")},
        "session": session.to_dict(),
    }


_PARAMETER_REGISTRY_CACHE: dict[str, dict[str, Any]] | None = None


def _load_parameter_registry() -> dict[str, dict[str, Any]]:
    """Load the parameter registry mapping short names to flrcif_item_id.

    The registry is loaded once from ``chisurf.core.settings.parameter_registry``
    and cached for subsequent calls.

    Returns
    -------
    dict
        Mapping from short parameter name → entry dict.
    """
    global _PARAMETER_REGISTRY_CACHE
    if _PARAMETER_REGISTRY_CACHE is not None:
        return _PARAMETER_REGISTRY_CACHE
    try:
        import chisurf.core.settings as settings
        meta = getattr(settings, "parameter_registry", {})
        params = meta.get("parameters", meta) if isinstance(meta, dict) else {}
        _PARAMETER_REGISTRY_CACHE = {}
        for key, entry in params.items():
            if isinstance(entry, dict):
                _PARAMETER_REGISTRY_CACHE[key] = entry
    except Exception:
        _PARAMETER_REGISTRY_CACHE = {}
    return _PARAMETER_REGISTRY_CACHE


def _lookup_flrcif_name(short_name: str) -> str | None:
    """Look up the canonical flrCIF item identifier for a short parameter name.

    Parameters
    ----------
    short_name : str
        ChiSurf internal short parameter name (e.g. ``'E_FRET'``, ``'bg'``).

    Returns
    -------
    str or None
        The ``flrcif_item_id`` if found, else ``None``.
    """
    registry = _load_parameter_registry()
    entry = registry.get(short_name)
    if isinstance(entry, dict):
        flrcif = entry.get("flrcif_item_id")
        if isinstance(flrcif, str) and flrcif:
            return flrcif
    return None


def _require_chinet() -> Any:
    """Return the chinet module or raise an import error.

    Returns
    -------
    Any
        Imported chinet module.

    Raises
    ------
    ImportError
        If optional ``chinet`` dependency is not installed.
    """
    if chinet is None:
        raise ImportError(
            "Optional dependency 'chinet' is required for chinet/MFDB integration"
        )
    return chinet


def _json_dumps(value: Any) -> str:
    """Serialize a value as deterministic JSON.

    Parameters
    ----------
    value : Any
        Value to serialize.

    Returns
    -------
    str
        JSON string.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _json_loads(value: str | None) -> Any:
    """Deserialize a JSON string.

    Parameters
    ----------
    value : str or None
        JSON string.

    Returns
    -------
    Any
        Parsed value.
    """
    if not value:
        return None
    return json.loads(value)


def _ensure_vocabulary(db: MFDatabase) -> None:
    """Ensure chinet vocabulary values are active in MFDB.

    Parameters
    ----------
    db : MFDatabase
        MFDB connection.
    """
    for value in (CHINET_SESSION_ARTIFACT, CHINET_NODE_ARTIFACT):
        try:
            db.validate_extensible_vocab("artifact_kind", value)
        except ValueError:
            db.register_vocabulary_value(
                "artifact_kind",
                value,
                display_name=value,
                description="Chinet graph artifact",
                is_builtin=True,
                is_active=True,
            )


def _ensure_experiment(db: MFDatabase, experiment_id: str | None) -> None:
    """Create a placeholder experiment when an experiment_id is supplied.

    Parameters
    ----------
    db : MFDatabase
        MFDB connection.
    experiment_id : str or None
        Experiment identifier.
    """
    if not experiment_id:
        return
    try:
        row = db.conn.execute(
            "SELECT 1 FROM flr_experiment WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
    except Exception:
        return
    if row is None:
        try:
            db.add_experiment(experiment_id, status="pending", details="Created by chinet adapter")
        except Exception:
            return


def _artifact_id(prefix: str, object_id: str) -> str:
    """Build a stable MFDB artifact identifier.

    Parameters
    ----------
    prefix : str
        Artifact prefix.
    object_id : str
        Chinet object identifier.

    Returns
    -------
    str
        Artifact identifier.
    """
    return f"{prefix}:{object_id}"


def _validate_parameter_payload(parameter: dict[str, Any]) -> None:
    """Validate an explicit MFDB parameter payload before writing it.

    Parameters
    ----------
    parameter : dict
        Parameter payload.

    Raises
    ------
    ValueError
        If required fields are missing or the parameter type is invalid.
    """
    if not parameter.get("parameter_uuid"):
        raise ValueError("parameter_uuid is required")
    if not parameter.get("name"):
        raise ValueError("name is required")
    validate_vocabulary(
        parameter.get("parameter_type", "free"),
        PARAMETER_TYPES,
        "parameter_type",
    )


def _validate_archive_artifact_payload(
    artifact: dict[str, Any],
    default_artifact_id: str | None = None,
) -> dict[str, Any]:
    """Normalize and validate an archive input artifact payload before writing.

    Parameters
    ----------
    artifact : dict
        Artifact payload.
    default_artifact_id : str or None, optional
        Fallback artifact identifier when the payload omits one.

    Returns
    -------
    dict
        Normalized artifact payload.

    Raises
    ------
    ValueError
        If the payload is missing required fields or has invalid vocabulary values.
    """
    if not isinstance(artifact, dict):
        raise ValueError("artifact payload must be a mapping")
    artifact_id = artifact.get("artifact_id", artifact.get("id")) or default_artifact_id
    if not artifact_id:
        raise ValueError("artifact_id is required")
    kind = artifact.get("artifact_kind", artifact.get("artifact_type", "processed_data"))
    storage_mode = artifact.get("storage_mode", "local_file")
    validation_status = artifact.get("validation_status", "unvalidated")
    validate_vocabulary(storage_mode, STORAGE_MODES, "storage_mode")
    validate_vocabulary(validation_status, VALIDATION_STATUS_VALUES, "validation_status")
    return {
        "artifact_id": artifact_id,
        "artifact_kind": kind,
        "storage_mode": storage_mode,
        "experiment_id": artifact.get("experiment_id"),
        "file_path": artifact.get("file_path"),
        "url": artifact.get("url"),
        "folder_path": artifact.get("folder_path"),
        "mime_type": artifact.get("mime_type"),
        "size_bytes": artifact.get("size_bytes"),
        "checksum": artifact.get("checksum"),
        "checksum_algorithm": artifact.get("checksum_algorithm", "sha256"),
        "row_count": artifact.get("row_count"),
        "validation_status": validation_status,
        "validation_message": artifact.get("validation_message"),
        "metadata": artifact.get("metadata"),
        "data_json": artifact.get("data_json"),
        "data_blob": artifact.get("data_blob"),
        "data_format": artifact.get("data_format"),
        "role": artifact.get("role"),
        "ordinal": artifact.get("ordinal", 0),
        "link_metadata": artifact.get("link_metadata"),
    }


def _validate_extensible_vocab(db: MFDatabase, field_name: str, value: str | None) -> None:
    """Validate an extensible MFDB vocabulary value before writes.

    Parameters
    ----------
    db : MFDatabase
        MFDB connection.
    field_name : str
        Vocabulary field name.
    value : str or None
        Vocabulary value.

    Raises
    ------
    ValueError
        If the value is not active in MFDB vocabulary.
    """
    if value is None:
        return
    db.validate_extensible_vocab(field_name, value)


def _normalize_archive_input_artifacts(
    db: MFDatabase,
    input_artifacts: list[dict[str, Any]] | None,
    experiment_id: str | None,
    operation_id: str,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Validate archive input artifacts before writing them.

    Parameters
    ----------
    db : MFDatabase
        MFDB connection.
    input_artifacts : list of dict or None
        Input artifact payloads.
    experiment_id : str or None
        Default experiment identifier.
    operation_id : str
        Operation identifier used for generated artifact IDs.

    Returns
    -------
    list of tuple
        Normalized artifact payload and link metadata.
    """
    normalized: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for index, artifact in enumerate(input_artifacts or []):
        if not isinstance(artifact, dict):
            raise ValueError("artifact payload must be a mapping")
        artifact_id = str(artifact.get("artifact_id") or f"{operation_id}:input:{index}")
        payload = _validate_archive_artifact_payload(
            artifact,
            default_artifact_id=artifact_id,
        )
        payload["artifact_id"] = artifact_id
        if payload.get("experiment_id") is None:
            payload["experiment_id"] = experiment_id
        _validate_extensible_vocab(db, "artifact_kind", payload["artifact_kind"])
        _validate_extensible_vocab(db, "data_format", payload.get("data_format"))
        normalized.append(
            (
                payload,
                {
                    "role": artifact.get("role") or "input_data",
                    "metadata": artifact.get("metadata", {}),
                },
            )
        )
    return normalized


def _validate_fit_state_payload(
    fit_state_payload: dict[str, Any],
    explicit_parameters: list[dict[str, Any]] | None,
) -> None:
    """Validate fit-state parameter payloads before archive writes.

    Parameters
    ----------
    fit_state_payload : dict
        Fit-state payload.
    explicit_parameters : list of dict or None
        Explicit parameter payloads.

    Raises
    ------
    ValueError
        If an explicit parameter or derived fit-state parameter has invalid metadata.
    """
    for param in explicit_parameters or []:
        if isinstance(param, dict):
            _validate_parameter_payload(param)
    for state in (fit_state_payload.get("parameters") or {}).values():
        link_target = state.get("link_target")
        parameter_type = "linked" if link_target else "fixed" if state.get("fixed") else "free"
        validate_vocabulary(parameter_type, PARAMETER_TYPES, "parameter_type")


def _parameter_uuid_for_port(port: Any, explicit: dict[str, Any] | None = None) -> str:
    """Return the MFDB parameter UUID for a chinet port.

    Parameters
    ----------
    port : Port
        Chinet port.
    explicit : dict or None, optional
        Explicit parameter metadata.

    Returns
    -------
    str
        Parameter UUID.
    """
    if explicit and explicit.get("parameter_uuid"):
        return str(explicit["parameter_uuid"])
    return str(port.oid)


def _port_name(node_key: str, port_key: str, port: Any) -> str:
    """Return a human-readable port parameter name.

    Parameters
    ----------
    node_key : str
        Node key.
    port_key : str
        Port key.
    port : Port
        Chinet port.

    Returns
    -------
    str
        Parameter name.
    """
    return str(port_key or port.name or f"{node_key}.{port_key}")


def _parameter_payload_for_port(
    port: Any,
    node_key: str,
    port_key: str,
    session_id: str,
    node_id: str,
    explicit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an MFDB parameter payload for a chinet port.

    Parameters
    ----------
    port : Port
        Chinet port.
    node_key : str
        Node key.
    port_key : str
        Port key.
    session_id : str
        Chinet session identifier.
    node_id : str
        Chinet node identifier.
    explicit : dict or None, optional
        Explicit parameter metadata.

    Returns
    -------
    dict
        Parameter payload.
    """
    value = port.value
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        value = float(value[0]) if value else 0.0
    else:
        value = float(value)

    lb, ub = port.bounds
    metadata = {
        "schema_name": CHINET_PARAMETER_SCHEMA,
        "chinet_session_id": session_id,
        "chinet_node_id": node_id,
        "chinet_port_id": port.oid,
        "port_name": _port_name(node_key, port_key, port),
        "port_direction": "output" if port.is_output else "input",
        "value_type": int(port.get_value_type()),
        "is_reactive": bool(port.is_reactive),
        "source": "chinet.port",
    }
    metadata.update(explicit.get("metadata", {}) if isinstance(explicit, dict) else {})

    parameter_type = explicit.get("parameter_type") if isinstance(explicit, dict) else None
    if parameter_type is None:
        if port.is_linked():
            parameter_type = "linked"
        elif port.fixed:
            parameter_type = "fixed"
        else:
            parameter_type = "free"

    short_name = str((explicit or {}).get("name") or _port_name(node_key, port_key, port))
    canonical_name = _lookup_flrcif_name(short_name) or short_name
    metadata["chisurf_parameter_name"] = short_name
    return {
        "parameter_uuid": _parameter_uuid_for_port(port, explicit),
        "name": canonical_name,
        "value": value,
        "initial_value": value,
        "lower_bound": None if lb is None else float(lb),
        "upper_bound": None if ub is None else float(ub),
        "bounds_on": bool(port.is_bounded),
        "parameter_type": parameter_type,
        "metadata": metadata,
        **{k: v for k, v in (explicit or {}).items() if k in {
            "standard_error",
            "confidence_interval_low",
            "confidence_interval_high",
            "units",
            "expression",
            "prior",
            "mapping",
        }},
    }


def _parameter_uuid_by_port(parameters: list[dict[str, Any]] | None) -> dict[str, str]:
    """Map chinet port IDs to explicit parameter UUIDs.

    Parameters
    ----------
    parameters : list of dict or None
        Explicit parameter payloads.

    Returns
    -------
    dict
        Port ID to parameter UUID mapping.
    """
    mapping: dict[str, str] = {}
    for param in parameters or []:
        if not isinstance(param, dict):
            continue
        port_id = param.get("metadata", {}).get("chinet_port_id")
        if port_id and param.get("parameter_uuid"):
            mapping[str(port_id)] = str(param["parameter_uuid"])
    return mapping


def _store_parameters(
    db: MFDatabase,
    operation_id: str,
    session: Any,
    explicit_parameters: list[dict[str, Any]] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Store chinet port parameters in MFDB.

    Parameters
    ----------
    db : MFDatabase
        MFDB connection.
    operation_id : str
        Operation identifier.
    session : Session
        Chinet session.
    explicit_parameters : list of dict or None, optional
        Explicit parameter payloads.

    Returns
    -------
    tuple of list and dict
        Stored parameter UUIDs and port ID to parameter UUID mapping.
    """
    explicit_by_port: dict[str, dict[str, Any]] = {}
    for param in explicit_parameters or []:
        if not isinstance(param, dict):
            continue
        _validate_parameter_payload(param)
        port_id = param.get("metadata", {}).get("chinet_port_id")
        if port_id:
            explicit_by_port[str(port_id)] = param

    stored: list[str] = []
    mapping: dict[str, str] = {}
    for node_key, node in session.nodes.items():
        for port_key, port in node.ports.items():
            explicit = explicit_by_port.get(port.oid)
            payload = _parameter_payload_for_port(
                port,
                str(node_key),
                str(port_key),
                session.oid,
                node.oid,
                explicit,
            )
            db.record_parameter(operation_id=operation_id, **payload)
            stored.append(payload["parameter_uuid"])
            mapping[port.oid] = payload["parameter_uuid"]
    return stored, mapping


def _store_explicit_parameters(
    db: MFDatabase,
    operation_id: str,
    explicit_parameters: list[dict[str, Any]] | None,
    stored: set[str],
) -> None:
    """Store explicit parameters that are not already represented by ports.

    Parameters
    ----------
    db : MFDatabase
        MFDB connection.
    operation_id : str
        Operation identifier.
    explicit_parameters : list of dict or None
        Explicit parameter payloads.
    stored : set
        Parameter UUIDs already stored.
    """
    for param in explicit_parameters or []:
        if not isinstance(param, dict):
            continue
        if param.get("parameter_uuid") in stored:
            continue
        _validate_parameter_payload(param)
        db.record_parameter(operation_id=operation_id, **param)


def _store_parameter_links(
    db: MFDatabase,
    operation_id: str,
    session: Any,
    port_mapping: dict[str, str],
) -> int:
    """Store chinet port links as parameter dependency edges.

    Parameters
    ----------
    db : MFDatabase
        MFDB connection.
    operation_id : str
        Operation identifier.
    session : Session
        Chinet session.
    port_mapping : dict
        Port ID to parameter UUID mapping.

    Returns
    -------
    int
        Number of edges written.
    """
    validate_vocabulary("parameter_depends_on", RELATIONSHIP_TYPES, "relationship_type")
    count = 0
    for node in session.nodes.values():
        for port in node.ports.values():
            if port.link is None:
                continue
            db.add_edge(
                source_node_type="parameter",
                source_node_id=port_mapping.get(port.link.oid, port.link.oid),
                target_node_type="parameter",
                target_node_id=port_mapping.get(port.oid, port.oid),
                relationship_type="parameter_depends_on",
                operation_id=operation_id,
                metadata={
                    "schema_name": "chinet.parameter_link.v1",
                    "chinet_session_id": session.oid,
                },
            )
            count += 1
    return count


def _find_session_artifact_id(db: MFDatabase, session_id: str) -> str | None:
    """Find the MFDB artifact identifier for a chinet session id."""
    for artifact_kind in (CHINET_SESSION_ARTIFACT, "analysis_result"):
        for row in db.list_artifacts(artifact_type=artifact_kind):
            metadata = _json_loads(row.get("metadata_json")) or {}
            session_matches = (
                metadata.get("session_id") == session_id
                or metadata.get("chinet_session_id") == session_id
            )
            if session_matches:
                return row["artifact_id"]
    return None


def _find_owning_session(obj: Any) -> Any | None:
    """Return the registered chinet session that owns an object, if any."""
    client = _require_chinet()
    for candidate in client.DB.iter_objects():
        if not hasattr(candidate, "nodes"):
            continue
        for node in getattr(candidate, "nodes", {}).values():
            if obj is node:
                return candidate
            for port in getattr(node, "ports", {}).values():
                if obj is port:
                    return candidate
    return None


class MFDBChinetBackend:
    """Transparent MFDB backend for chinet object persistence."""

    def __init__(
        self,
        db: MFDatabase,
        operation_id: str | None = None,
        experiment_id: str | None = None,
        store_node_artifacts: bool = True,
        parameters: list[dict[str, Any]] | None = None,
        operation_type: str = "model_fitting",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.db = db
        self.operation_id = operation_id
        self.experiment_id = experiment_id
        self.store_node_artifacts = store_node_artifacts
        self.parameters = parameters
        self.operation_type = operation_type
        self.metadata = metadata or {}

    def close(self) -> None:
        """Close the underlying MFDB connection."""
        self.db.close()

    def connect_object(self, obj: Any, *args: Any, **kwargs: Any) -> bool:
        """Connect a chinet object to the configured MFDB backend."""
        client = _require_chinet()
        self.operation_id = kwargs.pop("operation_id", self.operation_id)
        self.experiment_id = kwargs.pop("experiment_id", self.experiment_id)
        self.store_node_artifacts = kwargs.pop("store_node_artifacts", self.store_node_artifacts)
        self.parameters = kwargs.pop("parameters", self.parameters)
        self.operation_type = kwargs.pop("operation_type", self.operation_type)
        self.metadata.update(kwargs.pop("metadata", {}) or {})
        client.DB.register(obj)
        session = obj if hasattr(obj, "nodes") else _find_owning_session(obj)
        if session is not None and self.operation_id is None:
            self.operation_id = f"chinet:{session.oid}"
        return True

    def disconnect_object(self, obj: Any) -> bool:
        """Disconnect a chinet object from the configured MFDB backend."""
        return True

    def read_object(self, obj: Any, oid: str) -> bool:
        """Read a chinet session from MFDB into ``obj``."""
        client = _require_chinet()
        if not hasattr(obj, "nodes"):
            return False
        artifact_id = _find_session_artifact_id(self.db, str(oid))
        if artifact_id is None:
            return False
        restored = load_chinet_session(self.db, artifact_id)
        obj.oid = restored.oid
        obj.nodes = restored.nodes
        obj.set_document(restored._document)
        client.DB.register(obj)
        for node in obj.nodes.values():
            client.DB.register(node)
            for port in node.ports.values():
                client.DB.register(port)
        return True

    def write_object(self, obj: Any) -> bool:
        """Write a chinet session, or its owning session, to MFDB."""
        client = _require_chinet()
        session = obj if hasattr(obj, "nodes") else _find_owning_session(obj)
        if session is None:
            client.DB.register(obj)
            return True
        if self.operation_id is None:
            self.operation_id = f"chinet:{session.oid}"
        store_chinet_session(
            self.db,
            session,
            operation_id=self.operation_id,
            experiment_id=self.experiment_id,
            parameters=self.parameters,
            store_node_artifacts=self.store_node_artifacts,
            operation_type=self.operation_type,
            metadata=self.metadata,
        )
        return True


def configure_mfdb_backend(
    db: MFDatabase | None = None,
    db_path: str | None = None,
    operation_id: str | None = None,
    experiment_id: str | None = None,
    store_node_artifacts: bool = True,
    parameters: list[dict[str, Any]] | None = None,
    operation_type: str = "model_fitting",
    metadata: dict[str, Any] | None = None,
) -> MFDBChinetBackend:
    """Configure chinet transparent persistence through MFDB.

    Parameters
    ----------
    db : MFDatabase or None, optional
        Existing MFDB connection.
    db_path : str or None, optional
        SQLite MFDB path used when ``db`` is omitted.
    operation_id : str or None, optional
        Default MFDB operation identifier for writes.
    experiment_id : str or None, optional
        Default experiment identifier for writes.
    store_node_artifacts : bool, default=True
        Whether to write one node artifact per chinet node.
    parameters : list of dict or None, optional
        Explicit parameter payloads.
    operation_type : str, default='model_fitting'
        Default MFDB operation type.
    metadata : dict or None, optional
        Default operation metadata.

    Returns
    -------
    MFDBChinetBackend
        Configured transparent backend.
    """
    client = _require_chinet()
    if not hasattr(client.DB, "set_backend"):
        raise NotImplementedError(_MFDB_BACKEND_UNSUPPORTED)
    backend = MFDBChinetBackend(
        db or MFDatabase(db_path),
        operation_id=operation_id,
        experiment_id=experiment_id,
        store_node_artifacts=store_node_artifacts,
        parameters=parameters,
        operation_type=operation_type,
        metadata=metadata,
    )
    client.DB.set_backend(backend)
    return backend


def clear_mfdb_backend(close: bool = False) -> None:
    """Remove the configured chinet MFDB backend.

    Parameters
    ----------
    close : bool, default=False
        Whether to close the backend MFDB connection.
    """
    client = _require_chinet()
    if not hasattr(client.DB, "get_backend"):
        # chinet >=0.3 removed the pluggable backend; nothing to clear.
        return
    backend = client.DB.get_backend()
    if close and backend is not None and hasattr(backend, "close"):
        backend.close()
    client.DB.clear_backend()


def store_chinet_session(
    db: MFDatabase,
    session: Any,
    operation_id: str,
    experiment_id: str | None = None,
    fit_refs: list[dict[str, Any]] | None = None,
    parameters: list[dict[str, Any]] | None = None,
    store_node_artifacts: bool = True,
    operation_type: str = "model_fitting",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Store a chinet session and related parameters in MFDB.

    Parameters
    ----------
    db : MFDatabase
        MFDB connection.
    session : Session
        Chinet session to persist.
    operation_id : str
        MFDB operation identifier.
    experiment_id : str or None, optional
        Associated experiment identifier.
    fit_refs : list of dict or None, optional
        ChiSurf fit references.
    parameters : list of dict or None, optional
        Explicit parameter payloads.
    store_node_artifacts : bool, default=True
        Whether to write one artifact per chinet node.
    operation_type : str, default='model_fitting'
        Existing MFDB operation type.
    metadata : dict or None, optional
        Additional operation metadata.

    Returns
    -------
    dict
        Created IDs and counts.
    """
    validate_vocabulary(operation_type, OPERATION_TYPES, "operation_type")
    validate_vocabulary("pending", STATUS_VALUES, "status")
    validate_vocabulary("contains", RELATIONSHIP_TYPES, "relationship_type")
    validate_vocabulary("parameter_depends_on", RELATIONSHIP_TYPES, "relationship_type")

    schema = _session_to_schema(session)
    schema_for_storage = dict(schema)
    if fit_refs is not None:
        schema_for_storage["fit_refs"] = fit_refs
    for param in parameters or []:
        if isinstance(param, dict):
            _validate_parameter_payload(param)

    operation_metadata = {
        "schema_name": schema["schema_name"],
        "schema_version": schema["schema_version"],
        "chinet_session_id": session.oid,
        "source": "chinet",
        **(metadata or {}),
    }
    if fit_refs:
        operation_metadata["fit_refs"] = fit_refs

    session_artifact_id = _artifact_id(CHINET_SESSION_ARTIFACT, session.oid)
    node_artifact_ids: list[str] = (
        [_artifact_id(CHINET_NODE_ARTIFACT, node.oid) for node in session.nodes.values()]
        if store_node_artifacts
        else []
    )

    with db.transaction():
        _ensure_vocabulary(db)
        _ensure_experiment(db, experiment_id)
        db.record_operation(
            operation_id=operation_id,
            operation_type=operation_type,
            experiment_id=experiment_id,
            software_package="chinet",
            software_module="chinet.session",
            software_version=schema["software"]["version"],
            metadata=operation_metadata,
        )

        db.register_artifact(
            artifact_id=session_artifact_id,
            artifact_kind=CHINET_SESSION_ARTIFACT,
            storage_mode="embedded_json",
            experiment_id=experiment_id,
            data_format="json",
            metadata={
                "schema_name": schema["schema_name"],
                "schema_version": schema["schema_version"],
                "session_id": session.oid,
                "source": "chinet",
                "operation_id": operation_id,
            },
            data_json=_json_dumps(schema_for_storage),
            validation_status="valid",
        )
        db.record_operation_link(
            operation_id=operation_id,
            artifact_id=session_artifact_id,
            direction="output",
            role="chinet_session",
            metadata={"schema_name": schema["schema_name"]},
        )

        if store_node_artifacts:
            for node_key, node in session.nodes.items():
                node_doc = {
                    "node_id": node.oid,
                    "session_id": session.oid,
                    "name": str(node_key or node.name),
                    "callback": node.callback,
                    "callback_type": (
                        node.callback_type_string or ("python" if node.callback_class else "")
                    ),
                    "valid": bool(node.node_valid_),
                    "ports": list(node.ports.keys()),
                }
                node_artifact_id = _artifact_id(CHINET_NODE_ARTIFACT, node.oid)
                db.register_artifact(
                    artifact_id=node_artifact_id,
                    artifact_kind=CHINET_NODE_ARTIFACT,
                    storage_mode="embedded_json",
                    experiment_id=experiment_id,
                    data_format="json",
                    metadata={
                        "schema_name": "chinet.node.v1",
                        "session_id": session.oid,
                        "node_id": node.oid,
                        "source": "chinet",
                        "operation_id": operation_id,
                    },
                    data_json=_json_dumps(node_doc),
                    validation_status="valid",
                )
                db.record_operation_link(
                    operation_id=operation_id,
                    artifact_id=node_artifact_id,
                    direction="output",
                    role="chinet_node",
                    metadata={"node_id": node.oid},
                )
                db.add_edge(
                    source_node_type="artifact",
                    source_node_id=session_artifact_id,
                    target_node_type="artifact",
                    target_node_id=node_artifact_id,
                    relationship_type="contains",
                    operation_id=operation_id,
                    metadata={"session_id": session.oid, "node_id": node.oid},
                )

        parameter_ids, port_mapping = _store_parameters(db, operation_id, session, parameters)
        _store_explicit_parameters(db, operation_id, parameters, set(parameter_ids))
        edge_count = _store_parameter_links(db, operation_id, session, port_mapping)

    if fit_refs is not None:
        session._document["fit_refs"] = fit_refs

    return {
        "ok": True,
        "operation_id": operation_id,
        "session_id": session.oid,
        "session_artifact_id": session_artifact_id,
        "node_artifact_ids": node_artifact_ids,
        "parameter_ids": parameter_ids,
        "parameter_count": len(parameter_ids),
        "dependency_edge_count": edge_count,
        "schema_name": schema["schema_name"],
        "schema_version": schema["schema_version"],
    }


def load_chinet_session(db: MFDatabase, artifact_id: str) -> Any:
    """Load and reconstruct a chinet session from an MFDB artifact.

    Parameters
    ----------
    db : MFDatabase
        MFDB connection.
    artifact_id : str
        Chinet session artifact identifier.

    Returns
    -------
    Session
        Reconstructed chinet session.
    """
    client = _require_chinet()
    artifact = db.get_artifact(artifact_id)
    if artifact is None:
        raise ValueError(f"Chinet session artifact not found: {artifact_id}")
    if artifact["artifact_kind"] not in {CHINET_SESSION_ARTIFACT, "analysis_result"}:
        raise ValueError(f"Artifact {artifact_id!r} is not a chinet session artifact")
    payload = _json_loads(artifact.get("data_json"))
    # chinet >=0.3: reconstruct from the to_dict() serialization stored under "session"
    # (older artifacts stored the raw chinet schema directly — fall back to that).
    session_dict = payload.get("session", payload) if isinstance(payload, dict) else payload
    return client.Session.from_dict(session_dict)


def archive_fit_to_mfdb(
    db: MFDatabase,
    fit: Any,
    operation_id: str,
    experiment_id: str | None = None,
    fit_id: str | None = None,
    dataset_id: str | None = None,
    input_artifacts: list[dict[str, Any]] | None = None,
    parameters: list[dict[str, Any]] | None = None,
    chinet_session: Any | None = None,
    operation_type: str = "local_fit",
) -> dict[str, Any]:
    """Archive one ChiSurf fit and its chinet-backed parameter graph to MFDB.

    Parameters
    ----------
    db : MFDatabase
        MFDB connection.
    fit : Fit
        ChiSurf fit object.
    operation_id : str
        MFDB operation identifier.
    experiment_id : str or None, optional
        Associated experiment identifier.
    fit_id : str or None, optional
        Fit identifier.
    dataset_id : str or None, optional
        Input dataset identifier.
    input_artifacts : list of dict or None, optional
        Input artifact payloads.
    parameters : list of dict or None, optional
        Explicit parameter payloads.
    chinet_session : Session or None, optional
        Existing chinet session. If omitted, one is built from fit parameters.
    operation_type : str, default='local_fit'
        Existing MFDB operation type.

    Returns
    -------
    dict
        Created IDs and counts.
    """
    from chisurf.core.project import fit_state

    fit_id = fit_id or getattr(fit, "unique_identifier", operation_id)
    fit_state_payload = fit_state.fit_to_state(fit)
    _validate_fit_state_payload(fit_state_payload, parameters)
    if chinet_session is None:
        chinet_session = _session_from_fit_state_payload(fit_state_payload)

    dataset_payload = None
    if dataset_id:
        dataset_payload = _validate_archive_artifact_payload(
            {
                "artifact_id": dataset_id,
                "artifact_kind": "processed_data",
                "storage_mode": "local_file",
                "experiment_id": experiment_id,
                "data_format": "unknown",
                "metadata": {"dataset_id": dataset_id, "source": "chisurf.fit"},
                "role": "input_data",
                "validation_status": "unvalidated",
            },
            default_artifact_id=dataset_id,
        )
        _validate_extensible_vocab(db, "artifact_kind", dataset_payload["artifact_kind"])
        _validate_extensible_vocab(db, "data_format", dataset_payload.get("data_format"))

    input_payloads = _normalize_archive_input_artifacts(
        db,
        input_artifacts,
        experiment_id,
        operation_id,
    )

    fit_artifact_id = _artifact_id("fit_result", str(fit_id))
    fit_payload = _validate_archive_artifact_payload(
        {
            "artifact_id": fit_artifact_id,
            "artifact_kind": "fit_result",
            "storage_mode": "embedded_json",
            "experiment_id": experiment_id,
            "data_format": "json",
            "metadata": {
                "schema_name": FIT_STATE_SCHEMA,
                "fit_id": str(fit_id),
                "operation_id": operation_id,
                "model_module": fit_state_payload.get("model_module"),
                "model_class": fit_state_payload.get("model_class"),
                "source": "chisurf.core.project.fit_state",
            },
            "role": "fit_state",
            "data_json": _json_dumps(fit_state_payload),
            "validation_status": "valid",
        },
        default_artifact_id=fit_artifact_id,
    )
    _validate_extensible_vocab(db, "artifact_kind", fit_payload["artifact_kind"])
    _validate_extensible_vocab(db, "data_format", fit_payload.get("data_format"))

    with db.transaction():
        result = store_chinet_session(
            db,
            chinet_session,
            operation_id=operation_id,
            experiment_id=experiment_id,
            fit_refs=[{"fit_uid": str(fit_id), "operation_id": operation_id}],
            parameters=parameters,
            operation_type=operation_type,
            metadata={
                "fit_id": str(fit_id),
                "model_module": fit_state_payload.get("model_module"),
                "model_class": fit_state_payload.get("model_class"),
            },
        )

        if dataset_payload is not None:
            artifact_kwargs = dataset_payload.copy()
            role = artifact_kwargs.pop("role", "input_data")
            ordinal = artifact_kwargs.pop("ordinal", 0)
            link_metadata = artifact_kwargs.pop("link_metadata", None)
            artifact_id = artifact_kwargs["artifact_id"]
            db.register_artifact(**artifact_kwargs)
            db.record_operation_link(
                operation_id=operation_id,
                artifact_id=artifact_id,
                direction="input",
                role=role,
                ordinal=ordinal,
                metadata=link_metadata or {"dataset_id": dataset_id},
            )

        for payload, link in input_payloads:
            artifact_kwargs = payload.copy()
            role = artifact_kwargs.pop("role", "input_data")
            ordinal = artifact_kwargs.pop("ordinal", 0)
            link_metadata = artifact_kwargs.pop("link_metadata", None)
            artifact_id = artifact_kwargs["artifact_id"]
            db.register_artifact(**artifact_kwargs)
            db.record_operation_link(
                operation_id=operation_id,
                artifact_id=artifact_id,
                direction="input",
                role=role,
                ordinal=ordinal,
                metadata=link_metadata or link.get("metadata", {}),
            )

        artifact_kwargs = fit_payload.copy()
        role = artifact_kwargs.pop("role", "fit_state")
        ordinal = artifact_kwargs.pop("ordinal", 0)
        link_metadata = artifact_kwargs.pop("link_metadata", None)
        fit_artifact_id = artifact_kwargs["artifact_id"]
        db.register_artifact(**artifact_kwargs)
        db.record_operation_link(
            operation_id=operation_id,
            artifact_id=fit_artifact_id,
            direction="output",
            role=role,
            ordinal=ordinal,
            metadata=link_metadata or {"schema_name": FIT_STATE_SCHEMA, "fit_id": str(fit_id)},
        )

        parameter_ids = _store_fit_state_parameters(db, operation_id, fit_state_payload, parameters)
        dependency_edge_count = _store_fit_state_links(db, operation_id, fit_state_payload)
        result.update(
            {
                "fit_id": str(fit_id),
                "fit_state_artifact_id": fit_artifact_id,
                "fit_state": fit_state_payload,
                "fit_parameter_ids": parameter_ids,
                "fit_parameter_count": len(parameter_ids),
                "fit_dependency_edge_count": dependency_edge_count,
            }
        )
    return result


def _session_from_fit_state_payload(fit_state_payload: dict[str, Any]) -> Any:
    """Build a chinet session from a fit-state payload.

    Parameters
    ----------
    fit_state_payload : dict
        ChiSurf fit-state payload.

    Returns
    -------
    Session
        Chinet session containing one fresh port per parameter.
    """
    client = _require_chinet()
    model_name = str(
        fit_state_payload.get("model_class")
        or fit_state_payload.get("model_module")
        or "Model"
    )
    node = client.Node(name=model_name)
    session = client.Session({model_name: node})
    port_by_id: dict[str, Any] = {}
    for uid, state in (fit_state_payload.get("parameters") or {}).items():
        if not isinstance(state, dict):
            continue
        bounds = state.get("bounds") or [None, None]
        lb = bounds[0] if len(bounds) > 0 else None
        ub = bounds[1] if len(bounds) > 1 else None
        port = client.Port(
            value=state.get("value", 0.0),
            oid=str(uid),
            name=str(state.get("name") or uid),
            fixed=bool(state.get("fixed", False)),
            is_output=False,
            is_reactive=False,
            is_bounded=bool(state.get("bounds_on", False) or lb is not None or ub is not None),
            lb=0.0 if lb is None else float(lb),
            ub=0.0 if ub is None else float(ub),
            value_type=0,
        )
        port_key = str(state.get("name") or uid)
        node.ports[port_key] = port
        port.set_node(node)
        port_by_id[str(uid)] = port
    node.fill_input_output_port_lookups()
    for uid, state in (fit_state_payload.get("parameters") or {}).items():
        if not isinstance(state, dict):
            continue
        target_uid = state.get("link_target")
        if target_uid and str(target_uid) in port_by_id and str(uid) in port_by_id:
            port_by_id[str(uid)].set_link(port_by_id[str(target_uid)])
    return session


def _store_fit_state_parameters(
    db: MFDatabase,
    operation_id: str,
    fit_state_payload: dict[str, Any],
    explicit_parameters: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Store fit-state parameters in MFDB.

    Parameters
    ----------
    db : MFDatabase
        MFDB connection.
    operation_id : str
        Operation identifier.
    fit_state_payload : dict
        Fit-state payload.
    explicit_parameters : list of dict or None, optional
        Explicit parameter payloads.

    Returns
    -------
    list of str
        Stored parameter UUIDs.
    """
    stored: list[str] = []
    seen: set[str] = set()
    for param in explicit_parameters or []:
        if isinstance(param, dict):
            _validate_parameter_payload(param)
    for state in (fit_state_payload.get("parameters") or {}).values():
        link_target = state.get("link_target")
        parameter_type = "linked" if link_target else "fixed" if state.get("fixed") else "free"
        validate_vocabulary(parameter_type, PARAMETER_TYPES, "parameter_type")

    for param in explicit_parameters or []:
        if not isinstance(param, dict):
            continue
        param_uuid = param.get("parameter_uuid")
        if param_uuid is None:
            param_uuid = hashlib.sha256(_json_dumps(param).encode()).hexdigest()
        param_uuid = str(param_uuid)
        db.record_parameter(operation_id=operation_id, **param)
        stored.append(param_uuid)
        seen.add(param_uuid)

    for uid, state in (fit_state_payload.get("parameters") or {}).items():
        if uid in seen:
            continue
        link_target = state.get("link_target")
        parameter_type = "linked" if link_target else "fixed" if state.get("fixed") else "free"
        bounds = state.get("bounds") or [None, None]
        short_name = str(state.get("name") or uid)
        canonical_name = _lookup_flrcif_name(short_name) or short_name
        db.record_parameter(
            parameter_uuid=str(uid),
            operation_id=operation_id,
            name=canonical_name,
            value=float(state.get("value") or 0.0),
            initial_value=float(state.get("value") or 0.0),
            lower_bound=None if bounds[0] is None else float(bounds[0]),
            upper_bound=None if bounds[1] is None else float(bounds[1]),
            bounds_on=bool(state.get("bounds_on", False)),
            parameter_type=parameter_type,
            metadata={
                "schema_name": FIT_STATE_SCHEMA,
                "source": "chisurf.core.project.fit_state",
                "fit_parameter_uid": str(uid),
                "chisurf_parameter_name": short_name,
                "link_target": link_target,
            },
        )
        stored.append(str(uid))
        seen.add(str(uid))
    return stored


def _store_fit_state_links(
    db: MFDatabase,
    operation_id: str,
    fit_state_payload: dict[str, Any],
) -> int:
    """Store fit-state parameter links as dependency edges.

    Parameters
    ----------
    db : MFDatabase
        MFDB connection.
    operation_id : str
        Operation identifier.
    fit_state_payload : dict
        Fit-state payload.

    Returns
    -------
    int
        Number of dependency edges written.
    """
    validate_vocabulary("parameter_depends_on", RELATIONSHIP_TYPES, "relationship_type")
    count = 0
    for uid, state in (fit_state_payload.get("parameters") or {}).items():
        target_uid = state.get("link_target")
        if not target_uid:
            continue
        db.add_edge(
            source_node_type="parameter",
            source_node_id=str(target_uid),
            target_node_type="parameter",
            target_node_id=str(uid),
            relationship_type="parameter_depends_on",
            operation_id=operation_id,
            metadata={
                "schema_name": FIT_STATE_SCHEMA,
                "source": "chisurf.core.project.fit_state",
            },
        )
        count += 1
    return count
