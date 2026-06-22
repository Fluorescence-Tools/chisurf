"""Simple API for plugins to register results in MFDB."""
from __future__ import annotations

import logging
import uuid
from dataclasses import is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from chisurf.core.mfdb.base import MFDBClientBase
from chisurf.core.mfdb.operation_parameters import OperationParameterError

if TYPE_CHECKING:
    import pandas

    from chisurf.core.mfdb.session import SessionContext

logger = logging.getLogger(__name__)

_GLOBAL_DB: MFDBClientBase | None = None


class LinkValidationError(ValueError):
    """A requested provenance link target (sample/parent) does not exist.

    Raised at the registration boundary *before* any rows are written, so the
    registration leaves no partial artifact/object state. Callers (e.g. the
    burst pipeline) may catch this to report the bad link as a warning rather
    than aborting, while genuine integration errors during the write still
    surface loudly. Subclasses :class:`ValueError` for backward compatibility.
    """


def _resolve_active_user_id() -> str:
    """Resolve the active MFDB user ID from settings.

    Thin wrapper over the canonical resolver (PRD-17) so writes stamp the same
    identity reads scope by.
    """
    from chisurf.core.mfdb.session import configured_default_user_id

    return configured_default_user_id()


def register_result(
    kind: str,
    data: str | bytes | Path | pandas.DataFrame | dict | None = None,
    sample_id: str = "",
    parent_artifact_id: str = "",
    operation_type: str = "",
    parameters: dict | None = None,
    metadata: dict | None = None,
    data_format: str = "",
    setup_id: str = "",
    setup_version: int | None = None,
    db: MFDBClientBase | None = None,
    is_public: bool = False,
    session: "SessionContext | None" = None,
) -> str:
    """Register a plugin result in MFDB.

    Parameters
    ----------
    kind : str
        Artifact kind. Prefer a value from ``models.ARTIFACT_KINDS`` such as
        ``"processed_data"``, ``"fit_result"``, ``"fcs_correlation"``, or
        ``"burst_table"``.
    data : str, bytes, Path, pandas.DataFrame, dict, or None, optional
        Payload to archive. Paths are copied into the object store; bytes are
        stored directly; dictionaries, data frames, ``DataCurve`` objects, and
        typed payload dataclasses are serialized through the msgpack payload
        codec. ``None`` creates a metadata-only artifact.
    sample_id : str, optional
        Existing sample identifier to link via ``measured_sample``.
    parent_artifact_id : str, optional
        Existing source artifact. The new result receives a ``derived_from``
        edge and the operation receives an input link.
    operation_type : str, optional
        Operation vocabulary value. Defaults to ``"analysis"``.
    parameters : dict, optional
        Mapping from parameter name to scalar value or a dictionary with
        ``value``, ``error``, ``fixed``, ``bounds``, and ``units``.
    metadata : dict, optional
        JSON metadata stored on both artifact and operation.
    data_format : str, optional
        Format hint. File payloads infer this from the suffix when omitted.
    setup_id : str, optional
        MFDB setup identifier associated with the operation.
    setup_version : int, optional
        Setup version stored in metadata for traceability.
    db : MFDBClientBase, optional
        Explicit database. Hot paths should pass one database or call
        ``set_global_db`` once; otherwise the registry opens the resolved user
        database as a fallback.
    is_public : bool, default=False
        Whether the artifact is visible to all users (public) or only
        the owning user (private).
    session : SessionContext, optional
        Resolved identity/session (PRD-17). When provided, the artifact is owned
        by ``session.user_id``; otherwise the canonical default user is resolved.

    Returns
    -------
    str
        Created artifact identifier, or ``""`` when no database is available
        or registration fails.
    """
    if db is None:
        db = _get_global_db()
    if db is None:
        logger.warning("register_result: no MFDB available; result not registered (kind=%s)", kind)
        return ""

    artifact_id = str(uuid.uuid4())
    operation_id = str(uuid.uuid4())
    object_uuid: str | None = None
    storage_mode = "embedded_json"
    size_bytes: int | None = None
    checksum: str | None = None
    mime_type: str | None = None
    effective_metadata = dict(metadata or {})
    if setup_id:
        effective_metadata.setdefault("setup_id", setup_id)
    if setup_version is not None:
        effective_metadata.setdefault("setup_version", setup_version)

    try:
        _validate_links(db, sample_id=sample_id, parent_artifact_id=parent_artifact_id)
        if parameters:
            from chisurf.core.mfdb.operation_parameters import validate_operation_parameters

            conn = getattr(db, "conn", None)
            if conn is not None:
                # No-op unless this operation type has a declared .dic schema.
                validate_operation_parameters(conn, operation_type or "analysis", parameters)
        with db.transaction():
            if data is not None:
                object_uuid, storage_mode, data_format, size_bytes, checksum, mime_type = _store_data(
                    db,
                    kind,
                    data,
                    data_format,
                    metadata=effective_metadata,
                )

            user_id = session.user_id if session is not None else _resolve_active_user_id()
            # Ensure the active user exists before stamping ownership; a
            # configured default_user_id that was never seeded would otherwise
            # fail the created_by_user_id foreign key and roll back the whole
            # registration (silent data loss).
            ensure_user = getattr(db, "ensure_user", None)
            if user_id and callable(ensure_user):
                ensure_user(user_id)
            db.register_artifact(
                artifact_id=artifact_id,
                artifact_kind=kind,
                data_format=data_format or None,
                storage_mode=storage_mode,
                object_uuid=object_uuid,
                size_bytes=size_bytes,
                checksum=checksum,
                checksum_algorithm="md5" if checksum else "sha256",
                mime_type=mime_type,
                metadata=effective_metadata or None,
                created_by_user_id=user_id,
                is_public=is_public,
            )
            # Record the creator in the many-to-many owner set.
            add_owner = getattr(db, "add_artifact_owner", None)
            if user_id and callable(add_owner):
                add_owner(artifact_id, user_id)

            op_type = operation_type or "analysis"
            db.record_operation(
                operation_id=operation_id,
                operation_type=op_type,
                setup_id=setup_id or None,
                status="succeeded",
                metadata=effective_metadata or None,
            )

            db.record_operation_link(
                operation_id=operation_id,
                artifact_id=artifact_id,
                direction="output",
                role=kind,
            )

            if parent_artifact_id:
                db.record_operation_link(
                    operation_id=operation_id,
                    artifact_id=parent_artifact_id,
                    direction="input",
                    role="source",
                )
                db.add_edge(
                    source_node_type="artifact",
                    source_node_id=artifact_id,
                    target_node_type="artifact",
                    target_node_id=parent_artifact_id,
                    relationship_type="derived_from",
                )

            if sample_id:
                db.link_artifact_to_sample(artifact_id, sample_id)

            if parameters:
                _record_parameters(db, operation_id, parameters)

    except (LinkValidationError, OperationParameterError) as exc:
        # A bad link target or a parameter set that violates the operation type's
        # declared schema. Both are checked before the transaction, so nothing was
        # persisted. Caller-recoverable conditions (bad input), not silent data
        # loss, so surface without the alarming error-level traceback and re-raise.
        logger.warning("register_result: %s (kind=%s); nothing registered", exc, kind)
        raise
    except Exception as exc:
        # db is not None (we already returned "" above for that case), so this
        # is a real integration error — FK violation, vocab rejection, etc.
        # Surface it loudly instead of silently dropping data.
        logger.error("register_result failed (kind=%s): %s", kind, exc, exc_info=True)
        raise

    logger.info("Registered result: kind=%s artifact=%s operation=%s", kind, artifact_id, operation_type or "analysis")
    return artifact_id


def set_global_db(db: MFDBClientBase | None) -> None:
    """Set or clear the process-local MFDB override.

    Parameters
    ----------
    db : MFDBClientBase or None
        Database to use for later ``register_result`` calls. Passing ``None``
        clears the override.

    Returns
    -------
    None
        This function mutates module state only.
    """
    global _GLOBAL_DB
    _GLOBAL_DB = db


def register_operation(
    operation_type: str,
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    parameters: dict | None = None,
    setup_id: str = "",
    status: str = "succeeded",
    metadata: dict | None = None,
    db: MFDBClientBase | None = None,
    validate: bool = True,
) -> str:
    """Record a data operation as a node (the uniform PRD-11 contract).

    Records the operation, its typed input/output ports
    (``mfdb_operation_artifact``), ``derived_from`` edges (each output ← each
    input), and its parameters — validated against the operation type's `.dic`
    schema (``mfdb_operation_parameter_def``) when ``validate`` is set. Parameters
    may be scalar, rich dicts, or a list for a repeatable (role-indexed) parameter.

    Returns the operation_id, or ``""`` when MFDB is unavailable.
    """
    db = db if db is not None else _get_global_db()
    if db is None:
        logger.warning("register_operation: no MFDB available (operation_type=%s)", operation_type)
        return ""

    if validate and parameters:
        from chisurf.core.mfdb.operation_parameters import validate_operation_parameters

        conn = getattr(db, "conn", None)
        if conn is not None:
            validate_operation_parameters(conn, operation_type, parameters)

    operation_id = str(uuid.uuid4())
    inputs = list(inputs or [])
    outputs = list(outputs or [])
    try:
        with db.transaction():
            db.record_operation(
                operation_id=operation_id,
                operation_type=operation_type,
                setup_id=setup_id or None,
                status=status,
                metadata=metadata or None,
            )
            for artifact_id in inputs:
                db.record_operation_link(
                    operation_id=operation_id,
                    artifact_id=artifact_id,
                    direction="input",
                    role="source",
                )
            for artifact_id in outputs:
                db.record_operation_link(
                    operation_id=operation_id,
                    artifact_id=artifact_id,
                    direction="output",
                    role="result",
                )
                for source_id in inputs:
                    db.add_edge(
                        source_node_type="artifact",
                        source_node_id=artifact_id,
                        target_node_type="artifact",
                        target_node_id=source_id,
                        relationship_type="derived_from",
                    )
            if parameters:
                _record_parameters(db, operation_id, parameters)
    except OperationParameterError:
        raise
    except Exception as exc:
        logger.error("register_operation failed (operation_type=%s): %s", operation_type, exc, exc_info=True)
        raise
    logger.info("Registered operation: type=%s op=%s inputs=%d outputs=%d", operation_type, operation_id, len(inputs), len(outputs))
    return operation_id


def register_raw_measurement(
    file_path: str,
    sample_id: str = "",
    metadata: dict | None = None,
    setup_id: str = "",
    setup_version: int | None = None,
    db: MFDBClientBase | None = None,
    is_public: bool = False,
    session: "SessionContext | None" = None,
) -> str:
    """Register a raw measurement file.

    Parameters
    ----------
    file_path : str
        Path to a raw TTTR, SPC, BH, or related measurement file.
    sample_id : str, optional
        Existing sample identifier to link.
    metadata : dict, optional
        Metadata for the artifact and import operation.
    setup_id : str, optional
        MFDB setup identifier associated with the measurement import.
    setup_version : int, optional
        Setup version stored in metadata for traceability.
    db : MFDBClientBase, optional
        Explicit database connection.
    is_public : bool, default=False
        Whether the raw measurement artifact is visible to all users.

    Returns
    -------
    str
        Created artifact identifier, or ``""`` on optional MFDB failure.
    """
    return register_result(
        kind="raw_measurement",
        data=file_path,
        sample_id=sample_id,
        operation_type="measurement_import",
        metadata=metadata,
        setup_id=setup_id,
        setup_version=setup_version,
        db=db,
        is_public=is_public,
        session=session,
    )


def register_processed_data(
    data: Any,
    parent_artifact_id: str,
    sample_id: str = "",
    operation_type: str = "analysis",
    parameters: dict | None = None,
    metadata: dict | None = None,
    setup_id: str = "",
    setup_version: int | None = None,
    db: MFDBClientBase | None = None,
    session: "SessionContext | None" = None,
) -> str:
    """Register processed data derived from another artifact.

    Parameters
    ----------
    data : Any
        Processed payload accepted by ``register_result``.
    parent_artifact_id : str
        Source artifact identifier.
    sample_id : str, optional
        Existing sample identifier to link.
    operation_type : str, optional
        Operation vocabulary value.
    parameters : dict, optional
        Processing parameters to record.
    metadata : dict, optional
        Metadata for the artifact and operation.
    setup_id : str, optional
        MFDB setup identifier associated with the operation.
    setup_version : int, optional
        Setup version stored in metadata for traceability.
    db : MFDBClientBase, optional
        Explicit database connection.

    Returns
    -------
    str
        Created artifact identifier, or ``""`` on optional MFDB failure.
    """
    return register_result(
        kind="processed_data",
        data=data,
        sample_id=sample_id,
        parent_artifact_id=parent_artifact_id,
        operation_type=operation_type,
        parameters=parameters,
        metadata=metadata,
        setup_id=setup_id,
        setup_version=setup_version,
        db=db,
        session=session,
    )


def register_fit_result(
    fit_data: dict,
    parent_artifact_id: str,
    sample_id: str = "",
    parameters: dict | None = None,
    metadata: dict | None = None,
    db: MFDBClientBase | None = None,
) -> str:
    """Register a fit result.

    Parameters
    ----------
    fit_data : dict
        Fit payload to serialize.
    parent_artifact_id : str
        Artifact that was fit.
    sample_id : str, optional
        Existing sample identifier to link.
    parameters : dict, optional
        Fit parameters to record.
    metadata : dict, optional
        Metadata for the artifact and operation.
    db : MFDBClientBase, optional
        Explicit database connection.

    Returns
    -------
    str
        Created artifact identifier, or ``""`` on optional MFDB failure.
    """
    return register_result(
        kind="fit_result",
        data=fit_data,
        sample_id=sample_id,
        parent_artifact_id=parent_artifact_id,
        operation_type="local_fit",
        parameters=parameters,
        metadata=metadata,
        db=db,
    )


def register_calibration(
    data: Any,
    calibration_type: str,
    sample_id: str = "",
    parent_artifact_id: str = "",
    parameters: dict | None = None,
    method: str = "",
    notes: str = "",
    db: MFDBClientBase | None = None,
) -> str:
    """Register a calibration result.

    Parameters
    ----------
    data : Any
        Calibration payload accepted by ``register_result``.
    calibration_type : str
        Calibration class, for example ``"g_factor"`` or ``"gamma"``.
    sample_id : str, optional
        Existing sample identifier to link.
    parent_artifact_id : str, optional
        Source artifact when calibration was derived from a measurement.
    parameters : dict, optional
        Calibration parameters to record.
    method : str, optional
        Method label. Use ``"user_provided"`` for literature or manually
        entered values without a source artifact.
    notes : str, optional
        Notes or citation text.
    db : MFDBClientBase, optional
        Explicit database connection.

    Returns
    -------
    str
        Created artifact identifier, or ``""`` on optional MFDB failure.
    """
    meta = {"calibration_type": calibration_type}
    if method:
        meta["method"] = method
    if notes:
        meta["notes"] = notes
    return register_result(
        kind="calibration_data",
        data=data,
        sample_id=sample_id,
        parent_artifact_id=parent_artifact_id,
        operation_type="calibration",
        parameters=parameters,
        metadata=meta,
        db=db,
    )


def read_result(db: MFDBClientBase, artifact_id: str) -> Any:
    """Read a registered result payload.

    Parameters
    ----------
    db : MFDBClientBase
        Database connection.
    artifact_id : str
        Artifact identifier to read.

    Returns
    -------
    Any
        Decoded typed payload for ``msgpack`` artifacts, raw object bytes for
        object-backed non-msgpack artifacts, or ``None`` when there is no
        payload.
    """
    artifact = db.get_artifact(artifact_id)
    if artifact is None:
        raise KeyError(f"Unknown artifact_id {artifact_id!r}")
    object_uuid = artifact.get("object_uuid")
    if object_uuid is None:
        return None
    blob = db.get_object(object_uuid)
    if artifact.get("data_format") == "msgpack":
        from chisurf.core.mfdb.payload_codec import REGISTRY, PayloadSchemaError, decode_payload

        payload = decode_payload(blob)
        artifact_kind = artifact.get("artifact_kind")
        payload_kind = getattr(payload, "KIND", None)
        if artifact_kind in REGISTRY and artifact_kind != payload_kind:
            raise PayloadSchemaError(
                f"artifact kind {artifact_kind!r} does not match payload kind {payload_kind!r}"
            )
        return payload
    return blob


def _get_global_db() -> MFDBClientBase | None:
    """Resolve the active MFDB connection.

    Parameters
    ----------
    None
        Resolution uses module state, the database connector singleton, and
        the configured user database path.

    Returns
    -------
    MFDBClientBase or None
        Active database, or ``None`` when the environment cannot provide one.
    """
    if _GLOBAL_DB is not None:
        return _GLOBAL_DB

    try:
        from chisurf.plugins.core.database_connector.services import _connector

        if getattr(_connector, "_db", None) is not None:
            return _connector._db
    except Exception:
        pass

    try:
        from chisurf.core.mfdb.database_resolver import resolve_database_path
        from chisurf.core.mfdb.repository import MFDatabase

        return MFDatabase(resolve_database_path())
    except Exception as exc:
        logger.warning("result_registry: could not open MFDB: %s", exc)
        return None


def _validate_links(db: MFDBClientBase, sample_id: str, parent_artifact_id: str) -> None:
    """Validate optional provenance links before writing payload objects.

    Parameters
    ----------
    db : MFDBClientBase
        Database wrapper.
    sample_id : str
        Optional sample identifier.
    parent_artifact_id : str
        Optional parent artifact identifier.

    Returns
    -------
    None
        Raises :class:`LinkValidationError` when a requested link target does
        not exist. Validation runs before any rows are written, so a failure
        leaves no partial artifact/object rows behind.
    """
    if parent_artifact_id and db.get_artifact(parent_artifact_id) is None:
        raise LinkValidationError(f"Unknown parent_artifact_id {parent_artifact_id!r}")
    if sample_id and not db.sample_exists(sample_id):
        raise LinkValidationError(f"Unknown sample_id {sample_id!r}")


def _store_data(
    db: MFDBClientBase,
    kind: str,
    data: str | bytes | Path | pandas.DataFrame | dict,
    data_format: str,
    metadata: dict | None = None,
) -> tuple[str, str, str, int | None, str | None, str | None]:
    """Store a payload in the object store.

    Parameters
    ----------
    db : MFDBClientBase
        Database wrapper exposing ``put_object``.
    kind : str
        MFDB artifact kind requested by the caller.
    data : str, bytes, Path, pandas.DataFrame, or dict
        Payload to store.
    data_format : str
        Optional caller-supplied data format.
    metadata : dict, optional
        Metadata copied into the payload envelope for structured payloads.

    Returns
    -------
    tuple
        ``(object_uuid, storage_mode, data_format, size_bytes, checksum,
        mime_type)``.
    """
    try:
        import pandas as pd
    except Exception:
        pd = None

    if isinstance(data, (str, Path)):
        path = Path(data)
        fmt = data_format or path.suffix.lstrip(".") or "unknown"
        ref = db.put_object(path=path, filename=path.name)
        return ref["object_uuid"], "local_file", fmt, ref["size_bytes"], ref["content_md5"], None

    if isinstance(data, bytes):
        ref = db.put_object(data=data, filename="payload.bin")
        return ref["object_uuid"], "embedded_blob", data_format or "bin", ref["size_bytes"], ref["content_md5"], None

    payload_kind, payload = _coerce_payload(kind, data, pandas_module=pd)
    from chisurf.core.mfdb.payload_codec import encode_payload

    blob, fmt = encode_payload(payload_kind, payload, meta=metadata or {})
    ref = db.put_object(data=blob, filename=f"{payload_kind}.msgpack", mime_type="application/msgpack")
    return ref["object_uuid"], "embedded_blob", fmt, ref["size_bytes"], ref["content_md5"], "application/msgpack"

    raise TypeError(f"Unsupported data type for register_result: {type(data)!r}")


def _coerce_payload(kind: str, data: Any, pandas_module: Any = None) -> tuple[str, Any]:
    """Convert structured plugin data to a typed payload model.

    Parameters
    ----------
    kind : str
        Requested artifact kind.
    data : Any
        Structured plugin data.
    pandas_module : Any, optional
        Imported pandas module, if available.

    Returns
    -------
    tuple
        ``(payload_kind, payload_object)`` for ``encode_payload``.
    """
    from chisurf.core.mfdb.payload_codec import REGISTRY
    from chisurf.core.mfdb.payload_models import BurstTable, GenericCurve, GenericTable

    if is_dataclass(data) and getattr(data, "KIND", None):
        if kind in REGISTRY and data.KIND != kind:
            raise ValueError(f"artifact kind {kind!r} does not match payload kind {data.KIND!r}")
        return data.KIND, data

    if pandas_module is not None and isinstance(data, pandas_module.DataFrame):
        if kind == BurstTable.KIND:
            return BurstTable.KIND, BurstTable.from_dataframe(data)
        if kind in REGISTRY:
            return kind, _payload_from_dataframe(kind, data)
        return GenericTable.KIND, GenericTable.from_dataframe(data)

    if _looks_like_data_curve(data):
        return GenericCurve.KIND, GenericCurve.from_data_curve(data)

    if isinstance(data, dict):
        if kind in REGISTRY:
            return kind, data
        if {"x", "y"} <= set(data):
            return GenericCurve.KIND, GenericCurve(
                x=np.asarray(data["x"], dtype=np.float64),
                y=np.asarray(data["y"], dtype=np.float64),
                ex=np.asarray(data["ex"], dtype=np.float64) if data.get("ex") is not None else None,
                ey=np.asarray(data["ey"], dtype=np.float64) if data.get("ey") is not None else None,
                mask=np.asarray(data["mask"], dtype=bool) if data.get("mask") is not None else None,
            )
        return GenericTable.KIND, GenericTable.from_mapping(data)

    raise TypeError(f"Unsupported structured payload type for register_result: {type(data)!r}")


def _payload_from_dataframe(kind: str, df: Any) -> Any:
    """Convert supported DataFrame layouts to matching typed payloads.

    Parameters
    ----------
    kind : str
        Requested payload kind.
    df : pandas.DataFrame
        Source tabular data.

    Returns
    -------
    Any
        Payload object whose ``KIND`` matches ``kind``.
    """
    from chisurf.core.mfdb.payload_models import (
        BurstSelection,
        FcsCorrelation,
        Spectrum,
        TcspcDecay,
        TttrPhotonStream,
    )

    if kind == FcsCorrelation.KIND and {"lag", "correlation"} <= set(df.columns):
        return FcsCorrelation(
            lag=_df_array(df, "lag", np.float64),
            correlation=_df_array(df, "correlation", np.float64),
            error=_df_optional_array(df, "error", np.float64),
            weights=_df_optional_array(df, "weights", np.float64),
        )
    if kind == Spectrum.KIND and {"wavelength", "intensity"} <= set(df.columns):
        return Spectrum(
            wavelength=_df_array(df, "wavelength", np.float64),
            intensity=_df_array(df, "intensity", np.float64),
            spectrum_type=_df_scalar(df, "spectrum_type", default="unknown"),
            normalized=_df_optional_bool(df, "normalized"),
        )
    if kind == TcspcDecay.KIND and {"time", "counts"} <= set(df.columns):
        return TcspcDecay(
            time=_df_array(df, "time", np.float64),
            counts=_df_array(df, "counts", np.int64),
            irf=_df_optional_array(df, "irf", np.float64),
            channel=_df_optional_scalar(df, "channel"),
            adc_resolution_ns=_df_optional_float(df, "adc_resolution_ns"),
            micro_time_resolution_ns=_df_optional_float(df, "micro_time_resolution_ns"),
        )
    if kind == BurstSelection.KIND:
        return BurstSelection(
            burst_ids=_df_optional_array(df, "burst_ids", np.int64, aliases=("burst_id",)),
            start_indices=_df_optional_array(df, "start_indices", np.uint64, aliases=("start_index", "start")),
            stop_indices=_df_optional_array(df, "stop_indices", np.uint64, aliases=("stop_index", "stop")),
            mask=_df_optional_bool_array(df, "mask"),
            labels=_df_optional_str_list(df, "labels", aliases=("label",)),
        )
    if kind == TttrPhotonStream.KIND and {"macro_times", "micro_times", "routing_channels"} <= set(df.columns):
        return TttrPhotonStream(
            macro_times=_df_array(df, "macro_times", np.uint64),
            micro_times=_df_array(df, "micro_times", np.uint32),
            routing_channels=_df_array(df, "routing_channels", np.uint16),
            event_types=_df_optional_array(df, "event_types", np.uint8),
            macro_time_resolution_s=_df_scalar_float(df, "macro_time_resolution_s"),
            micro_time_resolution_s=_df_scalar_float(df, "micro_time_resolution_s"),
            vendor_format=_df_optional_scalar(df, "vendor_format"),
            record_type=_df_optional_scalar(df, "record_type"),
            container_type=_df_optional_scalar(df, "container_type"),
        )
    raise ValueError(f"DataFrame columns cannot be coerced to payload kind {kind!r}")


def _column_name(df: Any, name: str, aliases: tuple[str, ...] = ()) -> str | None:
    for candidate in (name, *aliases):
        if candidate in df.columns:
            return candidate
    return None


def _df_array(df: Any, name: str, dtype: Any, aliases: tuple[str, ...] = ()) -> np.ndarray:
    column = _column_name(df, name, aliases)
    if column is None:
        raise ValueError(f"missing DataFrame column {name!r}")
    return np.asarray(df[column].to_numpy(copy=True), dtype=dtype)


def _df_optional_array(df: Any, name: str, dtype: Any, aliases: tuple[str, ...] = ()) -> np.ndarray | None:
    column = _column_name(df, name, aliases)
    if column is None:
        return None
    return np.asarray(df[column].to_numpy(copy=True), dtype=dtype)


def _df_optional_bool_array(df: Any, name: str, aliases: tuple[str, ...] = ()) -> np.ndarray | None:
    column = _column_name(df, name, aliases)
    if column is None:
        return None
    return np.asarray([_parse_bool(value, column) for value in df[column].to_numpy(copy=True)], dtype=bool)


def _df_unique_values(df: Any, name: str) -> list[Any]:
    if name not in df.columns:
        return []
    values = [value for value in df[name].to_numpy(copy=True) if not _is_missing(value)]
    unique = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return unique


def _df_scalar(df: Any, name: str, default: str) -> str:
    values = _df_unique_values(df, name)
    if not values:
        return default
    if len(values) != 1:
        raise ValueError(f"DataFrame column {name!r} must contain one unique scalar value")
    return str(values[0])


def _df_optional_scalar(df: Any, name: str) -> str | None:
    values = _df_unique_values(df, name)
    if not values:
        return None
    if len(values) != 1:
        raise ValueError(f"DataFrame column {name!r} must contain one unique scalar value")
    return str(values[0])


def _df_scalar_float(df: Any, name: str) -> float:
    values = _df_unique_values(df, name)
    if len(values) != 1:
        raise ValueError(f"DataFrame column {name!r} must contain one unique scalar value")
    return float(values[0])


def _df_optional_float(df: Any, name: str) -> float | None:
    values = _df_unique_values(df, name)
    if not values:
        return None
    if len(values) != 1:
        raise ValueError(f"DataFrame column {name!r} must contain one unique scalar value")
    return float(values[0])


def _df_optional_bool(df: Any, name: str) -> bool | None:
    values = _df_unique_values(df, name)
    if not values:
        return None
    if len(values) != 1:
        raise ValueError(f"DataFrame column {name!r} must contain one unique scalar value")
    return _parse_bool(values[0], name)


def _parse_bool(value: Any, name: str) -> bool:
    """Parse boolean table metadata without Python string truthiness.

    Parameters
    ----------
    value : Any
        Candidate boolean value from a DataFrame.
    name : str
        Column name used for diagnostics.

    Returns
    -------
    bool
        Parsed boolean value.
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        if value in (0, 1):
            return bool(value)
    if isinstance(value, (float, np.floating)):
        if value in (0.0, 1.0):
            return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "t", "yes", "y", "1"}:
            return True
        if normalized in {"false", "f", "no", "n", "0"}:
            return False
    raise ValueError(f"DataFrame column {name!r} has ambiguous boolean value {value!r}")


def _df_optional_str_list(df: Any, name: str, aliases: tuple[str, ...] = ()) -> list[str] | None:
    column = _column_name(df, name, aliases)
    if column is None:
        return None
    return [str(value) for value in df[column].to_numpy(copy=True)]


def _is_missing(value: Any) -> bool:
    if value is None or type(value).__name__ == "NAType":
        return True
    try:
        return bool(np.isscalar(value) and np.isnan(value))
    except (TypeError, ValueError):
        return False


def _looks_like_data_curve(data: Any) -> bool:
    """Return whether an object has the DataCurve attributes needed for archival.

    Parameters
    ----------
    data : Any
        Candidate object.

    Returns
    -------
    bool
        ``True`` when the object exposes ``x`` and ``y`` curve arrays.
    """
    return hasattr(data, "x") and hasattr(data, "y") and not isinstance(data, (dict, str, bytes, Path))


def _record_parameters(db: MFDBClientBase, operation_id: str, parameters: dict[str, Any]) -> None:
    """Write parameters to ``mfdb_parameter``.

    Parameters
    ----------
    db : MFDBClientBase
        Database wrapper exposing ``record_parameter``.
    operation_id : str
        Operation identifier that owns the parameters.
    parameters : dict
        Mapping of parameter names to scalar values, rich parameter dicts, or a
        **list** of entries for a repeatable (role-indexed) parameter — each list
        entry being a scalar (role = its index) or a dict that may carry a ``role``
        (the data-side analog of chinet's multiple ports, PRD-11).

    Returns
    -------
    None
        Parameters are written through repository methods.
    """
    for name, value in parameters.items():
        if isinstance(value, (list, tuple)):
            for i, entry in enumerate(value):
                role = str(entry.get("role")) if isinstance(entry, dict) and entry.get("role") is not None else str(i)
                _record_one_parameter(db, operation_id, str(name), entry, role=role)
            continue
        _record_one_parameter(db, operation_id, str(name), value, role=None)


def _record_one_parameter(
    db: MFDBClientBase, operation_id: str, name: str, value: Any, role: str | None
) -> None:
    """Record a single ``mfdb_parameter`` row, with an optional role index."""
    param_uuid = str(uuid.uuid4())
    if isinstance(value, dict):
        bounds = value.get("bounds") or [None, None]
        lower_bound, upper_bound = _coerce_bounds(bounds)
        db.record_parameter(
            parameter_uuid=param_uuid,
            operation_id=operation_id,
            name=name,
            value=_coerce_float(value.get("value")),
            standard_error=_coerce_float(value.get("error")),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            bounds_on=lower_bound is not None or upper_bound is not None,
            units=value.get("units"),
            parameter_type="fixed" if value.get("fixed") else "free",
            role=role,
            metadata={k: v for k, v in value.items() if k not in {"value", "error", "bounds", "units", "fixed", "role"}},
        )
        return

    db.record_parameter(
        parameter_uuid=param_uuid,
        operation_id=operation_id,
        name=name,
        value=_coerce_float(value),
        parameter_type="fixed",
        role=role,
        metadata=None if isinstance(value, (int, float)) else {"raw_value": value},
    )


def _coerce_bounds(bounds: Any) -> tuple[float | None, float | None]:
    """Return a normalized two-value numeric bounds tuple.

    Parameters
    ----------
    bounds : Any
        Candidate bounds sequence.

    Returns
    -------
    tuple
        Lower and upper bound values, each either ``float`` or ``None``.
    """
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        return None, None
    return _coerce_float(bounds[0]), _coerce_float(bounds[1])


def _coerce_float(value: Any) -> float | None:
    """Convert numeric values to float.

    Parameters
    ----------
    value : Any
        Candidate numeric value.

    Returns
    -------
    float or None
        Float value for ints and floats except booleans; otherwise ``None``.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None
